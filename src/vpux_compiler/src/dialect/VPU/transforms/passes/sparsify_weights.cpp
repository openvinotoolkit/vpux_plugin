//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/sparsity_strategy.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {

//
// SparsifyWeightsPass
//

class SparsifyWeightsPass final : public VPU::SparsifyWeightsBase<SparsifyWeightsPass> {
public:
    explicit SparsifyWeightsPass(VPU::WeightsSparsityHeuristic heuristic, std::optional<double> manualThreshold,
                                 Logger log)
            : _heuristic(heuristic), _manualThreshold(manualThreshold) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    VPU::WeightsSparsityHeuristic _heuristic;
    std::optional<double> _manualThreshold;
};

//
// safeRunOnFunc
//

void SparsifyWeightsPass::safeRunOnFunc() {
    using namespace VPU::NCESparsity;

    auto func = getOperation();
    auto module = getOperation();
    auto& ctx = getContext();

    std::unique_ptr<BaseWeightsSparsityStrategy> enablementStrategy;
    if (_heuristic == VPU::WeightsSparsityHeuristic::CMX) {
        _log.trace("Using CMX-based heuristic");
        const Byte availableCMX = VPU::getTotalCMXSize(module);
        enablementStrategy = std::make_unique<CMXConsumptionBasedWeightsSparsityStrategy>(
                availableCMX, CMX_BASED_STRATEGY_DEFAULT_INTERVALS, _manualThreshold);
    } else if (_heuristic == VPU::WeightsSparsityHeuristic::RATIO) {
        _log.trace("Using ratio-based heuristic");
        enablementStrategy = std::make_unique<RatioBasedWeightsSparsityStrategy>(
                WEIGHTS_SPARSITY_FLOAT_RATIO_THRESHOLD, WEIGHTS_SPARSITY_INT_RATIO_THRESHOLD, _manualThreshold);
    } else {
        VPUX_THROW("Unsupported heuristic: {0}", _heuristic);
    }

    int64_t numCandidatesSparseWeights = 0;
    int64_t numSparsifiedWeights = 0;

    DenseMap<Const::DeclareOp, SmallVector<VPU::SparseOpInterface>> sparseCandidates;

    auto innerLog = _log.nest();

    // Walk the IR and find all sparse weights candidates
    func->walk([&](VPU::SparseOpInterface sparsifiableOp) {
        if (!VPU::supportsSparseWeights(sparsifiableOp.getOperation())) {
            return;
        }

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(sparsifiableOp.getOperation());
        VPUX_THROW_UNLESS(nceOp != nullptr, "Unexpected non-NCE operation that supports weights sparsity");

        const auto weights = nceOp.getWeightsOperand();
        if (weights == nullptr) {
            return;
        }
        auto weightsType = weights.getType().cast<vpux::NDTypeInterface>();
        if (weightsType.getElemTypeSize().count() < CHAR_BIT) {
            _log.trace("Op '{0}' at '{1}' is not supporting sparsity for sub 8-bit weights", sparsifiableOp->getName(),
                       sparsifiableOp->getLoc());
            return;
        }

        _log.trace("Op '{0}' at '{1}' is a candidate for sparsifying its weights", sparsifiableOp->getName(),
                   sparsifiableOp->getLoc());

        if (weights.getType().isa<VPU::SparseTensorType>()) {
            innerLog.trace("Weights are already sparse");
            return;
        }

        auto weightsOp = weights.getDefiningOp<Const::DeclareOp>();
        if (weightsOp == nullptr) {
            innerLog.trace("Expected weights parent to be constant, but got '{0}'", weights.getDefiningOp()->getName());
            return;
        }

        if (mlir::isa<vpux::VPU::NCECompressConvolutionOp>(sparsifiableOp)) {
            innerLog.trace("Operation uses the compressed convolution feature. Skipping");
            return;
        }

        sparseCandidates[weightsOp].push_back(sparsifiableOp);
        ++numCandidatesSparseWeights;
    });

    DenseMap<Const::DeclareOp, VPU::GroupSparseTensorOp> localReplacementCache;
    // poor man's way to only create a GroupSparseTensor once. done this way to
    // limit the amount of changes around multi-threaded code.
    const auto getCachedSparseTensorOp = [&](Const::DeclareOp weightsOp, const Const::ContentSetup& newContentAttrSetup,
                                             VPU::SparsityCompressionAttr sparsityCompressionAttr) {
        auto it = localReplacementCache.find(weightsOp);
        if (it != localReplacementCache.end()) {
            return it->second;
        }

        mlir::OpBuilder builder(weightsOp);
        auto sparsityMapContent = newContentAttrSetup.clone().getSparsityMap().get();
        auto sparsifiedContent = newContentAttrSetup.clone().sparsify(false).get();
        const auto sparsifiedWeights = builder.create<Const::DeclareOp>(weightsOp.getLoc(), sparsifiedContent.getType(),
                                                                        std::move(sparsifiedContent));
        const auto sparsityMap = builder.create<Const::DeclareOp>(weightsOp.getLoc(), sparsityMapContent.getType(),
                                                                  std::move(sparsityMapContent));
        auto groupedView =
                builder.create<VPU::GroupSparseTensorOp>(weightsOp.getLoc(), sparsifiedWeights->getResult(0),
                                                         sparsityMap->getResult(0), true, sparsityCompressionAttr);

        it = localReplacementCache.insert({weightsOp, groupedView}).first;
        return it->second;
    };

    std::mutex irModificationMutex;

    // In parallel, count sparse elements in sparse candidates and decide which ones should be made sparse
    const auto tryToSparsify = [&](VPU::SparseOpInterface sparsifiableOp, Const::DeclareOp weightsOp,
                                   const Const::Content& foldedContent) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(sparsifiableOp.getOperation());
        const auto weights = nceOp.getWeightsOperand();
        auto weightsType = weights.getType().cast<vpux::NDTypeInterface>();

        const auto foldedElemType = foldedContent.getType().getElementType();
        const auto inputType = sparsifiableOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto hasFloatInput = inputType.getElementType().isa<mlir::FloatType>();
        const auto numNonSparseElemsPerOC = vpux::countNonSparseElementsPerOC(foldedContent, foldedElemType);
        if (!enablementStrategy->shouldSparsifyWeights(innerLog, weightsType, numNonSparseElemsPerOC, hasFloatInput)) {
            innerLog.trace("Weights will not be sparsified", sparsifiableOp->getName(), sparsifiableOp->getLoc());
            return;
        }
        innerLog.trace("Sparsifying weights for op '{0}' at '{1}'", sparsifiableOp->getName(),
                       sparsifiableOp->getLoc());
        const auto numElemsType =
                mlir::RankedTensorType::get({static_cast<int64_t>(numNonSparseElemsPerOC.size())}, getInt64Type(&ctx));
        const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numNonSparseElemsPerOC));
        const auto axisAttr = getIntAttr(&ctx, Dims4D::Filter::OC.ind());
        const auto alignmentAttr = getIntAttr(&ctx, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
        const auto sparsityCompressionAttr =
                VPU::SparsityCompressionAttr::get(&ctx, axisAttr, numElemsAttr, alignmentAttr);

        // Fold the original constant to drop the original transformations
        // This is done in order to avoid repeating the folding that was done in this pass later in the compilation
        auto foldedContentType = foldedContent.getType();
        if (auto qType = foldedElemType.dyn_cast<mlir::quant::QuantizedType>()) {
            foldedContentType = foldedContentType.changeElemType(normalizeQuantStorageType(qType));
        }
        // It is necessary to copy the contents of the folded constant into a new buffer since the data conversion
        // does not take place until the values are extracted (e.g. inside `Const::Content::copyTo`)
        // For example, if the original constants are FP32, they will still occupy storage for FP32 elements after
        // folding until they are stored in a new buffer allocated for INT8 data and copied into it
        const auto contentSize = checked_cast<size_t>(foldedContentType.getTotalAllocSize().count());
        std::vector<char> newContent(contentSize);
        foldedContent.copyTo(MutableArrayRef(newContent.data(), contentSize));
        const auto foldedBaseContent = mlir::DenseElementsAttr::getFromRawBuffer(
                mlir::cast<mlir::ShapedType>(foldedContentType), ArrayRef(newContent));
        auto newContentAttrSetup = Const::ContentAttr::transform(foldedBaseContent);
        // Folded constants with INT8 element types have to be cast to quantized types for the correct type to be
        // inferred from the new Const::ContentAttr
        if (auto qType = foldedElemType.dyn_cast<mlir::quant::QuantizedType>()) {
            newContentAttrSetup = newContentAttrSetup.quantCast(qType);
        }

        // IR modification is not thread safe according to MLIR documentation.
        std::lock_guard<std::mutex> guard(irModificationMutex);

        VPU::GroupSparseTensorOp groupedView =
                getCachedSparseTensorOp(weightsOp, newContentAttrSetup, sparsityCompressionAttr);

        weightsOp->replaceUsesWithIf(groupedView, [useToReplace = sparsifiableOp.getOperation()](mlir::OpOperand& use) {
            return use.getOwner() == useToReplace;
        });
        if (weightsOp->getUses().empty()) {
            weightsOp->erase();
        }

        ++numSparsifiedWeights;
    };

    loop_1d(LoopExecPolicy::Parallel, &getContext(), sparseCandidates.size(), [&](const size_t index) {
        // Note: since we are folding here, doing a linear iterator increment is
        // likely cheap enough.
        auto it = sparseCandidates.begin();
        std::advance(it, index);

        auto [weightsOp, sparsifiableOps] = *it;
        const auto foldedContent = weightsOp.getContent();
        for (auto sparsifiableOp : sparsifiableOps) {
            tryToSparsify(sparsifiableOp, weightsOp, foldedContent);
        }
    });

    _log.trace("Sparsified weights for {0} operations out of {1} candidates", numSparsifiedWeights,
               numCandidatesSparseWeights);
}

}  // namespace

//
// createSparsifyWeightsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSparsifyWeightsPass(VPU::WeightsSparsityHeuristic heuristic,
                                                                 std::optional<double> manualThreshold, Logger log) {
    return std::make_unique<SparsifyWeightsPass>(heuristic, manualThreshold, log);
}
