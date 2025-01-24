//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/max_lstm_hidden_size_constant.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LSTMSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LSTMSequenceOpAdaptor lstm(operands, attrs, prop);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = lstm.getInputData().getType();
    const auto inDataShape = mlir::cast<vpux::NDTypeInterface>(inDataType).getShape();

    const auto initialHiddenStateType = mlir::cast<vpux::NDTypeInterface>(lstm.getInitialHiddenState().getType());
    const auto initialHiddenStateShape = initialHiddenStateType.getShape();
    const auto elementType = initialHiddenStateType.getElementType();
    const auto tensorAttr = createTensorAttrFromType(initialHiddenStateType);

    const auto batchSize = initialHiddenStateShape[Dims4D::Act::N];
    const auto numDirections = initialHiddenStateShape[Dims4D::Act::C];
    const auto hiddenSize = initialHiddenStateShape.back();

    const auto lengthIndex = inDataShape.size() - 2;
    int64_t sequenceLength = inDataShape[Dim(lengthIndex)];

    const SmallVector<int64_t> outputHiddenValuesShape{batchSize, numDirections, sequenceLength, hiddenSize};

    auto outputHiddenValuesType = mlir::RankedTensorType::get(outputHiddenValuesShape, elementType, tensorAttr);
    const auto outputHiddenStateType = mlir::RankedTensorType::get(initialHiddenStateShape, elementType, tensorAttr);
    const auto outputCellStateType = mlir::RankedTensorType::get(initialHiddenStateShape, elementType, tensorAttr);

    if (inDataShape.isStatic()) {
        inferredReturnTypes.push_back(outputHiddenValuesType);
    } else {
        const auto outputHVRank = outputHiddenValuesShape.size();
        auto outHVBounds = SmallVector<int64_t>(outputHVRank);
        const auto inDataBoundedType = mlir::cast<vpux::BoundedTypeInterface>(inDataType);

        for (size_t i = 0; i < outputHVRank; i++) {
            if (outputHiddenValuesShape[i] == mlir::ShapedType::kDynamic) {
                outHVBounds[i] = parseIntArrayAttr<int64_t>(inDataBoundedType.getBounds())[lengthIndex];
            } else {
                outHVBounds[i] = outputHiddenValuesShape[i];
            }
        }
        inferredReturnTypes.push_back(mlir::cast<vpux::BoundedTypeInterface>(outputHiddenValuesType)
                                              .changeBounds(getIntArrayAttr(ctx, outHVBounds)));
    }

    inferredReturnTypes.push_back(outputHiddenStateType);
    inferredReturnTypes.push_back(outputCellStateType);

    return mlir::success();
}

namespace {

static mlir::ModuleOp getModule(::mlir::OpBuilder& odsBuilder) {
    auto block = odsBuilder.getInsertionBlock();
    auto parentOp = block->getParentOp();
    while (parentOp && !llvm::isa<mlir::ModuleOp>(parentOp)) {
        parentOp = parentOp->getParentOp();
    }
    return llvm::cast<mlir::ModuleOp>(parentOp);
}

mlir::Value createSyncBuffer(mlir::OpBuilder& rewriter, ShapeRef shape) {
    const auto auxIndicesType = mlir::RankedTensorType::get(shape.raw(), getSInt32Type(rewriter.getContext()));
    return Const::createConst(rewriter,
                              appendLoc(mlir::UnknownLoc::get(rewriter.getContext()), "LSTMSequence_SyncBuffer"),
                              auxIndicesType, ArrayRef<int32_t>(0));
}

}  // namespace

void vpux::VPU::LSTMSequenceOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      ::mlir::Value inputData, ::mlir::Value initialHiddenState,
                                      ::mlir::Value initialCellState, ::mlir::Value reccurenceWeights,
                                      ::mlir::IntegerAttr sequenceLength, vpux::IE::RNNSequenceDirectionAttr direction,
                                      vpux::VPU::MultiClusterStrategyAttr multiClusterStrategy) {
    const auto module = getModule(odsBuilder);
    auto tileOp = IE::getTileExecutor(module);

    const auto numShavesPerTile = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT).getCount();
    const auto syncBuffer = createSyncBuffer(odsBuilder, Shape{1, 1, 1, numShavesPerTile});

    build(odsBuilder, odsState, inputData, initialHiddenState, initialCellState, reccurenceWeights, syncBuffer,
          sequenceLength, direction, multiClusterStrategy);
}

bool vpux::VPU::LSTMSequenceOp::isSupported(vpux::IE::LSTMSequenceOp op) {
    if (op.getReccurenceWeights().getDefiningOp<Const::DeclareOp>() == nullptr) {
        return false;
    }

    auto maxHiddenSize = getMaxLstmSequenceHiddenSizeConstant(VPU::getArch(op));

    // shave implementation allow reduced size. Bigger size can and are map on DPU.
    const auto initialHiddenStateShape = getShape(op.getInitialHiddenState());

    // shave asm implement just 16 element alignment hidden size. Except that, speed is low.
    constexpr int64_t alignmentRequired(16);
    if (initialHiddenStateShape.back() > maxHiddenSize) {
        return false;
    }
    if (initialHiddenStateShape.back() % alignmentRequired != 0) {
        return false;
    }
    return true;
}

//
// ClusteredOpInterface
//

bool vpux::VPU::LSTMSequenceOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inputShape = getShape(getInputData());
    const auto numDirections = inputShape[Dims4D::Act::C];

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && numDirections == 2) {
        return true;
    }

    const auto batchSize = inputShape[Dims4D::Act::N];
    return strategy == VPU::MultiClusterStrategy::SplitOverBatch && batchSize > 1;
}

bool VPU::LSTMSequenceOp::isOperationSplitOverKernelCompatible(ShapeRef, ShapeRef, ShapeRef) {
    const auto numDirections = getShape(getInputData())[Dims4D::Act::C];
    return numDirections == 2;
}

bool VPU::LSTMSequenceOp::isOperationSplitOverBatchCompatible(ShapeRef) {
    const auto batchSize = getShape(getInputData())[Dims4D::Act::N];
    return batchSize > 1;
}

vpux::VPU::DistributionInfo vpux::VPU::LSTMSequenceOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::LSTMSequenceOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::LSTMSequenceOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::LSTMSequenceOp::supportCycleCostCalculation() {
    return false;
}
