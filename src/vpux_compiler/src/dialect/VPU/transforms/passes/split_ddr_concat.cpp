//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPU;

namespace {  // namespace

class SplitConcatRewriter final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    SplitConcatRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::ConcatOp>(ctx, benefitHigh), _log(log) {
    }

    bool isLegalConcatPattern(VPU::ConcatOp concatOp, vpux::Logger log) const;
    bool isCopyOpFromCmxWithSingleUser(mlir::Operation* op, vpux::Logger log) const;
    bool areDistributedTypesConcatenable(VPU::DistributedTensorType firstType,
                                         VPU::DistributedTensorType secondType) const;
    bool areDistributionTypesConsistent(mlir::Value first, mlir::Value second) const;
    mlir::FailureOr<size_t> parentOpSize(mlir::Value concatInput) const;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

std::unordered_set<Dim> getConcatAxesFromOffsets(VPU::ConcatOpAdaptor concat, ShapeRef outShape) {
    std::unordered_set<Dim> res;

    for (const auto inVal : concat.getInputs()) {
        const auto curShape = getShape(inVal);

        for (const auto ind : irange(outShape.size())) {
            const auto d = Dim(ind);

            if (curShape[d] != outShape[d]) {
                res.insert(d);
            }
        }
    }

    return res;
}

bool SplitConcatRewriter::isCopyOpFromCmxWithSingleUser(mlir::Operation* op, vpux::Logger log) const {
    auto isCopyToDDR = [](VPU::CopyOp copyOp) -> bool {
        auto origOp = copyOp->getParentOfType<VPU::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                    : copyOp->getParentOp();
        return origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
    };

    auto isCopyFromCmx = [](VPU::CopyOp copyOp) -> bool {
        auto origOp = copyOp->getParentOfType<VPU::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                    : copyOp->getParentOp();
        return origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::CMX_NN;
    };

    auto isNotNCE = [&](mlir::Operation* op) -> bool {
        auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op);
        auto clusterTilingOp = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(op);
        if (clusterTilingOp != nullptr) {
            if (auto innerOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(clusterTilingOp.getInnerTaskOp())) {
                nceOp = innerOp;
            }
        }

        return nceOp == nullptr;
    };

    auto isOduPermuteOrNCEPermute = [&](mlir::Operation* op) -> bool {
        auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op);
        auto clusterTilingOp = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(op);
        if (clusterTilingOp != nullptr) {
            if (auto innerOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(clusterTilingOp.getInnerTaskOp())) {
                nceOp = innerOp;
            }
        }
        if (nceOp == nullptr) {
            return false;
        }
        auto outputLayout = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getDimsOrder();
        if (outputLayout != DimsOrder::NHWC) {
            return true;
        }

        if (mlir::isa<VPU::NCEPermuteOp>(nceOp)) {
            log.trace("Skip CMXConcat for NCEPermuteOp");
            return true;
        }
        return false;
    };

    auto isInPlace = [](mlir::Operation* op) -> bool {
        auto nceEltwiseOp = mlir::dyn_cast_or_null<VPU::NCEEltwiseOp>(op);
        auto clusterTilingOp = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(op);
        if (clusterTilingOp) {
            nceEltwiseOp = clusterTilingOp.getInnerTaskOpOfType<VPU::NCEEltwiseOp>();
        }
        if (nceEltwiseOp == nullptr) {
            return false;
        }
        if (nceEltwiseOp.getIsInplace()) {
            return true;
        }
        return false;
    };

    auto haveUsersNotOnlyCopyOps = [](mlir::Operation* op) -> bool {
        // avoid cmx concats which are complex, where the inputs to the concat would be used by other operations
        const auto userIsCopyOp = [](mlir::Operation* user) {
            if (mlir::isa<VPU::CopyOp>(user)) {
                return true;
            }

            auto clusterTiling = mlir::dyn_cast<VPU::NCEClusterTilingOp>(user);
            if (clusterTiling == nullptr) {
                return false;
            }

            return clusterTiling.getInnerTaskOpOfType<VPU::CopyOp>() != nullptr;
        };

        if (op == nullptr) {
            return false;
        }

        for (auto result : op->getResults()) {
            for (auto user : result.getUsers()) {
                if (!userIsCopyOp(user)) {
                    return true;
                }
            }
        }

        return false;
    };

    auto copyOp = mlir::dyn_cast_or_null<VPU::CopyOp>(op);
    auto clusterCopyOp = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(op);
    if (clusterCopyOp) {
        if (auto innerOp = mlir::dyn_cast<VPU::CopyOp>(clusterCopyOp.getInnerTaskOp())) {
            copyOp = innerOp;
        }
    }

    if (copyOp == nullptr) {
        log.trace("Concat input pattern does not contain a copy op");
        return false;
    }

    const bool isLegalCopyPattern = isCopyToDDR(copyOp) && isCopyFromCmx(copyOp) && op->getResult(0).hasOneUse();
    auto* parentOp = op->getOperand(0).getDefiningOp();

    return isLegalCopyPattern && !isOduPermuteOrNCEPermute(parentOp) && !isInPlace(parentOp) &&
           !haveUsersNotOnlyCopyOps(parentOp) && !isNotNCE(parentOp);
}

bool SplitConcatRewriter::isLegalConcatPattern(VPU::ConcatOp concatOp, vpux::Logger log) const {
    log.trace("Checking isLegalConcatPattern");

    if (concatOp.getOutput().getType().cast<vpux::NDTypeInterface>().getMemoryKind() != VPU::MemoryKind::DDR) {
        log.trace("SplitConcatRewriter: pattern only affecting concat ops in DDR '{0}'", concatOp->getLoc());
        return false;
    }

    // Check if the Concat op satisfies the CMX Concat conditions or not
    auto isSingleAxisConcat = [](mlir::ArrayAttr offset) {
        // If a concat has at least one static_offset attribute of 2 or more non-zero axis
        // it is considered as multiple-axis concat, vice versa
        // e.g., static_offset of a multiple-axis concat:
        // [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]
        auto offsetVector = parseIntArrayAttr<int64_t>(offset);
        return offsetVector.size() - std::count(offsetVector.begin(), offsetVector.end(), 0) <= 1;
    };

    if (concatOp.getStaticOffsetsAttr()) {
        if (!llvm::all_of(concatOp.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>(), isSingleAxisConcat)) {
            log.trace("DDR Concat operation is not single axis");
            return false;
        }
    }

    // check that the output of the concat operation doesn't fit in cmx
    const auto availableMem = VPU::getTotalCMXSize(concatOp);
    // TODO: review this check about concat size
    const uint32_t totConcatOutputsSize =
            concatOp.getOutput().getType().cast<vpux::NDTypeInterface>().getTotalAllocSize().count();
    if (totConcatOutputsSize < availableMem.count()) {
        log.trace("The pass only affects concat ops that cannot fit entirely in CMX");
        return false;
    }

    auto invalidPattern = llvm::none_of(concatOp.getInputs(), [&](auto input) {
        // if none of the inputs satisfies the required pattern then there's nothing to be optimised
        return isCopyOpFromCmxWithSingleUser(input.getDefiningOp(), log);
    });

    if (invalidPattern) {
        log.trace("SplitConcatRewriter: nothing to be optimised '{0}'", concatOp->getLoc());
    } else {
        log.trace("SplitConcatRewriter: Found legal Concat pattern at op '{0}'", concatOp->getLoc());
    }

    return !invalidPattern;
}

bool SplitConcatRewriter::areDistributedTypesConcatenable(VPU::DistributedTensorType firstType,
                                                          VPU::DistributedTensorType secondType) const {
    if (firstType.getOrder() != secondType.getOrder() || firstType.getMemSpace() != secondType.getMemSpace()) {
        return false;
    }

    // checks modes are compatible, num_clusters is the same and num_tiles is the same, if applicable
    if (mlir::failed(areDistributionAttrsCompatible(firstType, secondType,
                                                    /*allowDifferentPerClusterMemoryView = */ true))) {
        return false;
    }

    const auto areInOutShapesOffsetsCompatible = [&](const SmallVector<Shape>& lhs,
                                                     const SmallVector<Shape>& rhs) -> bool {
        for (const auto& pair : zip(lhs, rhs)) {
            const auto shapesOffsetsLhs = std::get<0>(pair);
            const auto shapesOffsetsRhs = std::get<1>(pair);

            if (shapesOffsetsLhs.size() != shapesOffsetsRhs.size()) {
                return false;
            }

            for (size_t idx = 0; idx < shapesOffsetsLhs.size(); idx++) {
                // If dim is not a concatenation axis, check that per cluster shapes/offsets are the same for
                // the input & output.
                // Since pass only allows CMX Concat on single axis, it can be assumed that concatenation axis
                // is the axis where the dims do not match.
                // When checking consistency for output pattern parts, the two shapes should be equal, therefore
                // full memory view is verified.
                const auto dim = Dim(idx);
                if (firstType.getShape()[dim] == secondType.getShape()[dim]) {
                    if (shapesOffsetsLhs[dim] != shapesOffsetsRhs[dim]) {
                        return false;
                    }
                }
            }
        }

        return true;
    };

    const auto firstPerClusterOffsets = firstType.getPerClusterMemoryShapeOffsets();
    const auto secondPerClusterOffsets = secondType.getPerClusterMemoryShapeOffsets();
    if (!areInOutShapesOffsetsCompatible(firstPerClusterOffsets, secondPerClusterOffsets)) {
        return false;
    }

    const auto firstPerClusterShapes = firstType.getPerClusterMemoryShapes();
    const auto secondPerClusterShapes = secondType.getPerClusterMemoryShapes();
    if (!areInOutShapesOffsetsCompatible(firstPerClusterShapes, secondPerClusterShapes)) {
        return false;
    }

    return true;
}

bool SplitConcatRewriter::areDistributionTypesConsistent(mlir::Value first, mlir::Value second) const {
    auto firstClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(first.getDefiningOp());
    auto secondClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(second.getDefiningOp());
    if (firstClusterOp == nullptr && secondClusterOp == nullptr) {
        // if both are not cluster ops it means they are both normal copy ops (due to the previous pattern check)
        return true;
    }

    const auto firstDistributedTypeInterfaceOutput =
            firstClusterOp->getOperand(0).getType().dyn_cast<VPU::DistributedTypeInterface>();
    const auto secondDistributedTypeInterfaceOutput =
            secondClusterOp->getOperand(0).getType().dyn_cast<VPU::DistributedTypeInterface>();

    if (firstDistributedTypeInterfaceOutput == nullptr && secondDistributedTypeInterfaceOutput == nullptr) {
        return true;
    }
    if (firstDistributedTypeInterfaceOutput == nullptr || secondDistributedTypeInterfaceOutput == nullptr) {
        _log.trace("Can't concatenate distributed tensor with ranked tensor");
        return false;
    }

    const bool firstIsDistributed = firstDistributedTypeInterfaceOutput.containsDistributedTypes();
    const bool secondIsDistributed = secondDistributedTypeInterfaceOutput.containsDistributedTypes();

    if (firstIsDistributed != secondIsDistributed) {
        _log.trace("Can't concatenate distributed tensor with ranked tensor");
        return false;
    }

    // Both distributed types
    const auto firstDistrType =
            firstDistributedTypeInterfaceOutput.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
    const auto secondDistrType =
            secondDistributedTypeInterfaceOutput.getDistributedTypes().front().cast<VPU::DistributedTensorType>();

    if (!areDistributedTypesConcatenable(firstDistrType, secondDistrType)) {
        _log.trace("Not matching distributed tensor attributes between concat inputs: `{0}` and `{1}`", firstDistrType,
                   secondDistrType);
        return false;
    }

    return true;
}

mlir::FailureOr<size_t> SplitConcatRewriter::parentOpSize(mlir::Value ddrConcatInput) const {
    // given the concatInput, if the parent graph is matching the isCopyOpFromCmxWithSingleUser() conditions,
    // the defining op is a copy op or cluster copy op. We want to check if we can remove the copy and
    // get the parent of the copy to be directly the input of a cmx concat, so we need to calculate
    // the size of the parent of the copy op. If the isCopyOpFromCmxWithSingleUser pattern is not respected,
    // this size is not considered anyway in the algorithm and the specific DDR concat input is kept in DDR

    auto* parentCopy = ddrConcatInput.getDefiningOp();
    if (parentCopy == nullptr) {
        return mlir::failure();
    }
    auto* nceParentOp = parentCopy->getOperand(0).getDefiningOp();
    if (nceParentOp == nullptr) {
        return mlir::failure();
    }

    size_t nceParentInputsSize = 0, nceParentOutputsSize = 0;
    for (auto nceParentInput : nceParentOp->getOperands()) {
        nceParentInputsSize += nceParentInput.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize().count();
    }
    for (auto nceParentOutput : nceParentOp->getResults()) {
        nceParentOutputsSize += nceParentOutput.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize().count();
    }

    return nceParentInputsSize + nceParentOutputsSize;
}

/*
Convert this subgraph:

           NCE        NCE        NCE        NCE
            |          |          |          |
          Copy        Copy       Copy       Copy
    Op1  (CMX2DDR)  (CMX2DDR)  (CMX2DDR)  (CMX2DDR)  Op2  Op3
     |      |          |          |          |        |    |
     \      \          \          |          /        /    /
                             Concat (DDR)

into:
        NCE   NCE   NCE   NCE
         |     |     |     |
         \     |     |     /
             Concat (CMX)
                  |
    Op1         Copy         Op2  Op3
     |        (CMX2DDR)       |    |
     \            |           /    /
              Concat (DDR)

*/

mlir::LogicalResult SplitConcatRewriter::matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!isLegalConcatPattern(origOp, _log)) {
        _log.trace("SplitConcatRewriter: Cannot split this concat Op");
        return mlir::failure();
    }

    const auto axis = getConcatAxesFromOffsets(origOp, getShape(origOp.getOutput()));
    if (axis.size() != 1) {
        _log.trace("concatOp is not single axis: '{0}'", origOp->getLoc());
        return mlir::failure();
    }
    const auto axisValue = *axis.begin();

    // Just as a first implementation reduce the available memory amount
    const auto availableMem = VPU::getTotalCMXSize(origOp);
    // TODO review this testing value (effectiveCmxTileMem is a reduced fraction of cmx tile size)
    const auto effectiveCmxTileMem = 1.0 * availableMem.count();
    // Only perform the optimisation when the single inputs are relatively small. The benefit should be maximum when
    // there is a considerable number of small inputs which would generate a lot of small dma copies to DDR, which get
    // optimised by this pass.
    // TODO: determine if there is an optimum value for maxConcatInputOptimizedSize and what it is
    const auto maxConcatInputOptimizedSize = effectiveCmxTileMem / 8;

    SmallVector<SmallVector<mlir::Value>> CopyOutputGroups;
    SmallVector<mlir::Value> currentSubvector;
    uint32_t currentSum = 0;

    const bool isConcatOverH = (axisValue.ind() == Dims4D::Act::H.ind());
    bool isSplitOverH = false;

    for (const auto& ddrConcatInputValue : origOp.getInputs()) {
        bool isPatternValid = isCopyOpFromCmxWithSingleUser(ddrConcatInputValue.getDefiningOp(), _log);
        bool inputParentFitsInCmx = false;
        bool currentSumFitInCmx = false;
        size_t concatInputParentOpSize{}, ddrConcatInputValueSize{};
        if (isPatternValid) {
            const auto res = parentOpSize(ddrConcatInputValue);
            if (mlir::failed(res)) {
                _log.trace("Problem in calculating parentOpSize");
                return mlir::failure();
            }

            concatInputParentOpSize = res.value();
            inputParentFitsInCmx = (concatInputParentOpSize <= effectiveCmxTileMem);
            currentSumFitInCmx = ((concatInputParentOpSize + currentSum) <= effectiveCmxTileMem);
            ddrConcatInputValueSize =
                    ddrConcatInputValue.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize().count();
        }

        if (!isPatternValid || !inputParentFitsInCmx || (ddrConcatInputValueSize > maxConcatInputOptimizedSize)) {
            // Check if the current value, and the parent op's inputs size exceed effectiveCmxTileMem,
            // which means that this concat input can't fit in cmx in any case.
            // Check also if the input respects the optimised pattern, otherwise it will be kept untouched as new
            // ddr concat op input
            if (!currentSubvector.empty()) {
                CopyOutputGroups.push_back(currentSubvector);
                currentSubvector.clear();
                currentSum = 0;
                isSplitOverH = false;
            }
            CopyOutputGroups.push_back(SmallVector<mlir::Value>{ddrConcatInputValue});
        } else {
            // The current input value respect the optimisable pattern (isPatternValid == true)
            auto isSingleOpSplitOnH = [](mlir::Value inVal) {
                auto clusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(inVal.getDefiningOp());
                if (clusterOp == nullptr) {
                    return false;
                }
                const auto distTypeInterface =
                        clusterOp->getOperand(0).getType().dyn_cast<VPU::DistributedTypeInterface>();
                if (distTypeInterface == nullptr) {
                    return false;
                }
                if (!distTypeInterface.containsDistributedTypes()) {
                    return false;
                }
                const auto disType = distTypeInterface.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
                const auto disMode = disType.getDistribution().getMode().getValue();
                const auto numTilesAttr = disType.getDistribution().getNumTiles();
                if (numTilesAttr == nullptr) {
                    return false;
                }
                const auto numTiles = parseIntArrayAttr<int64_t>(numTilesAttr);

                return (disMode == VPU::DistributionMode::SEGMENTED || disMode == VPU::DistributionMode::OVERLAPPED) &&
                       (numTiles[Dims4D::Act::H.ind()] > 1);
            };

            isSplitOverH = isSplitOverH || isSingleOpSplitOnH(ddrConcatInputValue);
            const bool isMemConsistentPerCluster = !(isConcatOverH && isSplitOverH);
            // Check if concat types are compatible
            const bool areTypesConsistent =
                    currentSubvector.empty()
                            ? true
                            : areDistributionTypesConsistent(ddrConcatInputValue, currentSubvector.back());

            // Check if adding the current value would exceed the cmx mem limit
            if (!currentSumFitInCmx || !isMemConsistentPerCluster || !areTypesConsistent) {
                // If so, save the current subvector and start a new one
                if (!currentSubvector.empty()) {
                    CopyOutputGroups.push_back(currentSubvector);
                    currentSubvector.clear();
                }
                currentSum = 0;
                isSplitOverH = false;
            }
            // Add the current value to the subvector and update the sum
            currentSubvector.push_back(ddrConcatInputValue);
            currentSum += concatInputParentOpSize;
        }
    }

    // Add the last subvector if it's not empty
    if (!currentSubvector.empty()) {
        CopyOutputGroups.push_back(currentSubvector);
    }

    auto nothingToOptimise = llvm::all_of(CopyOutputGroups, [&](auto input) {
        return input.size() == 1;
    });
    if (nothingToOptimise) {
        _log.trace("Nothing to optimise: '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    // From the vector of output groups build out the new concat ops
    SmallVector<mlir::Value> ddrConcatInputs;

    for (const auto& copyOutputGroup : CopyOutputGroups) {
        if (copyOutputGroup.size() == 1) {
            // there's only one copy in the input group, meaning that the copyOp tensor alone doesn't fit into cmx
            // or the input doesn't fit into the isCopyOpFromCmxWithSingleUser pattern:
            // preserve it as an input operation to the DDR concat
            ddrConcatInputs.push_back(copyOutputGroup[0]);
        } else {
            // create a cmx-concat operation followed by a CopyOp to DDR
            SmallVector<mlir::Value> cmxConcatInputs;
            cmxConcatInputs.reserve(copyOutputGroup.size());

            for (auto& copyOutputVal : copyOutputGroup) {
                auto op = copyOutputVal.getDefiningOp();
                auto copyOp = mlir::dyn_cast_or_null<VPU::CopyOp>(op);
                auto clusterCopyOp = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(op);
                if (clusterCopyOp != nullptr) {
                    if (auto innerOp = mlir::dyn_cast<VPU::CopyOp>(clusterCopyOp.getInnerTaskOp())) {
                        copyOp = innerOp;
                    }
                }

                if (copyOp == nullptr) {
                    _log.trace("Concat input is not Copy op");
                    return mlir::failure();
                }
                cmxConcatInputs.push_back(op->getOperand(0));
            }

            // create cmx concat operation
            auto cmxConcatOp = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), cmxConcatInputs, axisValue.ind());

            // create new copy op from cmx to ddr using cmx concat output
            mlir::OpBuilder builder(cmxConcatOp);
            builder.setInsertionPointAfter(cmxConcatOp);
            auto copyOutputMemSpace = origOp.getType().cast<vpux::NDTypeInterface>().getMemSpace();

            auto nceOutDistributedType = cmxConcatOp.getOutput().getType().dyn_cast<VPU::DistributedTensorType>();
            auto nceOutType = (nceOutDistributedType != nullptr)
                                      ? nceOutDistributedType.getCompactType().dyn_cast<vpux::NDTypeInterface>()
                                      : cmxConcatOp.getOutput().getType().cast<vpux::NDTypeInterface>();

            auto newDDRType = nceOutType.changeMemSpace(copyOutputMemSpace);

            mlir::Value newCopyOutput;
            if (nceOutDistributedType) {
                const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                         mlir::ValueRange newOperands) {
                    auto outputTensorDistributedCopyOp =
                            builder.create<VPU::CopyOp>(loc, newOperands[0], copyOutputMemSpace);
                    builder.create<VPU::YieldOp>(loc, outputTensorDistributedCopyOp.getOutput());
                };

                auto newCopy = builder.create<VPU::NCEClusterTilingOp>(
                        cmxConcatOp->getLoc(), newDDRType, cmxConcatOp.getOutput(), outputTensorBodyBuilder);

                newCopyOutput = newCopy->getResult(0);
            } else {
                auto newCopy =
                        builder.create<VPU::CopyOp>(cmxConcatOp->getLoc(), cmxConcatOp.getOutput(), copyOutputMemSpace);

                newCopyOutput = newCopy->getResult(0);
            }

            // the copy output is part of the ddr concat input
            ddrConcatInputs.push_back(newCopyOutput);
        }
    }

    // now we have the new input vector for
    auto newDDRConcatOp = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), ddrConcatInputs, axisValue.ind());
    rewriter.replaceOp(origOp, newDDRConcatOp.getOutput());

    return mlir::success();
}

class SplitDDRConcatIntoCMXPass final : public vpux::VPU::SplitDDRConcatIntoCMXBase<SplitDDRConcatIntoCMXPass> {
public:
    explicit SplitDDRConcatIntoCMXPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;

private:
    Logger _log;
};

//
// SplitDDRConcatIntoCMXPass
//

void SplitDDRConcatIntoCMXPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SplitConcatRewriter>(&ctx, _log);

    mlir::ConversionTarget target(ctx);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitDDRConcatIntoCMXPass(Logger log) {
    return std::make_unique<SplitDDRConcatIntoCMXPass>(log);
}
