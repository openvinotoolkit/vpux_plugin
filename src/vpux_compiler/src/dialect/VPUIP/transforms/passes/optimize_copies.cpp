//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

bool isCMX2CMXCopy(vpux::VPU::MemoryKind srcMemory, vpux::VPU::MemoryKind dstMemory) {
    return srcMemory == dstMemory && srcMemory == VPU::MemoryKind::CMX_NN;
}

bool isNonDistributedCastCompatible(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    auto inDistributedType = inType.dyn_cast<VPUIP::DistributedBufferType>();
    if (inDistributedType == nullptr || mlir::isa<VPUIP::DistributedBufferType>(outType)) {
        return false;
    }
    const auto mode = inDistributedType.getDistribution().getMode().getValue();
    if (!VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) &&
        mode != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        return false;
    }
    return inDistributedType.getShape() == outType.getShape() &&
           inDistributedType.getElementType() == outType.getElementType() &&
           inDistributedType.getMemoryKind() == outType.getMemoryKind() &&
           inDistributedType.getStrides() == outType.getStrides() &&
           inDistributedType.getDimsOrder() == outType.getDimsOrder();
}

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 4;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

mlir::Value getRootBuffer(mlir::Value buffer) {
    vpux::ValueSourceInfo aliasInfo(buffer);
    auto rootBuffers = aliasInfo.getRoots(buffer);
    VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value expected to have only one root. Got {1}", rootBuffers.size());
    return *rootBuffers.begin();
}

// Check the user of the copyOp is an EltwiseOp with is_inplace
bool isEltwiseInplaceUser(VPUIP::CopyOp copyOp) {
    mlir::Operation* op = copyOp.getOperation();
    auto clusterTiling = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterTiling) {
        op = clusterTiling.getOperation();
    }

    auto opUsers = op->getResult(0).getUsers();
    if (opUsers.empty()) {
        return false;
    }

    if (!op->hasOneUse()) {
        auto firstUserOp = *opUsers.begin();
        for (auto userOp : llvm::make_early_inc_range(opUsers)) {
            if (firstUserOp != userOp) {
                return false;
            }
        }
    }

    auto copyUser = *opUsers.begin();
    auto userClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(copyUser);
    if (userClusterTilingOp != nullptr) {
        copyUser = userClusterTilingOp.getInnerTaskOp();
    }

    const auto isEltwiseInplaceCandidate = [](VPUIP::NCEClusterTaskOp op) {
        if (op.getTaskType() != VPUIP::NCETaskType::ELTWISE) {
            return false;
        }
        return op.getIsInplace().value_or(false);
    };

    auto userClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(copyUser);
    if (userClusterTaskOp != nullptr) {
        return isEltwiseInplaceCandidate(userClusterTaskOp);
    }

    return false;
}

//
// CopyOpSequence
//

class CopyOpSequence final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CopyOpSequence(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopyOpSequence::matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    /*
     Remove redundant Copy-to-Copy sequence:
         ParentCopyOp
              |
           CopyOp
     */
    _log.trace("CopyOpSequence: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = _log.nest();
    auto parentCopyOp = copyOp.getInput().getDefiningOp<VPUIP::CopyOp>();
    if (parentCopyOp == nullptr) {
        StringRef parentOpName = "None";
        if (auto parentOp = copyOp.getInput().getDefiningOp()) {
            parentOpName = parentOp->getName().getStringRef();
        } else if (copyOp.getInput().isa<mlir::BlockArgument>()) {
            parentOpName = "BlockArgument";
        }
        nestedLogger.trace("Cannot match because parent isn't CopyOp, but '{0}'", parentOpName);
        return mlir::failure();
    }

    if (parentCopyOp.getOutputBuff().isa<mlir::BlockArgument>() ||
        !(isBufAllocOp(parentCopyOp.getOutputBuff().getDefiningOp()) ||
          VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentCopyOp.getOutputBuff()))) {
        nestedLogger.trace("Cannot match because parent's output buffer is not produced by allocation");
        return mlir::failure();
    }

    for (auto user : parentCopyOp.getOutput().getUsers()) {
        if (mlir::isa<VPUIP::SubViewOp>(user)) {
            // if intermediate SubViewOp users, skip due to accuracy loss
            // TODO E#35612: implement support for intermediate SubViewOp users
            nestedLogger.trace("Cannot match because intermediate SubViewOp users, skip due to accuracy loss");
            return mlir::failure();
        }
    }

    // In case the new copyOp will be eliminated after copyOp sequence optimization, and the user of copyOp is
    // an EltwiseOp with is_inplace, then the inplace buffer for EltwiseOp should be updated.
    auto parentCopyOpInputType = parentCopyOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto copyOutputType = copyOp.getOutputBuff().getType().cast<vpux::NDTypeInterface>();
    if (isCMX2CMXCopy(parentCopyOpInputType.getMemoryKind(), copyOutputType.getMemoryKind()) &&
        parentCopyOpInputType == copyOutputType && isEltwiseInplaceUser(copyOp)) {
        auto copyOpUser = copyOp->getResult(0).getUsers().begin();
        auto userClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(*copyOpUser);
        VPUX_THROW_UNLESS(userClusterTaskOp != nullptr, "Cannot get the user NCEClusterTaskOp");

        // Found the inplace buffer of nceOp and replace use
        auto nceOutputBuff = getRootBuffer(VPUIP::getLayerOutputs(userClusterTaskOp)[0]);
        auto copyOpOutBuff = getRootBuffer(VPUIP::getLayerOutputs(copyOp)[0]);
        if (nceOutputBuff == copyOpOutBuff) {
            auto parentCopyInputOp = parentCopyOp.getInput().getDefiningOp();
            if (parentCopyInputOp == nullptr) {
                return mlir::failure();
            }
            auto parentCopyOpInputBuff = VPUIP::getLayerOutputs(parentCopyInputOp)[0];
            rewriter.replaceAllUsesWith(nceOutputBuff, parentCopyOpInputBuff);
        }
    }

    rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyOp, parentCopyOp.getInput(), copyOp.getOutputBuff());

    // CopyOp can have MemoryEffect so "hanging" unused parentCopyOp might not be erased by MLIR automatically
    if (parentCopyOp->use_empty()) {
        rewriter.eraseOp(parentCopyOp);
    }

    nestedLogger.trace("Successfully fused sequence of copies into one op");
    return mlir::success();
}

//
// NCEClusterCopyOpSequence
//

class NCEClusterCopyOpSequence final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    NCEClusterCopyOpSequence(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NCEClusterCopyOpSequence::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                              mlir::PatternRewriter& rewriter) const {
    // Eliminate copy pairs - spills to DDR
    _log.trace("NCEClusterCopyOpSequence: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = _log.nest();

    auto clusterTiling = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (clusterTiling == nullptr) {
        nestedLogger.trace("Cannot match because copy operation isn't wrapped by NCEClusterTilingOp");
        return mlir::failure();
    }

    auto parentClusterTiling = clusterTiling->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    auto parentDistributedCastOp = clusterTiling->getOperand(0).getDefiningOp<VPUIP::DistributedCastOp>();
    if (parentDistributedCastOp) {
        parentClusterTiling = parentDistributedCastOp.getInput().getDefiningOp<VPUIP::NCEClusterTilingOp>();
    }
    if (parentClusterTiling == nullptr) {
        nestedLogger.trace("Cannot match because source producer isn't wrapped by NCEClusterTilingOp");
        return mlir::failure();
    }

    auto parentCopy = parentClusterTiling.getInnerTaskOpOfType<VPUIP::CopyOp>();
    if (parentCopy == nullptr) {
        nestedLogger.trace("Cannot match because predecessor isn't CopyOp");
        return mlir::failure();
    }

    auto isCompatibleDistributedType = [&](mlir::Value input, mlir::Value output) -> bool {
        auto inDistributedType = VPUIP::extractDataType(input).dyn_cast<VPUIP::DistributedBufferType>();
        auto outDistributedType = VPUIP::extractDataType(output).dyn_cast<VPUIP::DistributedBufferType>();
        if (inDistributedType == nullptr || outDistributedType == nullptr) {
            nestedLogger.trace("Cannot match because types aren't distributed");
            return false;
        }

        if (VPU::isDistributedCastCompatible(inDistributedType, outDistributedType).failed()) {
            nestedLogger.trace("Cannot match because of types incompatibility: '{0}' != '{1}'", inDistributedType,
                               outDistributedType);
            return false;
        }

        return true;
    };

    // The I/O types of this CopyOp-chain should be similar
    auto producerInput = parentClusterTiling.getOperand(0);
    auto output = clusterTiling.getResult(0);

    // In case the NCEClusterCopyOp sequence will be eliminated after optimization, and the user of
    // NCEClusterCopyOp is an EltwiseOp with is_inplace, then need to check the distributed type
    // compatible between input Op of NCEClusterCopyOp and EltwiseOp
    //              Input Op 1                     Input Op 2
    //                 |                              |
    //   ClusterTiling_Copy(CMX2DDR)       ClusterTiling_Copy(CMX2DDR)
    //                 |                              |
    //   ClusterTiling_Copy(DDR2CMX)       ClusterTiling_Copy(DDR2CMX)
    //       With inplace buffer
    //                         \             /
    //               ClusterTiling_NCE(EltwiseOp with is_inplace)
    //                                |
    // If distributed type compatible, then convert to:
    //              Input Op 1
    //          With inplace buffer        Input Op 2
    //                         \             /
    //               ClusterTiling_NCE(EltwiseOp with is_inplace)
    //                                |
    // If distributed type incompatible, then convert to:
    //              Input Op 1
    //                 |
    //   ClusterTiling_Copy(CMX2DDR)
    //                 |
    //   ClusterTiling_Copy(DDR2CMX)
    //       With inplace buffer            Input Op 2
    //                         \             /
    //               ClusterTiling_NCE(EltwiseOp with is_inplace)
    //                                |
    if (isEltwiseInplaceUser(copyOp)) {
        auto userClusterTilingOp =
                mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*clusterTiling.getResult(0).getUsers().begin());
        auto userClusterTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(userClusterTilingOp.getInnerTaskOp());
        VPUX_THROW_UNLESS(userClusterTaskOp != nullptr, "Cannot get the user NCEClusterTaskOp");

        // Found the inplace buffer of nceOp and replace use with compatible input Op buffer of NCEClusterCopyOp
        auto nceOutputBuff = getRootBuffer(VPUIP::getLayerOutputs(userClusterTilingOp)[0]);
        auto tilingCopyOpOutBuff = getRootBuffer(VPUIP::getLayerOutputs(clusterTiling)[0]);
        if (nceOutputBuff == tilingCopyOpOutBuff) {
            auto parentTilingCopyInputOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(
                    parentClusterTiling.getOperand(0).getDefiningOp());
            if (parentTilingCopyInputOp == nullptr) {
                return mlir::failure();
            }

            if (parentTilingCopyInputOp->getResult(0).getType() != output.getType() &&
                !isCompatibleDistributedType(parentTilingCopyInputOp->getResult(0), output)) {
                nestedLogger.trace(
                        "Do not fuse sequence copy as the user is EltwiseOp with inplace and incompatible type");
                return mlir::failure();
            }

            auto parentCopyOpInputBuff = VPUIP::getLayerOutputs(parentTilingCopyInputOp)[0];
            rewriter.replaceAllUsesWith(nceOutputBuff, parentCopyOpInputBuff);
        }
    }

    if (producerInput.getType() != output.getType()) {
        if (!isCompatibleDistributedType(producerInput, output)) {
            return mlir::failure();
        }

        rewriter.setInsertionPointAfter(parentClusterTiling);
        rewriter.replaceOpWithNewOp<VPUIP::DistributedCastOp>(clusterTiling, output.getType(), producerInput);

        if (parentClusterTiling->use_empty()) {
            rewriter.eraseOp(parentClusterTiling);
        }
        nestedLogger.trace("Successfully fused sequence of NCEClusterTiled copies into one op");
        return mlir::success();
    }

    rewriter.replaceOp(clusterTiling, producerInput);
    if (parentDistributedCastOp && parentDistributedCastOp->use_empty()) {
        rewriter.eraseOp(parentDistributedCastOp);
    }
    if (parentClusterTiling->use_empty()) {
        rewriter.eraseOp(parentClusterTiling);
    }
    nestedLogger.trace("Successfully fused sequence of NCEClusterTiled copies into one op");
    return mlir::success();
}

//
// CMXToCMXCopy
//

template <class ConcreteType>
ConcreteType getParentOp(mlir::Operation* copyOp) {
    return copyOp->getOperand(0).getDefiningOp<ConcreteType>();
}

bool isHighDimInputStrideCopy(VPUIP::NCEClusterTilingOp clusterCopyOp) {
    if (!mlir::isa_and_nonnull<VPUIP::SubViewOp>(clusterCopyOp.getOperand(0).getDefiningOp())) {
        return false;
    }
    // Copy cannot be eliminated for nested SubViewOps
    auto isNestedSubviewUser = llvm::any_of(clusterCopyOp->getUsers(), [](mlir::Operation* user) {
        return mlir::isa<VPUIP::SubViewOp>(user);
    });
    if (isNestedSubviewUser) {
        return false;
    }
    auto innerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(clusterCopyOp.getInnerTaskOp());
    if (innerCopyOp == nullptr) {
        return false;
    }
    auto inputType = clusterCopyOp->getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
    auto outputType = clusterCopyOp->getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    const auto inputElemSize = inputType.getElemTypeSize();
    const auto inputShape = inputType.getShape();
    const auto inputLayout = inputType.getDimsOrder();
    const auto outputElemSize = outputType.getElemTypeSize();
    const auto outputShape = outputType.getShape();
    const auto outputLayout = outputType.getDimsOrder();
    if (inputElemSize != outputElemSize || inputShape != outputShape || inputLayout != outputLayout) {
        return false;
    }
    auto inputMemShape = inputType.getMemShape().raw();
    auto inputMemStrides = inputType.getMemStrides().raw();
    auto getStrideDim = [&]() -> Dim {
        for (auto ind : irange(inputMemShape.size()) | reversed) {
            auto dim = Dim(ind);
            if (ind == inputMemShape.size() - 1 && inputMemStrides[ind] != inputElemSize) {
                return dim;
            } else if (ind != inputMemShape.size() - 1) {
                const auto prevMemDim = ind + 1;
                if (inputMemStrides[ind] != inputMemStrides[prevMemDim] * inputMemShape[prevMemDim]) {
                    return dim;
                }
            }
        }
        return Dim(0);
    };
    auto strideDim = getStrideDim();
    return strideDim == Dims4D::Act::N;
}

bool isDistributedInOutCompatible(VPUIP::NCEClusterTilingOp clusterCopyOp) {
    const auto tilingCopyInput = clusterCopyOp->getOperand(0);
    const auto tilingCopyOutput = clusterCopyOp->getResult(0);
    const auto inDistributedType = VPUIP::extractDataType(tilingCopyInput).dyn_cast<VPUIP::DistributedBufferType>();
    const auto outDistributedType = VPUIP::extractDataType(tilingCopyOutput).dyn_cast<VPUIP::DistributedBufferType>();
    if (inDistributedType != outDistributedType) {
        if (inDistributedType == nullptr || outDistributedType == nullptr) {
            return false;
        }

        if (VPU::areDistributionAttrsCompatible(inDistributedType, outDistributedType).failed()) {
            return false;
        }
    }

    return true;
}

bool isExcludedUser(mlir::Operation* op) {
    // For normal case, NCE or groupOp conncet to ConcatView directly
    if (mlir::isa<VPUIP::ConcatViewOp>(op)) {
        return true;
    }

    // For sparse with distributedCast case, NCE or groupOp conncet to distributedCastOp
    if (auto castOp = mlir::dyn_cast<VPUIP::DistributedCastOp>(op)) {
        if (castOp->hasOneUse() && mlir::isa<VPUIP::ConcatViewOp>(*castOp.getResult().getUsers().begin())) {
            return true;
        }
    }
    return false;
}

bool needInsertCopies(mlir::Operation* op, size_t resultIndex) {
    if (op->use_empty()) {
        return false;
    }

    const auto isCopy = [](mlir::Operation* user) {
        auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(user);
        auto tilingCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user);
        return copyOp != nullptr || (tilingCopyOp != nullptr && tilingCopyOp.getInnerTaskOpOfType<VPUIP::CopyOp>());
    };
    for (auto user : op->getResult(resultIndex).getUsers()) {
        if (VPUIP::isPureViewOp(user)) {
            if (isExcludedUser(user)) {
                continue;
            }

            // currently we can only propagate stride through quantizeCast, but could not for other view like. For
            // example: 1x16x32x64 genericReshape to 1x16x16x128(NCHW), if input stride is in H [33280,2080,65,1], don't
            // know how to set output stride. Some special case may work, like input stride in C [34816,2048,64, 1],
            // but haven't been handled.
            if (mlir::isa<VPUIP::QuantizeCastOp>(user)) {
                VPUX_THROW_UNLESS(user->getNumResults() == 1, "QuantizeCastOp must have single output");
                if (needInsertCopies(user, 0)) {
                    return true;
                }
                continue;
            }

            // Insert copies for other view like operation
            return true;
        }

        if (!isCopy(user)) {
            return true;
        }
    }
    return false;
}

void propagateStrideInfo(mlir::Operation* parent, size_t resultIndex, mlir::PatternRewriter& rewriter) {
    if (parent->use_empty()) {
        return;
    }

    auto origOutType = parent->getResult(resultIndex).getType().cast<vpux::NDTypeInterface>();
    const auto inReqs = StrideReqs::compact(origOutType.getRank());
    if (inReqs.checkStrides(origOutType)) {
        return;
    }
    auto parentStrides = getStrides(parent->getResult(resultIndex));

    const auto isTilingCopy = [](mlir::Operation* user) {
        auto tilingCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user);
        return tilingCopyOp != nullptr && tilingCopyOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
    };
    const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };
    for (auto user : llvm::make_early_inc_range(parent->getResult(resultIndex).getUsers())) {
        if (isExcludedUser(user)) {
            continue;
        }

        if (mlir::isa<VPUIP::CopyOp>(user)) {
            continue;
        }

        if (mlir::isa<VPUIP::QuantizeCastOp>(user)) {
            VPUX_THROW_UNLESS(user->getNumResults() == 1, "QuantizeCastOp must have single output");
            auto origType = user->getResult(0).getType().cast<vpux::NDTypeInterface>();
            auto newType = origType.changeStrides(parentStrides);
            user->getResult(0).setType(newType);
            propagateStrideInfo(user, 0, rewriter);
            continue;
        }

        if (isTilingCopy(user)) {
            // TilingCopy need to re-create to make sure stride info propagated.
            auto tilingCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user);
            rewriter.setInsertionPointAfter(parent);
            SmallVector<mlir::Value> inputsOutputOperands = {tilingCopyOp->getOperand(0),
                                                             tilingCopyOp.getOutputBuffs()[0]};
            auto newTilingCopy = rewriter.create<VPUIP::NCEClusterTilingOp>(tilingCopyOp->getLoc(),
                                                                            tilingCopyOp->getResult(0).getType(),
                                                                            inputsOutputOperands, copyOutBodyBuilder);
            auto allocOp = tilingCopyOp.getOutputBuffs()[0].getDefiningOp();
            if (newTilingCopy->isBeforeInBlock(allocOp)) {
                newTilingCopy->moveAfter(allocOp);
            }
            rewriter.replaceOp(tilingCopyOp, newTilingCopy->getResult(0));
            continue;
        }

        VPUX_THROW("Unsupported operation type {0} to propagate stride info", user->getName());
    }
}

void insertCopiesAfterNCETask(VPUIP::NCEClusterTaskOp parentNCE, size_t resultIndex, mlir::Type origType,
                              mlir::PatternRewriter& rewriter) {
    auto nceOutType = origType.dyn_cast<vpux::NDTypeInterface>();
    rewriter.setInsertionPointAfter(parentNCE);
    // To DDR
    auto newDDRType = nceOutType.changeMemSpace(VPU::MemoryKind::DDR);
    auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(appendLoc(parentNCE->getLoc(), "_new_DDR_buffer"),
                                                                newDDRType.cast<mlir::MemRefType>());
    auto newCopyToDDR = rewriter.create<VPUIP::CopyOp>(appendLoc(parentNCE->getLoc(), "_stride_to_compact"),
                                                       parentNCE->getResult(resultIndex), newAllocDDROp);

    // To CMX
    auto newAllocCMXOp = rewriter.create<mlir::memref::AllocOp>(appendLoc(parentNCE->getLoc(), "_new_CMX_buffer"),
                                                                nceOutType.cast<mlir::MemRefType>());
    auto newCopyToCMX = rewriter.create<VPUIP::CopyOp>(parentNCE->getLoc(), newCopyToDDR.getResult(), newAllocCMXOp);

    rewriter.replaceUsesWithIf(parentNCE->getResult(resultIndex), newCopyToCMX.getResult(),
                               [&](mlir::OpOperand& opOperand) {
                                   return opOperand.getOwner() != newCopyToDDR && !isExcludedUser(opOperand.getOwner());
                               });
}

void insertCopiesAfterNCETaskDistributedBuffer(VPUIP::NCEClusterTilingOp parentNCEClusterOp, size_t resultIndex,
                                               mlir::Type origType, mlir::PatternRewriter& rewriter) {
    VPUX_THROW_WHEN(!parentNCEClusterOp.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>(),
                    "Should be a Tiling NCE task but actually not");

    auto nceOutDistributedType = origType.dyn_cast<VPUIP::DistributedBufferType>();
    auto nceOutType = nceOutDistributedType.getCompactType().dyn_cast<vpux::NDTypeInterface>();
    const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };
    rewriter.setInsertionPointAfter(parentNCEClusterOp);
    // To DDR
    auto newDDRType = nceOutType.changeMemSpace(VPU::MemoryKind::DDR);
    auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(
            appendLoc(parentNCEClusterOp->getLoc(), "_new_DDR_buffer"), newDDRType.cast<mlir::MemRefType>());

    SmallVector<mlir::Value> ddrCopyOperands = {parentNCEClusterOp->getResult(resultIndex),
                                                static_cast<mlir::Value>(newAllocDDROp)};
    auto newTillingCopyToDDR =
            rewriter.create<VPUIP::NCEClusterTilingOp>(appendLoc(parentNCEClusterOp->getLoc(), "_stride_to_compact"),
                                                       newDDRType, ddrCopyOperands, copyOutBodyBuilder);
    // To CMX
    auto newDistributeBuff = rewriter.create<VPURT::AllocDistributed>(
            appendLoc(parentNCEClusterOp->getLoc(), "_new_CMX_buffer"), nceOutDistributedType, nullptr, nullptr);
    SmallVector<mlir::Value> cmxCopyOperands = {newTillingCopyToDDR->getResult(0),
                                                static_cast<mlir::Value>(newDistributeBuff)};
    auto newTillingCopyToCMX = rewriter.create<VPUIP::NCEClusterTilingOp>(
            parentNCEClusterOp->getLoc(), nceOutDistributedType, cmxCopyOperands, copyOutBodyBuilder);

    rewriter.replaceUsesWithIf(parentNCEClusterOp->getResult(resultIndex), newTillingCopyToCMX->getResult(0),
                               [&](mlir::OpOperand& opOperand) {
                                   return opOperand.getOwner() != newTillingCopyToDDR &&
                                          !isExcludedUser(opOperand.getOwner());
                               });
}

void handleStrideForOtherUsers(mlir::Operation* parent, size_t resultIndex, mlir::Type origType,
                               mlir::PatternRewriter& rewriter, Logger log) {
    if (needInsertCopies(parent, resultIndex)) {
        if (auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(parent)) {
            insertCopiesAfterNCETask(nceTask, resultIndex, origType, rewriter);
        } else if (auto clustringNceTask = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(parent)) {
            insertCopiesAfterNCETaskDistributedBuffer(clustringNceTask, resultIndex, origType, rewriter);
        } else {
            VPUX_THROW("Incorrect parent type {0}", parent->getName());
        }
        log.trace("Insert a pair of copy to handle stride");
    } else {
        propagateStrideInfo(parent, resultIndex, rewriter);
        log.trace("Propagate stride info to child");
    }
}

/// Finds an op->getResult(x) that is equal to value. Returns result index.
size_t findMatchingResultIndex(mlir::Value value, mlir::Operation* op) {
    const auto& results = op->getResults();
    auto it = llvm::find(results, value);
    VPUX_THROW_WHEN(it == results.end(), "Failed to find {0} in op's results: {1}", value, op);
    return std::distance(op->getResults().begin(), it);
}

mlir::LogicalResult removeClusterTilingCMXToCMXCopy(VPUIP::NCEClusterTilingOp copyClusterOp,
                                                    mlir::PatternRewriter& rewriter, Logger log) {
    log.trace("removeClusterTilingCMXToCMXCopy: Copy at {0}", copyClusterOp->getLoc());
    auto nestedLogger = log.nest();

    auto innerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(copyClusterOp.getInnerTaskOp());
    if (innerCopyOp == nullptr) {
        nestedLogger.trace("Cannot match because tiling op does not contain Copy");
        return mlir::failure();
    }

    auto inputType = copyClusterOp->getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
    auto outputType = copyClusterOp->getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    // Only remove redundant CMX2CMX CopyOps
    if (!isCMX2CMXCopy(inputType.getMemoryKind(), outputType.getMemoryKind())) {
        nestedLogger.trace("Cannot match because the transfer is not CMX->CMX");
        return mlir::failure();
    }

    auto distributedCast = getParentOp<VPUIP::DistributedCastOp>(copyClusterOp);

    // CMX Concat case with subView, update the buffers used
    if (auto copySubView = copyClusterOp.getOutputBuffs()[0].getDefiningOp<VPUIP::SubViewOp>()) {
        // case with subView - retrieve operations to be re-linked
        auto masterBuffer = VPUIP::getRootAlloc<VPURT::AllocDistributed>(copySubView.getSource());
        if (masterBuffer == nullptr) {
            nestedLogger.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }

        auto parentNCEClusterOp = distributedCast == nullptr ? getParentOp<VPUIP::NCEClusterTilingOp>(copyClusterOp)
                                                             : getParentOp<VPUIP::NCEClusterTilingOp>(distributedCast);
        if (parentNCEClusterOp == nullptr) {
            nestedLogger.trace("Cannot match because copy is not a successor of NCEClusterTiling or of a "
                               "NCEClusterTiling -> VPUIP.DistributedCast sequence");
            return mlir::failure();
        }

        mlir::Operation* parentOp = parentNCEClusterOp;
        const auto updateParentNCEOp = [&](size_t argIdx, mlir::Value value) {
            // Update result types of NCEClusterTiling
            parentNCEClusterOp->getResult(checked_cast<unsigned int>(argIdx)).setType(value.getType());
            // Update output buffers of NCEClusterTiling
            parentNCEClusterOp.getOutputBuffs()[argIdx].replaceAllUsesWith(value);
            // Update inner NCEClusterTask
            mlir::Operation* nceClusterTaskOp = parentNCEClusterOp.getInnerTaskOp();
            const auto newInnerType = value.getType().dyn_cast<VPUIP::DistributedBufferType>().getCompactType();
            nceClusterTaskOp->getResult(checked_cast<unsigned int>(argIdx)).setType(newInnerType);
            VPUIP::getLayerOutputs(nceClusterTaskOp)[argIdx].setType(newInnerType);
        };

        // Note: a distributed cast could be between nce task and tiling copy
        const auto inputValue =
                (distributedCast == nullptr) ? copyClusterOp.getInputs()[0] : distributedCast.getInput();
        const auto outputBuffIndex = findMatchingResultIndex(inputValue, parentNCEClusterOp);
        const auto origType = parentNCEClusterOp->getResult(outputBuffIndex).getType();
        if (parentNCEClusterOp.getOutputBuffs()[outputBuffIndex].getDefiningOp<VPUIP::SubViewOp>()) {
            nestedLogger.trace("NCE output is already the subview of Concat");
            return mlir::failure();
        }

        copySubView->moveBefore(parentNCEClusterOp);

        // replace the copy with the subView
        auto nceClusterOutput = copySubView.getResult();
        if (distributedCast != nullptr) {
            rewriter.setInsertionPointAfter(copySubView);
            auto ndTypeIfValue = copySubView.getType().cast<NDTypeInterface>();
            auto distributedCastType = distributedCast->getOperand(0).getType().cast<NDTypeInterface>().changeStrides(
                    ndTypeIfValue.getStrides());

            nestedLogger.trace("Creating DistributedCastOp with input = {0} and output type = {1}.", copySubView,
                               distributedCastType);

            auto newDistrCast = rewriter.create<VPUIP::DistributedCastOp>(parentNCEClusterOp->getLoc(),
                                                                          distributedCastType, copySubView);
            nceClusterOutput = newDistrCast.getResult();
        }

        updateParentNCEOp(outputBuffIndex, nceClusterOutput);
        rewriter.replaceAllUsesWith(copyClusterOp->getResult(0), parentNCEClusterOp->getResult(outputBuffIndex));

        // update IR location of the master buffer
        if (copySubView->isBeforeInBlock(masterBuffer)) {
            VPUIP::moveRootAllocBefore(masterBuffer, copySubView);
        }

        rewriter.eraseOp(copyClusterOp);
        if (distributedCast != nullptr) {
            rewriter.eraseOp(distributedCast);
        }

        // now we need to propagate stride info to other users
        handleStrideForOtherUsers(parentOp, outputBuffIndex, origType, rewriter, log);
    } else if (inputType == outputType ||
               (isHighDimInputStrideCopy(copyClusterOp) && isDistributedInOutCompatible(copyClusterOp))) {
        // case with no subView   Or
        // case with input subView
        // if the subView splits on the highest dimension
        // eliminate the CMX2CMX copy
        rewriter.replaceAllUsesWith(copyClusterOp->getResults(),
                                    copyClusterOp.getOperand(0).getDefiningOp()->getResults());
        rewriter.eraseOp(copyClusterOp);
        if (distributedCast != nullptr) {
            rewriter.eraseOp(distributedCast);
        }
    } else {
        log.trace("Copy not optimized {0}", copyClusterOp->getLoc());
        return mlir::failure();
    }

    nestedLogger.trace("Successfully removed sequence");
    return mlir::success();
}

mlir::LogicalResult removeCMXToCMXCopy(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    // Check current CopyOp source and destination
    log.trace("removeCMXToCMXCopy: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = log.nest();

    auto inputType = copyOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outputType = copyOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    // if detect the subview before input, remove the copies may cause CMX OOM
    // so check whether removing the CMX2CMX copy will exceed CMX limitation
    auto isInputCompatible = true;
    if (auto definingOp = copyOp.getInput().getDefiningOp()) {
        if (auto subViewOpBefore = mlir::dyn_cast<VPUIP::SubViewOp>(definingOp)) {
            // find the pattern: source data -> subview -> copy -> users
            // check for all users, if any user's CMX size exceeds the limitation, then cannot remove the copy
            auto newInputCmxSize =
                    subViewOpBefore.getSource().getType().cast<vpux::NDTypeInterface>().getTotalAllocSize();
            for (auto copyOpUser : copyOp.getOutput().getUsers()) {
                Byte requiredCMX = VPUIP::getRequiredCMXSize(copyOpUser);
                requiredCMX -= inputType.getTotalAllocSize();
                requiredCMX += newInputCmxSize;
                nestedLogger.trace("CMX Demand: {0} -> {1} for {2}", VPUIP::getRequiredCMXSize(copyOpUser), requiredCMX,
                                   copyOpUser->getLoc());
                if (requiredCMX > VPU::getTotalCMXSize(copyOp)) {
                    isInputCompatible = false;
                    break;
                }
            }
        }
    }

    // Only remove redundant CMX2CMX CopyOps
    if (!isCMX2CMXCopy(inputType.getMemoryKind(), outputType.getMemoryKind())) {
        nestedLogger.trace("Cannot match because the transfer is not CMX->CMX");
        return mlir::failure();
    }
    // CMX Concat case with SubView, update the buffers used
    if (auto copySubView = mlir::dyn_cast<VPUIP::SubViewOp>(copyOp.getOutputBuff().getDefiningOp())) {
        // case with SubView - retrieve operations to be re-linked
        auto parentNCE = getParentOp<VPUIP::NCEClusterTaskOp>(copyOp);

        if (parentNCE == nullptr) {
            nestedLogger.trace("Cannot match because copy operation is not a successor of NCEClusterTask");
            return mlir::failure();
        }

        auto masterBuffer = VPUIP::getRootAlloc<mlir::memref::AllocOp>(copySubView->getOperand(0));
        if (masterBuffer == nullptr) {
            nestedLogger.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }

        VPUIP::moveRootAllocBefore(copySubView, parentNCE);

        const auto outputBuffIndex = findMatchingResultIndex(copyOp.getInput(), parentNCE);
        const auto origType = parentNCE->getResult(outputBuffIndex).getType().dyn_cast<vpux::NDTypeInterface>();
        // replace the copy with the subView
        parentNCE->getResult(outputBuffIndex).setType(copySubView.getResult().getType());
        rewriter.replaceAllUsesWith(VPUIP::getLayerOutputs(parentNCE)[outputBuffIndex], copySubView.getResult());
        rewriter.replaceAllUsesWith(copyOp.getOutput(), copyOp.getInput());

        // update IR location of the master buffer
        if (copySubView->isBeforeInBlock(masterBuffer)) {
            VPUIP::moveRootAllocBefore(masterBuffer, copySubView);
        }

        rewriter.eraseOp(copyOp);

        // now we need to propagate stride info to other users
        handleStrideForOtherUsers(parentNCE, outputBuffIndex, origType, rewriter, log);
    } else if (inputType == outputType && isInputCompatible) {
        // case with no subView after output
        rewriter.replaceAllUsesWith(copyOp.getOutput(), copyOp.getInput());
        rewriter.eraseOp(copyOp);
    } else {
        log.trace("Copy not optimized {0}", copyOp->getLoc());
        return mlir::failure();
    }

    nestedLogger.trace("Successfully removed sequence");
    return mlir::success();
}

class CMXToCMXCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    CMXToCMXCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CMXToCMXCopy::matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const {
    /*
     Remove CMX2CMX Copy without SubView:
         Copy(DDR2CMX)                    Copy(DDR2CMX)
              |                                |
            NCEOp           =>               NCEOp
              |
         Copy(CMX2CMX)

     Remove CMX2CMX Copy with SubView:
        Copy(DDR2CMX)                Copy(DDR2CMX)  SubView
              |                                \     /
            NCEOp       SubView   =>            NCEOp
               \         /
              Copy(CMX2CMX)

     For Cluster-ed scenario, it is possible to have:
        Copy(DDR2CMX)                Copy(DDR2CMX)  SubView
              |                            |           |
            NCEOp        =>                |    (DistributedCast)
              |                            \        / (output_buff)
    (DistributedCast)  SubView                NCEOp
               \         / (output_buff)
              Copy(CMX2CMX)

    For Cluster-ed scenario with sparsity map, final subgraph should be:
               Alloc Data    Alloc SparsityMap
                    |                |
                 SubView          SubView    -> (if original SubView output had multiple
                    |                |           consumers, this is their new producer)
            (DistributedCast) (DistributedCast)
                    \                /
  (output_data_buff) \              / (output_sparsity_map_buff)
                      \            /
                           NCEOp


     */
    if (auto clusterTilingCopy = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        return removeClusterTilingCMXToCMXCopy(clusterTilingCopy, rewriter, _log);
    } else {
        return removeCMXToCMXCopy(copyOp, rewriter, _log);
    }
}

//
// DDRToDDRCopyOfNCECluster
//

class DDRToDDRCopyOfNCECluster final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    DDRToDDRCopyOfNCECluster(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isDDR2DDROfNCEClusterInput(VPUIP::CopyOp copyOp) {
    // ChildOp should be a copy op wrapped in ClusterTilingOp
    if (copyOp.getOutput().getUsers().empty()) {
        return false;
    }

    auto isClusterTilingCopyOp = [](mlir::Operation* user) {
        if (auto tilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user)) {
            return tilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>() != nullptr;
        }
        return false;
    };

    auto isLegalUpdateViewLikeInType = [](mlir::Operation* op, mlir::Value newInput) {
        auto iface = mlir::dyn_cast<mlir::InferTypeOpInterface>(*op);
        SmallVector<mlir::Type> newTypes;
        const auto isLegal =
                iface.inferReturnTypes(op->getContext(), op->getLoc(), mlir::ValueRange{newInput},
                                       op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(), newTypes)
                        .succeeded();
        return isLegal;
    };

    for (auto copyOpUser : copyOp.getOutput().getUsers()) {
        // TODO: Extend for other ViewLike ops E#74293
        if (mlir::isa<VPUIP::ShapeCastOp, VPUIP::SubViewOp>(copyOpUser) &&
            mlir::isa<mlir::InferTypeOpInterface>(copyOpUser)) {
            if (!isLegalUpdateViewLikeInType(copyOpUser, copyOp.getInput())) {
                return false;
            }

            for (auto pureViewOpUser : copyOpUser->getResult(0).getUsers()) {
                if (!isClusterTilingCopyOp(pureViewOpUser)) {
                    return false;
                }
            }
        } else if (!isClusterTilingCopyOp(copyOpUser)) {
            return false;
        }
    }

    return true;
}

bool hasValidParallelCopyBranchWithSubView(VPUIP::CopyOp copyOp, VPUIP::NCEClusterTilingOp parentOp) {
    if (parentOp->hasOneUse()) {
        return false;
    }

    auto subview = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
    if (subview == nullptr) {
        return false;
    }

    // If a CMX to DDR copy's input is a subview of SOH's output, the CMX2DDR copy's input tensor will have a SEGMENTED
    // or OVERLAPPED distribution. But the output data of the tensor's subview may be distributed on one cluster or
    // multiple clusters.In the current compiler logic, when calculating DMA cost and unroll DMA, it is assumed that the
    // data of the Tensor with SEGMENTED or OVERLAPPED distribution is distributed on multiple clusters. Therefore, SOH
    // optimization is temporarily turned off and turned on after subsequent compiler support.E60342
    if (auto distType = VPUIP::extractDataType(parentOp.getInputs()[0]).dyn_cast<VPUIP::DistributedBufferType>()) {
        if (distType.getDistribution().getMode().getValue() == VPU::DistributionMode::SEGMENTED ||
            distType.getDistribution().getMode().getValue() == VPU::DistributionMode::OVERLAPPED) {
            auto subviewshape = subview.getResult().getType().cast<vpux::NDTypeInterface>().getShape().raw();
            auto numTiles = parseIntArrayAttr<int64_t>(distType.getDistribution().getNumTiles());
            if (subviewshape.size() == 4 && subviewshape[Dims4D::Act::H.ind()] % numTiles[Dims4D::Act::H.ind()] != 0) {
                return false;
            }

            // In case of the CMX2DDR copy's input tensor has a SEGMENTED or OVERLAPPED distribution and the tile Axis
            // is H, and if the output data of the tensor's subview has tile offsets including H, then the tile result
            // may be incorrect after SOH optimization (When the offset is a non first cluster / the offset is only
            // used in a single cluster or a few clusters / the offset exists across two consecutive clusters), as
            // current compiler logic not support this case in calculating DMA cost and unroll DMA
            // TODO: Add optimization for this case, #E80157
            for (auto user : llvm::make_early_inc_range(parentOp.getResult(0).getUsers())) {
                if (auto subview = mlir::dyn_cast<VPUIP::SubViewOp>(*user)) {
                    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*subview.getResult().getUsers().begin());
                    auto offsetAttr = subview.getStaticOffsetsAttr();
                    const auto offsetsArray = parseIntArrayAttr<int64_t>(offsetAttr);
                    const auto tilingScheme = parseIntArrayAttr<int64_t>(distType.getDistribution().getNumTiles());
                    const auto tileAxis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
                    if (copyOp && offsetsArray[tileAxis]) {
                        return false;
                    }
                }
            }
        }
    }

    // check other parallel branch if it's a valid copy branch or not
    for (auto siblingOp : parentOp.getResults().getUsers()) {
        // Considering padding/slice case: Tiling_copy -> subview -> copy

        if (auto siblingSubview = mlir::dyn_cast<VPUIP::SubViewOp>(*siblingOp)) {
            if (!siblingSubview.getResult().hasOneUse()) {
                return false;
            }

            auto childOp = siblingSubview.getResult().getUsers().begin();
            auto childCopy = mlir::dyn_cast<VPUIP::CopyOp>(*childOp);
            // If childCopy is nullptr or its output buffer is not defined by a SubViewOp, return false.
            if (!childCopy || !childCopy.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>()) {
                return false;
            }

        } else if (auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(*siblingOp)) {
            // If siblingCopy is not the same as copyOp and its output buffer is not defined by a SubViewOp, return
            // false.
            if (siblingCopy != copyOp && !siblingCopy.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>()) {
                return false;
            }

        } else {
            return false;
        }
    }
    // check all branches and okay
    return true;
}

// For the case: parent of copyOp only have one output branch
// Parallel case should be processed by isParallelDDR2DDROfNCEClusterOutput()
// for clear logic
bool isDDR2DDROfNCEClusterOutput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a copy op wrapped in ClusterTilingOp
    // ChildOp should be a concat
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (parentOp == nullptr || parentOp.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
        return false;
    }
    if (copyOp.getOutput().getUsers().empty()) {
        return false;
    }
    for (auto user : copyOp.getOutput().getUsers()) {
        if (!mlir::isa<VPUIP::ConcatViewOp>(*user)) {
            return false;
        }
    }

    return parentOp->hasOneUse();
}

bool isParallelDDR2DDROfNCEClusterOutput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a copy op wrapped in ClusterTilingOp
    // ChildOp should be a concat
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (parentOp == nullptr || parentOp.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
        return false;
    }

    if (copyOp.getOutput().getUsers().empty()) {
        return false;
    }
    for (auto user : copyOp.getOutput().getUsers()) {
        if (!mlir::isa<VPUIP::ConcatViewOp>(*user)) {
            return false;
        }
    }

    /*
     Optimize the parallel DDR2DDR copies as CMX2DDR copies:
                 ClusterTiling_Copy(CMX2DDR)
                      /        \
            Copy(DDR2DDR)   (SubViews ->) Copy(DDR2DDR)
            /        \                 /       \
        SubView              |               SubView
                             |
                          Concat
    */
    return hasValidParallelCopyBranchWithSubView(copyOp, parentOp);
}

bool isStridedCopy(VPUIP::CopyOp copyOp) {
    // Here we check two options at the same time:
    // 1. Copy op is not strided, in the sense that step for copying dimension is 1
    // 2. Copy can handle full plane without offsets

    const auto outType = copyOp.getOutputBuff().getType().cast<vpux::NDTypeInterface>();
    const auto order = outType.getDimsOrder();
    const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(order, outType);
    auto compactStrides = order.toLogicalOrder(memStrides);

    auto actStrides = outType.getStrides();
    VPUX_THROW_UNLESS(compactStrides.size() == actStrides.size(),
                      "Compact ({0}) and actual ({1}) strides size mismatch", compactStrides.size(), actStrides.size());

    for (size_t i = 1; i < compactStrides.size(); i++) {
        if (compactStrides[Dim(i)] != actStrides[Dim(i)]) {
            return true;
        }
    }

    return false;
}

bool isDDR2DDROfConcatInput(VPUIP::CopyOp copyOp) {
    // ParentOp should be a concatView op
    // ChildOp should be a concatView too
    auto parentConcatOp = copyOp.getInput().getDefiningOp<VPUIP::ConcatViewOp>();
    if (parentConcatOp == nullptr) {
        return false;
    }
    if (!copyOp.getOutput().hasOneUse()) {
        return false;
    }

    auto childConcatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*copyOp.getOutput().getUsers().begin());
    if (childConcatOp == nullptr) {
        return false;
    }

    // Exclude strided dma case
    size_t constCopyCnt = 0;
    auto predicteChildConcatInput = [&](mlir::Value op) {
        auto copy = op.getDefiningOp<VPUIP::CopyOp>();
        if (copy == nullptr || isStridedCopy(copy)) {
            return false;
        }

        auto concat = copy.getInput().getDefiningOp<VPUIP::ConcatViewOp>();
        if (concat == nullptr) {
            auto subView = copy.getInput().getDefiningOp<VPUIP::SubViewOp>();
            if (subView == nullptr) {
                auto parentCopyInputConst = VPUIP::getRootConst(copy.getInput());
                if (parentCopyInputConst) {
                    constCopyCnt++;
                    return true;
                }
                return false;
            } else if (!subView.getResult().hasOneUse()) {
                return false;
            }
            concat = subView.getSource().getDefiningOp<VPUIP::ConcatViewOp>();
        }

        return concat == parentConcatOp;
    };

    /*
     E.g., Optimize the left DDR2DDR copy in below case:
     case 1:
                      ConcatView
                      /         \
             Copy(DDR2DDR)      SubView
                     \            \
                      \        Copy(DDR2DDR)
                       \        /
                           |
                           |
                       ConcatView
    case 2:
                ConcatView
                    |
             Copy(DDR2DDR)      const.Declare
                     \            |
                      \        Copy(DDR2DDR)
                       \        /
                           |
                           |
                       ConcatView
    */
    if (!llvm::all_of(childConcatOp.getInputs(), predicteChildConcatInput)) {
        return false;
    }

    const auto childConcatInputsNum = childConcatOp.getInputs().size();

    const auto parentConcatUsers = parentConcatOp.getOutput().getUsers();
    const auto parentConcatUsersNum = std::distance(parentConcatUsers.begin(), parentConcatUsers.end());

    return (childConcatInputsNum - constCopyCnt) == static_cast<size_t>(parentConcatUsersNum);
}

mlir::LogicalResult removeDDR2DDRForNCEClusterInput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    rewriter.replaceAllUsesWith(copyOp.getOutput(), copyOp.getInput());

    // Update ViewLike Op Output Type
    SmallVector<mlir::Operation*> viewLikeOps;
    for (auto copyOpUser : copyOp.getInput().getUsers()) {
        if (mlir::isa<VPUIP::ShapeCastOp, VPUIP::SubViewOp>(copyOpUser)) {
            viewLikeOps.push_back(copyOpUser);
        }
    }

    for (auto viewLikeOp : viewLikeOps) {
        vpux::inferReturnTypes(viewLikeOp, vpux::InferShapedTypeMode::ALL);
    }

    log.trace("Successfully removed DDRToDDR input copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult removeDDR2DDRForNCEClusterOutput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                                     Logger log) {
    // CMX Concat case with subView, update the buffers used
    if (auto subViewOp = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>()) {
        // case with subView - retrieve operations to be re-linked
        auto masterBuffer = VPUIP::getRootAlloc<mlir::memref::AllocOp>(subViewOp->getOperand(0));
        if (masterBuffer == nullptr) {
            log.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }
        auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
        // replace the copy with VPUIP subView
        rewriter.setInsertionPoint(parentOp);
        auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(
                subViewOp->getLoc(), subViewOp.getSource(), subViewOp.getStaticOffsetsAttr(),
                subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());
        rewriter.replaceAllUsesWith(parentOp.getOutputBuffs()[0], newSubViewOp->getResult(0));
        parentOp->getResult(0).setType(newSubViewOp->getResult(0).getType());

        // update IR location of the master buffer
        if (newSubViewOp->isBeforeInBlock(masterBuffer)) {
            VPUIP::moveRootAllocBefore(masterBuffer, newSubViewOp);
        }
    } else {
        auto parentOp = copyOp.getInput().getDefiningOp<VPUIP::NCEClusterTilingOp>();
        auto allocOp = VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentOp.getOutputBuffs()[0]);
        if (allocOp == nullptr) {
            log.trace("Cannot match because source isn't master buffer");
            return mlir::failure();
        }

        for (auto user : copyOp.getOutput().getUsers()) {
            auto concatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(user);
            concatOp.getOutputBuff().replaceAllUsesWith(allocOp->getResult(0));
        }
    }

    rewriter.replaceAllUsesWith(copyOp.getOutput(), copyOp.getInput());
    log.trace("Successfully removed Clustered DDRToDDR output copy {0} at {1}", copyOp->getName(), copyOp->getLoc());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult removeParallelDDR2DDRForNCEClusterOutput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter,
                                                             Logger log) {
    auto parentOp = copyOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();

    for (auto user : llvm::make_early_inc_range(parentOp.getResult(0).getUsers())) {
        if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*user)) {
            auto subview = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();

            rewriter.setInsertionPointAfter(subview);
            const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                mlir::ValueRange newOperands) {
                builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
            };
            SmallVector<mlir::Value> inputsOutputOperands = {parentOp->getOperand(0), subview.getResult()};
            auto newCopyInCluster = rewriter.create<VPUIP::NCEClusterTilingOp>(
                    parentOp->getLoc(), subview->getResult(0).getType(), inputsOutputOperands, copyOutBodyBuilder);

            rewriter.replaceAllUsesWith(copyOp.getOutput(), newCopyInCluster->getResult(0));

            log.trace("Successfully removed Parallel DDRToDDR output copy {0} at {1}", copyOp->getName(),
                      copyOp->getLoc());
            rewriter.eraseOp(copyOp);
        }
    }

    for (auto user : llvm::make_early_inc_range(parentOp.getResult(0).getUsers())) {
        if (auto subview = mlir::dyn_cast<VPUIP::SubViewOp>(*user)) {
            auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*subview.getResult().getUsers().begin());
            if (copyOp == nullptr) {
                log.trace("CopyOp is null");
                continue;
            }
            auto outputSubview = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
            if (outputSubview == nullptr) {
                log.trace("Output subview is null");
                continue;
            }

            rewriter.setInsertionPointAfter(copyOp);
            // New a new subview for copy output
            auto newSubView = rewriter.create<VPUIP::SubViewOp>(
                    subview->getLoc(), parentOp->getOperand(0), subview.getStaticOffsetsAttr(),
                    subview.getStaticSizesAttr(), subview.getStaticStridesAttr());

            const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                mlir::ValueRange newOperands) {
                builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
            };
            SmallVector<mlir::Value> inputsOutputOperands = {newSubView.getResult(), outputSubview.getResult()};
            auto newCopyInCluster = rewriter.create<VPUIP::NCEClusterTilingOp>(
                    parentOp->getLoc(), outputSubview.getResult().getType(), inputsOutputOperands, copyOutBodyBuilder);

            rewriter.replaceAllUsesWith(copyOp.getOutput(), newCopyInCluster->getResult(0));
            log.trace("Successfully removed Parallel DDRToDDR output copy (with input subview) {0} at {1}",
                      copyOp->getName(), copyOp->getLoc());
            rewriter.eraseOp(copyOp);
            rewriter.eraseOp(subview);
        }
    }

    if (parentOp->use_empty()) {
        rewriter.eraseOp(parentOp);
    }
    return mlir::success();
}

static inline bool checkOpsSupportInferType(mlir::Operation* startOp, mlir::Operation* endOp, Logger log) {
    auto currentOp = startOp;

    while (currentOp != endOp) {
        if (!mlir::isa<mlir::InferTypeOpInterface, mlir::memref::AllocOp, VPUIP::NCEClusterTilingOp>(currentOp)) {
            log.trace("Unexpected op {0} at {1}", currentOp->getName(), currentOp->getLoc());
            return false;
        }
        currentOp = currentOp->getNextNode();
    }
    return true;
}

static inline void inferOpsTypeBetween(mlir::Operation* startOp, mlir::Operation* endOp) {
    auto currentOp = startOp;

    while (currentOp != endOp) {
        // In case the currentOp is a VPUIP::NCEClusterTilingOp and it doesn't support mlir::InferTypeOpInterface,
        // then will setType based on the SubViewOp of this NCEClusterTilingOp,
        // no adapt to set the inner type as only the strides changed.

        // Only AllocOp and NCEClusterTilingOp will call this if func after checkOpsSupportInferType func
        // no need to infer AllocOp's type
        if (!mlir::isa<mlir::InferTypeOpInterface>(currentOp)) {
            if (auto tilingCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(currentOp)) {
                for (auto result : currentOp->getResults() | indexed) {
                    result.value().setType(tilingCopyOp.getOutputBuffs()[result.index()].getType());
                }
                currentOp = currentOp->getNextNode();
            } else if (mlir::isa<mlir::memref::AllocOp>(currentOp)) {
                currentOp = currentOp->getNextNode();
                continue;
            } else {
                VPUX_THROW("Unexpected op type '{0}' at '{1}'", currentOp->getName(), currentOp->getLoc());
            }
        } else {
            vpux::inferReturnTypes(currentOp, vpux::InferShapedTypeMode::ALL);
            currentOp = currentOp->getNextNode();
        }
    }
}

mlir::LogicalResult removeDDR2DDRForConcatInput(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter, Logger log) {
    auto parentConcatOp = copyOp.getInput().getDefiningOp<VPUIP::ConcatViewOp>();
    auto parentMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentConcatOp.getOutputBuff());
    if (parentMemAlloc == nullptr) {
        log.trace("Cannot match because parentConcatOp output isn't master buffer");
        return mlir::failure();
    }

    auto childConcatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*copyOp.getOutput().getUsers().begin());
    auto childMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(childConcatOp.getOutputBuff());
    if (childMemAlloc == nullptr) {
        log.trace("Cannot match because childConcatOp output isn't master buffer");
        return mlir::failure();
    }

    auto childMemSize = vpux::getTotalSize(childMemAlloc->getResult(0));
    auto parentMemSize = vpux::getTotalSize(parentMemAlloc->getResult(0));
    if (childMemSize <= parentMemSize) {
        log.error("There is no redundant Copy operation since the child size ({0}) <= parent size ({1})", childMemSize,
                  parentMemSize);
        return mlir::failure();
    }

    if (!checkOpsSupportInferType(parentMemAlloc, childConcatOp, log)) {
        log.trace("Cannot match because some Ops doesn't support InferTypeOpInterface");
        return mlir::failure();
    }

    log.trace("Successfully removed DDRToDDR output copy {0} at {1} for Concat", copyOp->getName(), copyOp->getLoc());
    auto childCopySubview = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();

    auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(parentMemAlloc->getLoc(), childCopySubview.getSource(),
                                                          childCopySubview.getStaticOffsetsAttr(),
                                                          childCopySubview.getStaticSizesAttr());

    // update IR location of the master buffer
    if (parentMemAlloc->isBeforeInBlock(newSubViewOp)) {
        VPUIP::moveRootAllocBefore(newSubViewOp, parentMemAlloc);
    }
    // update IR location of the master buffer
    if (newSubViewOp->isBeforeInBlock(childMemAlloc)) {
        VPUIP::moveRootAllocBefore(childMemAlloc, newSubViewOp);
    }

    rewriter.replaceAllUsesWith(parentMemAlloc->getResult(0), newSubViewOp.getResult());
    rewriter.eraseOp(parentMemAlloc);
    // Re-Infer the Type of the Ops
    inferOpsTypeBetween(newSubViewOp, childConcatOp);

    rewriter.replaceAllUsesWith(copyOp.getOutput(), copyOp.getInput());
    rewriter.eraseOp(copyOp);
    return mlir::success();
}

mlir::LogicalResult DDRToDDRCopyOfNCECluster::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                              mlir::PatternRewriter& rewriter) const {
    /*
     Remove redundant DDR2DDR Copy of the NCECluster's input:
ClusterTiling_Copy                    ...        SubView
   (CMX2DDR)        SubView             \         /
          \         /              ClusterTiling_Copy(CMX2DDR)
          Copy(DDR2DDR)        =>            |
               |                           Concat
            Concat

     Remove redundant DDR2DDR Copy of the NCECluster's output:
          Copy(DDR2DDR)                                PureViewOp(Optional)
                |                                             |
        PureViewOp(Optional: ShapeCast, SubView)       ClusterTiling_Copy
                |                                         (DDR2CMX)
        ClusterTiling_Copy              =>                    |
            (DDR2CMX)                                  ClusterTiling_NCE
                |                                             |
        ClusterTiling_NCE
                |

     Optimize the parallel DDR2DDR copies as CMX2DDR copies:
                ClusterTiling_Copy(CMX2DDR)
                      /        \
            Copy(DDR2DDR)       Copy(DDR2DDR)       =>
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat

                         ...
                     /          \
ClusterTiling_Copy(CMX2DDR)   ClusterTiling_Copy(CMX2DDR)
            /        \          /       \
        SubView           |            SubView
                          |
                        Concat
     */
    _log.trace("DDRToDDRCopyOfNCECluster: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = _log.nest();
    if (!VPUIP::isCopyFromDDR(copyOp) || !VPUIP::isCopyToDDR(copyOp)) {
        nestedLogger.trace("Cannot match because isn't DDR->DDR copy");
        return mlir::failure();
    }

    if (isDDR2DDROfNCEClusterInput(copyOp)) {
        return removeDDR2DDRForNCEClusterInput(copyOp, rewriter, nestedLogger);
    } else if (isDDR2DDROfNCEClusterOutput(copyOp)) {
        return removeDDR2DDRForNCEClusterOutput(copyOp, rewriter, nestedLogger);
    } else if (isParallelDDR2DDROfNCEClusterOutput(copyOp)) {
        // TODO: Add this optimization in single cluster case
        return removeParallelDDR2DDRForNCEClusterOutput(copyOp, rewriter, nestedLogger);
    } else if (isDDR2DDROfConcatInput(copyOp)) {
        return removeDDR2DDRForConcatInput(copyOp, rewriter, nestedLogger);
    }
    std::string possibleReason;
    if (copyOp.getInput().getDefiningOp<Const::DeclareOp>()) {
        possibleReason = " Copy from Constant isn't optimizable";
    }
    nestedLogger.trace("Unsupported pattern.{0}", possibleReason);
    return mlir::failure();
}

//
// ConcatViewWithCopyBase
//

class ConcatViewWithCopyBase : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    ConcatViewWithCopyBase(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isLegalConcatViewPattern(VPUIP::ConcatViewOp origOp, vpux::Logger log) const;

    virtual bool hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const = 0;
    virtual mlir::Value getOutputBuffer(mlir::Operation* sourceOp) const = 0;
    virtual mlir::LogicalResult adaptBufferTypeToPemuteCastInput(mlir::Value buffer, VPUIP::PermuteCastOp permuteCast,
                                                                 Logger log) const = 0;
    virtual VPUIP::LayerOpInterface createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                                    mlir::PatternRewriter& rewriter) const = 0;

private:
    bool hasDuplicatedCopyOutput(VPUIP::ConcatViewOp origOp) const;

    Logger _log;
};

mlir::FailureOr<VPU::DistributionInfoAttr> deducePermuteCastInputDistributionInfoAttr(
        VPUIP::PermuteCastOp permuteCast, VPUIP::DistributedBufferType outputDistributedType) {
    auto perm = permuteCast.getMemPerm();
    auto inversePerm = mlir::inversePermutation(perm);

    auto inPermuteType = permuteCast->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto outPermuteType = permuteCast->getResult(0).getType().cast<vpux::NDTypeInterface>();

    return applyPermutationOnDistributionInfoAttr(outputDistributedType, inversePerm, outPermuteType.getDimsOrder(),
                                                  inPermuteType.getDimsOrder(), outPermuteType.getShape(),
                                                  inPermuteType.getShape());
}

mlir::LogicalResult ConcatViewWithCopyBase::matchAndRewrite(VPUIP::ConcatViewOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (!isLegalConcatViewPattern(origOp, _log)) {
        _log.nest().trace("Cannot fuse this ConcatView Op {0}", origOp.getLoc());
        return mlir::failure();
    }

    mlir::Operation* firstCopyOp;
    auto* childOp = getFirstUser(origOp.getResult());
    auto permuteCastOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(childOp);
    if (permuteCastOp != nullptr) {
        firstCopyOp = getFirstUser(permuteCastOp.getResult());
    } else {
        firstCopyOp = childOp;
    }
    VPUX_THROW_UNLESS(firstCopyOp != nullptr, "Cannot get the first user Op");

    _log.trace("Got ConcatView Op at '{0}'", origOp.getLoc());

    SmallVector<mlir::Value> concatInputs;
    auto outBuffer = getOutputBuffer(firstCopyOp);

    // record original buffer type before adaptBufferTypeToPemuteCastInput is called as it may be adjusted in it
    auto origBufferType = outBuffer.getType();
    // update buffer type if there is PermuteCastOp after ConcatViewOp
    if (permuteCastOp != nullptr) {
        if (mlir::failed(adaptBufferTypeToPemuteCastInput(outBuffer, permuteCastOp, _log))) {
            _log.nest().trace("Failed to adapt buffer type to PermuteCast input at '{0}'", origOp.getLoc());
            return mlir::failure();
        }
    }

    auto outBufferDefiningOp = outBuffer.getDefiningOp();
    VPUX_THROW_WHEN(outBufferDefiningOp == nullptr, "Cannot get defining op for {0}", outBuffer);
    rewriter.setInsertionPointAfter(outBufferDefiningOp);
    for (auto input : origOp.getInputs()) {
        auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
        auto subViewOp = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();

        auto newSubView =
                rewriter.create<VPUIP::SubViewOp>(copyOp.getLoc(), outBuffer, subViewOp.getStaticOffsetsAttr(),
                                                  subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());

        auto newCopyOp = createNewCopyOp(copyOp, newSubView, rewriter);

        concatInputs.push_back(newCopyOp->getResult(0));
    }

    rewriter.setInsertionPointAfter(firstCopyOp);
    auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(firstCopyOp->getLoc(), concatInputs, outBuffer);
    if (permuteCastOp != nullptr) {
        auto newPermuteCastOutputType = origBufferType;
        if (auto inDistributedType = newConcatOp.getOutput().getType().dyn_cast<VPUIP::DistributedBufferType>()) {
            auto perm = permuteCastOp.getMemPerm();
            auto inPermuteType = permuteCastOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
            auto outPermuteType = permuteCastOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

            auto outDistribution = applyPermutationOnDistributionInfoAttr(
                    inDistributedType, perm, inPermuteType.getDimsOrder(), outPermuteType.getDimsOrder(),
                    inPermuteType.getShape(), outPermuteType.getShape());
            VPUX_THROW_WHEN(mlir::failed(outDistribution), "Failed to infer output distribution");
            const auto orderMap =
                    mlir::AffineMapAttr::get(outPermuteType.getDimsOrder().toAffineMap(rewriter.getContext()));
            newPermuteCastOutputType = VPUIP::DistributedBufferType::get(
                    rewriter.getContext(), outPermuteType.getShape().raw(), outPermuteType.getElementType(), orderMap,
                    inDistributedType.getMemSpace(), outDistribution.value());
        }

        auto newPermuteCastOp =
                rewriter.create<VPUIP::PermuteCastOp>(permuteCastOp->getLoc(), newPermuteCastOutputType, newConcatOp,
                                                      permuteCastOp.getDstOrderAttr(), permuteCastOp.getMemPermAttr());
        auto distributedCast = rewriter.createOrFold<VPUIP::DistributedCastOp>(permuteCastOp->getLoc(), origBufferType,
                                                                               newPermuteCastOp.getResult());

        for (auto userCopyOp : llvm::make_early_inc_range(permuteCastOp.getResult().getUsers())) {
            rewriter.replaceOp(userCopyOp, distributedCast);
        }
    } else {
        for (auto userCopyOp : llvm::make_early_inc_range(origOp.getOutput().getUsers())) {
            rewriter.replaceOp(userCopyOp, newConcatOp.getOutput());
        }
    }

    _log.nest().trace("Successfully simplified ConcatView {0}", origOp->getLoc());
    return mlir::success();
}

/*
  Check pattern:
  Copy (DDR2DDR)  ...  Copy (DDR2DDR)
       \               /
        Concat View (DDR)
             |
        [PermuteCast]
             |
        Copy(DDR2CMX)
*/
bool ConcatViewWithCopyBase::isLegalConcatViewPattern(VPUIP::ConcatViewOp origOp, vpux::Logger log) const {
    if (!origOp.getOutput().hasOneUse() && !hasDuplicatedCopyOutput(origOp)) {
        log.nest().trace("Cannot find user copy op at '{0}'", origOp);
        return false;
    }
    for (auto input : origOp.getInputs()) {
        auto op = mlir::dyn_cast_or_null<VPUIP::CopyOp>(input.getDefiningOp());
        if (op == nullptr || !VPUIP::isCopyToDDR(op) || !VPUIP::isCopyFromDDR(op)) {
            return false;
        }
    }

    return hasLegalCopyUser(origOp);
}

bool ConcatViewWithCopyBase::hasDuplicatedCopyOutput(VPUIP::ConcatViewOp origOp) const {
    if (origOp.use_empty()) {
        return false;
    }
    auto isSameCopyType = [](mlir::Operation* preOp, mlir::Operation* nextOp) {
        auto preCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(preOp);
        auto nextCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(nextOp);
        if (preCopyOp != nullptr && nextCopyOp != nullptr) {
            auto preOutType = preCopyOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
            auto nextOutType = preCopyOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
            return preOutType == nextOutType;
        }

        auto preClusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(preOp);
        auto nextClusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(nextOp);
        if (preClusterCopyOp == nullptr || nextClusterCopyOp == nullptr) {
            return false;
        }
        auto preInnerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(preClusterCopyOp.getInnerTaskOp());
        auto nextInnerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(nextClusterCopyOp.getInnerTaskOp());
        if (preInnerCopyOp == nullptr || nextInnerCopyOp == nullptr) {
            return false;
        }
        auto preOutputType = preClusterCopyOp.getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
        auto nextOutputType = nextClusterCopyOp.getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
        return preOutputType == nextOutputType;
    };

    auto firstUser = *origOp.getOutput().getUsers().begin();
    return llvm::all_of(origOp.getOutput().getUsers(), [&](auto user) {
        return isSameCopyType(firstUser, user);
    });
}

//
// ConcatViewWithCopy
//

/*
  Copy (DDR -> DDR)  ...  Copy (DDR -> DDR)
                \               /
                Concat View (DDR)             =>           Copy (DDR -> CMX) ... Copy (DDR -> CMX)
                        |                                           \               /
                  [PermuteCast]                                     Concat View (CMX)
                        |                                                   |
                Copy (DDR -> CMX)                                     [PermuteCast]
*/

class ConcatViewWithCopy final : public ConcatViewWithCopyBase {
public:
    ConcatViewWithCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : ConcatViewWithCopyBase(ctx, benefit, log) {
    }

public:
    bool hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const override;
    mlir::Value getOutputBuffer(mlir::Operation* sourceOp) const override;
    mlir::LogicalResult adaptBufferTypeToPemuteCastInput(mlir::Value buffer, VPUIP::PermuteCastOp permuteCast,
                                                         Logger log) const override;
    VPUIP::LayerOpInterface createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                            mlir::PatternRewriter& rewriter) const override;
};

bool ConcatViewWithCopy::hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const {
    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*sourceOp->getUsers().begin());
    if (copyOp == nullptr) {
        auto maybePermuteCast = mlir::dyn_cast<VPUIP::PermuteCastOp>(*sourceOp->getUsers().begin());
        if (maybePermuteCast == nullptr || !maybePermuteCast->hasOneUse()) {
            return false;
        }

        copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*maybePermuteCast->getUsers().begin());
    }

    return copyOp != nullptr && VPUIP::isCopyFromDDR(copyOp) && !VPUIP::isCopyToDDR(copyOp) &&
           !VPUIP::isCopyWithStaticStrides(copyOp);
}

mlir::Value ConcatViewWithCopy::getOutputBuffer(mlir::Operation* sourceOp) const {
    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(sourceOp);
    VPUX_THROW_WHEN(copyOp == nullptr, "Unexpected op type at '{0}'", sourceOp->getLoc());
    return copyOp.getOutputBuff();
}

mlir::LogicalResult ConcatViewWithCopy::adaptBufferTypeToPemuteCastInput(mlir::Value buffer,
                                                                         VPUIP::PermuteCastOp permuteCastOp,
                                                                         Logger log) const {
    const auto origBufferType = buffer.getType();
    if (!mlir::isa<mlir::MemRefType>(origBufferType)) {
        log.trace("Unknown buffer type {0}", origBufferType);
        return mlir::failure();
    }
    if (VPUIP::getRootAlloc<mlir::memref::AllocOp>(buffer) == nullptr) {
        log.trace("Cannot match because buffer isn't master buffer");
        return mlir::failure();
    }

    const auto permuteCastInputType = permuteCastOp.getSource().getType().cast<NDTypeInterface>();
    const auto permuteCastInputOrder = permuteCastInputType.getDimsOrder();
    const auto permuteCastInputShape = permuteCastInputType.getShape();
    const auto outputType = buffer.getType().cast<NDTypeInterface>();
    buffer.setType(outputType.changeDimsOrder(permuteCastInputOrder).changeShape(permuteCastInputShape));

    return mlir::success();
}

VPUIP::LayerOpInterface ConcatViewWithCopy::createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                                            mlir::PatternRewriter& rewriter) const {
    return rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(copyInput, copyInput.getInput(), subViewOp.getResult());
}

//
// ConcatViewWithTilingCopy
//

/*
 Copy (DDR -> DDR)  ...  Copy (DDR -> DDR)
                \               /
                Concat View (DDR)             =>  Cluster Copy (DDR -> CMX) ... Cluster Copy (DDR -> CMX)
                        |                                           \               /
                  [PermuteCast]                                     Concat View (CMX)
                        |                                                   |
              Cluster Copy (DDR -> CMX)                             [PermuteCast]
*/

class ConcatViewWithTilingCopy final : public ConcatViewWithCopyBase {
public:
    ConcatViewWithTilingCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : ConcatViewWithCopyBase(ctx, benefit, log) {
    }

public:
    bool hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const override;
    mlir::Value getOutputBuffer(mlir::Operation* sourceOp) const override;
    mlir::LogicalResult adaptBufferTypeToPemuteCastInput(mlir::Value buffer, VPUIP::PermuteCastOp permuteCast,
                                                         Logger log) const override;
    VPUIP::LayerOpInterface createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                            mlir::PatternRewriter& rewriter) const override;
};

bool ConcatViewWithTilingCopy::hasLegalCopyUser(VPUIP::ConcatViewOp sourceOp) const {
    auto clusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*sourceOp->getUsers().begin());
    VPUIP::PermuteCastOp maybePermuteCast = nullptr;
    if (clusterOp == nullptr) {
        maybePermuteCast = mlir::dyn_cast<VPUIP::PermuteCastOp>(*sourceOp->getUsers().begin());
        if (maybePermuteCast == nullptr || !maybePermuteCast->hasOneUse()) {
            return false;
        }

        clusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*maybePermuteCast->getUsers().begin());
    }

    if (clusterOp == nullptr) {
        return false;
    }

    auto copyOp = clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
    if (copyOp == nullptr || isStridedCopy(copyOp)) {
        return false;
    }

    // Get the concat dims
    const auto inputType = sourceOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>();
    const auto outputType = sourceOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inputType.getShape();
    const auto outShape = outputType.getShape();
    VPUX_THROW_UNLESS(inShape.size() == outShape.size(), "Input shape size {0} is not equal to output shape size {1}",
                      inShape.size(), outShape.size());
    SmallVector<Dim> concatDims;
    for (auto idx : irange(inShape.size())) {
        if (inShape[Dim(idx)] != outShape[Dim(idx)]) {
            concatDims.push_back(Dim(idx));
        }
    }
    VPUX_THROW_WHEN(concatDims.empty(), "ConcatView inShape '{0}' same with the outShape '{1}'", inputType.getShape(),
                    outputType.getShape());

    const auto distributedType =
            VPUIP::extractDataType(clusterOp.getOutputBuffs()[0]).dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Cannot get distributedType");

    auto distribution = distributedType.getDistribution();
    if (maybePermuteCast != nullptr) {
        const auto result = deducePermuteCastInputDistributionInfoAttr(maybePermuteCast, distributedType);
        if (mlir::failed(result)) {
            return false;
        }

        distribution = result.value();
    }

    // For Overlapped mode, use compute_shape and compute_offset to unroll the DMA copy in unroll cluster copy
    // Then we will lost the stride info of the input. It will cause result incorrect
    //     TilingCopy (1x16x8x8)      TilingCopy(1x16x8x8)
    //                      \           /
    //                    Concat(1x32x8x8) (shape[1,32,5,8][1,32,5,8], offset[0,0,0,0][0,0,3,0])
    // TODO: E#78122 remove the checking after the jira fixed
    if (distribution.getMode().getValue() == VPU::DistributionMode::OVERLAPPED) {
        return false;
    }
    if (distribution.getNumTiles() != nullptr) {
        const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
        const auto tileAxis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
        auto outputLayout = outputType.getDimsOrder();
        auto tileAxisIndex = outputLayout.dimPos(Dim(tileAxis));
        auto isOutmostDimension = [&]() {
            for (auto i : concatDims) {
                if (tileAxisIndex > outputLayout.dimPos(i))
                    return false;
            }
            return true;
        };
        if (llvm::find(concatDims, Dim(tileAxis)) != concatDims.end() ||
            (outShape[Dim(tileAxis)] % tilingScheme[tileAxis] != 0 && !isOutmostDimension())) {
            // If the output buffer on tile dim can not be divided evenly on each tile, the buffer will be discontinous
            // after concat, so need to avoid such tranform.
            // E.g.:
            // VPUIP.SubView %source [0, 0, 0, 0] [1, 512, 35, 36] ->SEGMENTED with numTiles = [1, 1, 4, 1]
            // VPUIP.SubView %source [0, 128, 0, 0] [1, 512, 35, 36] -> SEGMENTED with numTiles = [1, 1, 4, 1]
            // The distribution in memory for this example would be:
            //             Cluster 0        Cluster 1        Cluster 2        Cluster 3
            // offset0  x_______________________________________________________________
            //          |  9 lines of   |  9 lines of   |  9 lines of   |  8 lines of   |
            //          | actual data   | actual data   | actual data   | actual data   |
            //          |               |               |               |---------------|
            // offset1  x---------------|---------------|---------------|---------------|
            //          |  9 lines of   |  9 lines of   |  9 lines of   |  8 lines of   |
            //          | actual data   | actual data   | actual data   | actual data   |
            //          |_______________|_______________|_______________|_______________|
            // Unexpected concat on cluster3
            //
            // Particularly case be accept concats if the concat dim is more inner compared to the clustering dim
            // E.g.:
            // VPUIP.SubView %source [0, 0, 0, 0] [1, 35, 36, 512] ->SEGMENTED with numTiles = [1, 4, 1, 1]
            // VPUIP.SubView %source [0, 0, 0, 128] [1, 35, 36, 512] -> SEGMENTED with numTiles = [1, 4, 1, 1]
            // The distribution in memory for this example would be((data arranged on vertical axis)):
            //             Cluster 0        Cluster 1        Cluster 2        Cluster 3
            // offset0  x_______________________________________________________________
            //          |  9 lines of   |  9 lines of   |  9 lines of   |  8 lines of   |
            //          | actual data   | actual data   | actual data   | actual data   |
            //          |               |               |               |---------------|
            // offset1  x---------------|---------------|---------------|---------------|
            //          |  9 lines of   |  9 lines of   |  9 lines of   |  8 lines of   |
            //          | actual data   | actual data   | actual data   | actual data   |
            //          |_______________|_______________|_______________|_______________|
            //  the data in the last cluster is indeed contiguous
            return false;
        }
    }

    return VPUIP::isCopyFromDDR(copyOp) && !VPUIP::isCopyToDDR(copyOp);
}

mlir::Value ConcatViewWithTilingCopy::getOutputBuffer(mlir::Operation* sourceOp) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(sourceOp);
    VPUX_THROW_WHEN(clusterTilingOp == nullptr, "Unexpected op type at '{0}'", sourceOp);
    return clusterTilingOp.getOutputBuffs()[0];
}

mlir::LogicalResult ConcatViewWithTilingCopy::adaptBufferTypeToPemuteCastInput(mlir::Value buffer,
                                                                               VPUIP::PermuteCastOp permuteCastOp,
                                                                               Logger log) const {
    const auto origBufferType = buffer.getType();
    if (!mlir::isa<VPUIP::DistributedBufferType>(origBufferType)) {
        log.trace("Unknown buffer type {0}", origBufferType);
        return mlir::failure();
    }
    if (VPUIP::getRootAlloc<VPURT::AllocDistributed>(buffer) == nullptr) {
        log.trace("Cannot match because buffer isn't master buffer");
        return mlir::failure();
    }

    const auto getNewDistributedType = [&](VPUIP::DistributedBufferType origType, ShapeRef newShape,
                                           DimsOrder newOrder) -> VPUIP::DistributedBufferType {
        const auto ctx = permuteCastOp->getContext();
        const auto newDistribution = deducePermuteCastInputDistributionInfoAttr(permuteCastOp, origType).value();
        const auto newOrderMap = mlir::AffineMapAttr::get(newOrder.toAffineMap(ctx));
        return VPUIP::DistributedBufferType::get(ctx, newShape.raw(), origType.getElementType(), newOrderMap,
                                                 origType.getMemSpace(), newDistribution);
    };

    const auto origDistributedBufferType = mlir::cast<VPUIP::DistributedBufferType>(origBufferType);
    const auto permuteCastInputType = permuteCastOp.getSource().getType().cast<NDTypeInterface>();
    const auto permuteCastInputShape = permuteCastInputType.getShape();
    const auto permuteCastInputOrder = permuteCastInputType.getDimsOrder();
    const auto newBufferType =
            getNewDistributedType(origDistributedBufferType, permuteCastInputShape, permuteCastInputOrder);
    buffer.setType(newBufferType);

    return mlir::success();
}

VPUIP::LayerOpInterface ConcatViewWithTilingCopy::createNewCopyOp(VPUIP::CopyOp copyInput, VPUIP::SubViewOp subViewOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(copyInput->getLoc(), newOperands[0], newOperands[1]);
    };
    auto inputsOutputOperands = {copyInput.getInput(), subViewOp.getResult()};
    auto newClusterTilingOutType = subViewOp.getResult().getType().cast<vpux::NDTypeInterface>();
    return rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(copyInput, newClusterTilingOutType,
                                                                  inputsOutputOperands, bodyBuilder);
}

//
// FuseCopyToTheFrontOfTilingCopy
//
/*
 Fuse copy into the front Tilling copy
          |                |
  TillingCopy    =>  TillingCopy
          |                |
         Copy
          |
*/

class FuseCopyToTheFrontOfTilingCopy final : public mlir::OpRewritePattern<VPUIP::NCEClusterTilingOp> {
public:
    FuseCopyToTheFrontOfTilingCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTilingOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTilingOp clusterTilingCopyOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseCopyToTheFrontOfTilingCopy::matchAndRewrite(VPUIP::NCEClusterTilingOp clusterTilingCopy,
                                                                    mlir::PatternRewriter& rewriter) const {
    /*
    case 1:
              |                          |
      TillingCopy(CMX2DDR)    =>     TillingCopy(CMX2CMX)
              |                          |
           Copy(DDR2CMX)
              |

    case 2:
              |                          |
      TillingCopy(CMX2DDR)    =>     TillingCopy(CMX2DDR)
              |                          |
           Copy(DDR2DDR)
              |
    */

    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(clusterTilingCopy.getInnerTaskOp());
    if (copyOp == nullptr || VPUIP::isCopyFromDDR(copyOp) || !VPUIP::isCopyToDDR(copyOp)) {
        return mlir::failure();
    }

    if (!clusterTilingCopy->hasOneUse()) {
        return mlir::failure();
    }

    auto tilingCopyOutput = clusterTilingCopy.getResult(0);
    auto outType = tilingCopyOutput.getType().dyn_cast<vpux::NDTypeInterface>();
    auto userCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*(tilingCopyOutput.getUsers().begin()));
    if (!userCopyOp) {
        return mlir::failure();
    }

    auto userOutType = userCopyOp.getOutputBuff().getType().dyn_cast<vpux::NDTypeInterface>();
    if (userOutType.changeMemSpace(VPU::MemoryKind::DDR) != outType) {
        return mlir::failure();
    }

    auto tilingCopyInput = clusterTilingCopy.getInputs()[0];
    if (isNonDistributedCastCompatible(VPUIP::extractDataType(tilingCopyInput), userOutType)) {
        // In this case the pattern will be optimized as a NonDistributedCast op
        return mlir::failure();
    }

    auto userOutputMemKind = userOutType.getMemoryKind();
    if (userOutputMemKind == VPU::MemoryKind::CMX_NN) {
        auto inputType = clusterTilingCopy.getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
        if (auto subviewOp = clusterTilingCopy.getInputs()[0].getDefiningOp<VPUIP::SubViewOp>()) {
            inputType = subviewOp.getViewSource().getType().cast<vpux::NDTypeInterface>();
        }

        Byte requiredCMX(0);
        requiredCMX += inputType.getTotalAllocSize();
        requiredCMX += userOutType.getTotalAllocSize();
        if (requiredCMX > VPU::getTotalCMXSize(userCopyOp)) {
            _log.trace("Available CMX size is {0}, but need {1}", VPU::getTotalCMXSize(userCopyOp), requiredCMX);
            return mlir::failure();
        }
    }

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(clusterTilingCopy->getLoc(), newOperands[0], newOperands[1]);
    };

    SmallVector<mlir::Value> inputsOutputOperands = {clusterTilingCopy->getOperand(0), userCopyOp.getOutputBuff()};

    rewriter.setInsertionPointAfter(userCopyOp);
    auto newClusterTilingCopyOp = rewriter.create<VPUIP::NCEClusterTilingOp>(clusterTilingCopy->getLoc(), userOutType,
                                                                             inputsOutputOperands, bodyBuilder);
    rewriter.replaceAllUsesWith(userCopyOp->getResults(), newClusterTilingCopyOp->getResults());
    rewriter.eraseOp(userCopyOp);
    rewriter.eraseOp(clusterTilingCopy);
    return mlir::success();
}

// FuseCopyToTheBackOfTilingCopy
//
/*
 Fuse copy into the back Tilling copy
          |
         Copy
          |
    TillingCopy   =>   TillingCopy
          |
*/

class FuseCopyToTheBackOfTilingCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    FuseCopyToTheBackOfTilingCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseCopyToTheBackOfTilingCopy::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    /*
              |
        Copy(CMX2DDR)
              |                          |
      TillingCopy(DDR2CMX)    =>     TillingCopy(CMX2CMX)
              |                          |
    */
    if (!copyOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto userClusterTilingCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*copyOp.getOutput().getUsers().begin());
    if (!userClusterTilingCopyOp) {
        return mlir::failure();
    } else {
        if (!mlir::isa<VPUIP::CopyOp>(userClusterTilingCopyOp.getInnerTaskOp())) {
            return mlir::failure();
        }
    }

    if (isCopyFromDDR(copyOp) || !isCopyToDDR(copyOp)) {
        return mlir::failure();
    }
    auto inType = copyOp.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    auto userInType = userClusterTilingCopyOp.getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
    auto userOutType = userClusterTilingCopyOp.getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    if (inType.changeMemSpace(VPU::MemoryKind::DDR) != userInType) {
        return mlir::failure();
    }

    auto userOutputMemKind = userOutType.getMemoryKind();
    if (userOutputMemKind == VPU::MemoryKind::CMX_NN) {
        auto inputType = copyOp.getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
        Byte requiredCMX(0);
        requiredCMX += inputType.getTotalAllocSize();
        requiredCMX += userOutType.getTotalAllocSize();
        if (requiredCMX > VPU::getTotalCMXSize(userClusterTilingCopyOp)) {
            _log.trace("Available CMX size is {0}, but need {1}", VPU::getTotalCMXSize(userClusterTilingCopyOp),
                       requiredCMX);
            return mlir::failure();
        }
    }

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(copyOp->getLoc(), newOperands[0], newOperands[1]);
    };

    SmallVector<mlir::Value> inputsOutputOperands = {copyOp.getInput(), userClusterTilingCopyOp.getOperand(1)};
    rewriter.setInsertionPointAfter(userClusterTilingCopyOp);
    auto newClusterTilingCopyOp = rewriter.create<VPUIP::NCEClusterTilingOp>(copyOp->getLoc(), userOutType,
                                                                             inputsOutputOperands, bodyBuilder);
    copyOp->dropAllUses();
    rewriter.eraseOp(copyOp);
    userClusterTilingCopyOp->replaceAllUsesWith(newClusterTilingCopyOp);
    rewriter.eraseOp(userClusterTilingCopyOp);

    return mlir::success();
}

//
// SubViewWithTilingCopy
//
/*
 Move SubView after TillingCopy, the assumption is to reduce copy op numbers if subview have multi tiling copy users
                        buffer
            /                            \
      subview(Tile on N)               subview(Tile on N)
           |                               |
      TilingCopy(Segmented on N)       TilingCopy(Segmented on N)
           |                               |
         MatMul                         MatMul

                           =>

                       buffer
                         |
                   TilingCopy(Duplicated)
               /                            \
      subview(Tile on N)                 subview(Tile on N)
              |                              |
DistributedCast(Duplicated|Segmented)    DistributedCast(Duplicated|Segmented)
              |                              |
           MatMul                          MatMul

*/

class SubViewWithTilingCopy : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    SubViewWithTilingCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, int64_t cmxSize, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _cmxSize(cmxSize), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::Value getSuitableSubViewPatternSourceBuffer(VPUIP::CopyOp origOp, vpux::Logger log) const;
    bool checkCMXFit(mlir::Value topBuffer) const;

private:
    int64_t _cmxSize{};
    Logger _log;
};

bool SubViewWithTilingCopy::checkCMXFit(mlir::Value topBuffer) const {
    auto type = topBuffer.getType().dyn_cast<vpux::NDTypeInterface>();
    // buffer will keep duplicated in cmx after tiling copy, so need to check the required cmx
    Byte requiredSize = type.getTotalAllocSize();
    if (type.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
        requiredSize += type.getTotalAllocSize();
    }
    return vpux::Byte(_cmxSize) >= requiredSize;
}

mlir::LogicalResult SubViewWithTilingCopy::matchAndRewrite(VPUIP::CopyOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    auto nestedLogger = _log.nest();
    auto topBuffer = getSuitableSubViewPatternSourceBuffer(origOp, nestedLogger);
    if (topBuffer == nullptr) {
        return mlir::failure();
    }

    auto ctx = origOp->getContext();
    const auto topBufferType = topBuffer.getType().cast<vpux::NDTypeInterface>();
    const auto copyOutputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto layout = mlir::AffineMapAttr::get(topBufferType.getDimsOrder().toAffineMap(origOp->getContext()));
    auto tilingCopy = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(origOp->getParentOp());
    auto distributedType = tilingCopy->getResult(0).getType().cast<VPUIP::DistributedBufferType>();
    auto distribution = distributedType.getDistribution();

    // create duplicated type
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
    const auto distributedAttr =
            VPU::DistributionInfoAttr::get(ctx, distributionModeAttr, distribution.getNumTiles(), nullptr, nullptr,
                                           nullptr, distribution.getNumClusters(), distribution.getAlignment(), nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr);

    auto distributedBufferType = VPUIP::DistributedBufferType::get(origOp->getContext(), topBufferType.getShape().raw(),
                                                                   topBufferType.getElementType(), layout,
                                                                   copyOutputType.getMemSpace(), distributedAttr);

    rewriter.setInsertionPointAfterValue(topBuffer);
    auto newBuffer = rewriter.create<VPURT::AllocDistributed>(appendLoc(origOp->getLoc(), "_extract"),
                                                              distributedBufferType, nullptr, nullptr);
    nestedLogger.trace("create new buff {0}", newBuffer);

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(appendLoc(origOp->getLoc(), "_extract"), newOperands[0], newOperands[1]);
    };
    SmallVector<mlir::Value> inputsOutputOperands = {topBuffer, newBuffer};
    auto newCopy = rewriter.create<VPUIP::NCEClusterTilingOp>(appendLoc(origOp->getLoc(), "_extract"),
                                                              newBuffer.getType(), inputsOutputOperands, bodyBuilder);
    nestedLogger.trace("Created ops '{0}'", newCopy);

    for (auto siblingOp : llvm::make_early_inc_range(topBuffer.getUsers())) {
        auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(siblingOp);
        if (siblingSubViewOp == nullptr) {
            continue;
        }
        VPUX_THROW_UNLESS(siblingSubViewOp.getResult().hasOneUse(), "subview should has one use");
        auto siblingCopyOp = *siblingSubViewOp.getResult().getUsers().begin();

        rewriter.setInsertionPoint(siblingSubViewOp);
        nestedLogger.trace("Creating VPUIP.SubView '{0}' at '{1}'", siblingSubViewOp->getName(),
                           siblingSubViewOp->getLoc());
        auto newSliceOp = rewriter.create<VPUIP::SubViewOp>(
                appendLoc(siblingSubViewOp->getLoc(), "_CMX"), newCopy->getResult(0),
                siblingSubViewOp.getStaticOffsetsAttr(), siblingSubViewOp.getStaticSizesAttr());

        auto siblingCopyOutType = siblingCopyOp->getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
        auto siblingDistribution = siblingCopyOutType.getDistribution();

        const auto targetDistributionModeAttr = VPU::DistributionModeAttr::get(
                ctx, VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED);
        VPU::DistributionInfoAttr targetDistributedAttr = nullptr;
        // If siblingDistribution has shapes and offsets set then call getNonOverlappedDistributedAttr to recompute them
        // else set them to nullptr
        if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(siblingDistribution)) {
            targetDistributedAttr = VPU::getNonOverlappedDistributedAttr(
                    siblingCopyOutType.getShape(), targetDistributionModeAttr, siblingDistribution.getNumTiles(),
                    siblingDistribution.getNumClusters(), siblingDistribution.getAlignment(),
                    siblingDistribution.getUniformDistributedSegments(), ctx);
        } else {
            targetDistributedAttr = VPU::DistributionInfoAttr::get(
                    ctx, targetDistributionModeAttr, siblingDistribution.getNumTiles(), siblingDistribution.getKernel(),
                    siblingDistribution.getPads(), siblingDistribution.getStrides(),
                    siblingDistribution.getNumClusters(), siblingDistribution.getAlignment(),
                    siblingDistribution.getUniformDistributedSegments(), nullptr, nullptr, nullptr, nullptr,
                    siblingDistribution.getEqualMemoryAndComputeView());
        }

        auto targetDistributedBufferType = VPUIP::DistributedBufferType::get(
                ctx, siblingCopyOutType.getShape().raw(), siblingCopyOutType.getElementType(),
                siblingCopyOutType.getLayout(), siblingCopyOutType.getMemSpace(), targetDistributedAttr);

        nestedLogger.trace("create new subview {0}", newSliceOp);
        auto distributedCastOp = rewriter.create<VPUIP::DistributedCastOp>(
                newSliceOp->getLoc(), targetDistributedBufferType, newSliceOp.getResult());
        nestedLogger.trace("create new cast {0}", distributedCastOp);

        rewriter.replaceAllUsesWith(siblingCopyOp->getResult(0), distributedCastOp.getResult());

        rewriter.eraseOp(siblingCopyOp);
        rewriter.eraseOp(siblingSubViewOp);
    }

    return mlir::success();
}

/*
  Check pattern:
          TopBuffer
             |
          SubView
             |
    Tiling Copy(Segmented on dim N)
*/

mlir::Value SubViewWithTilingCopy::getSuitableSubViewPatternSourceBuffer(VPUIP::CopyOp origOp, vpux::Logger log) const {
    auto isTilingCopyOpSegmentedOnN = [&log](VPUIP::NCEClusterTilingOp tilingCopyOp) {
        auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(tilingCopyOp.getInnerTaskOp());
        if (copyOp == nullptr) {
            log.trace("Not NCE Cluster Tiling Copy");
            return false;
        }
        auto outType = tilingCopyOp.getResults()[0].getType().cast<vpux::NDTypeInterface>();
        const auto distributedType = outType.dyn_cast<VPUIP::DistributedBufferType>();
        if (outType == nullptr) {
            return false;
        }

        return VPU::isSegmentedOverN(distributedType.getDistribution());
    };

    auto doesTilingCopyOpHasStridedOuput = [](VPUIP::NCEClusterTilingOp tilingCopyOp) {
        auto tilingCopyOutput = tilingCopyOp.getOutputs()[0];
        auto tilingCopyOutputType = VPUIP::extractDataType(tilingCopyOutput).cast<vpux::NDTypeInterface>();

        const auto outReqs = StrideReqs::compact(tilingCopyOutputType.getRank());
        return !outReqs.checkStrides(tilingCopyOutputType);
    };

    auto wrapperOp = origOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (wrapperOp == nullptr) {
        return nullptr;
    }
    auto parentSubViewOp = wrapperOp.getInputs()[0].getDefiningOp<VPUIP::SubViewOp>();
    if (parentSubViewOp == nullptr) {
        return nullptr;
    }
    auto topBuffer = parentSubViewOp.getSource();

    if (!checkCMXFit(topBuffer)) {
        return nullptr;
    }

    if (topBuffer.hasOneUse()) {
        return nullptr;
    }

    // Calculate the new required cmx size for user op, since the new input will be
    // changed into SEG|DUP instead of SEG
    auto allUserOpsCanFitCMX = llvm::all_of(wrapperOp->getUsers(), [&](auto user) {
        Byte requiredCMX = VPUIP::getRequiredCMXSize(user);

        // replace the original operand's required cmx size with new one
        requiredCMX -= getTotalSize(wrapperOp->getResult(0));
        requiredCMX += getTotalSize(topBuffer);
        return requiredCMX <= Byte(_cmxSize);
    });
    if (!allUserOpsCanFitCMX) {
        return nullptr;
    }

    auto topBufferUsers = topBuffer.getUsers();
    for (auto user : topBufferUsers) {
        if (!user->hasOneUse()) {
            return nullptr;
        }
        auto anotherSubView = mlir::dyn_cast<VPUIP::SubViewOp>(user);
        if (anotherSubView == nullptr || !VPUIP::isOpOnlySplitOnDim(anotherSubView, Dims4D::Act::N)) {
            return nullptr;
        }
        auto tilingCopy = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*anotherSubView.getResult().getUsers().begin());
        if (tilingCopy == nullptr || !isTilingCopyOpSegmentedOnN(tilingCopy) ||
            doesTilingCopyOpHasStridedOuput(tilingCopy)) {
            return nullptr;
        }
    }

    return topBuffer;
}

//
// DuplicatedCopyWithCMXCopy
//

/*
Remove copy op by change duplicated buffer into non-distributed buffer

       Distributed Buffer(Duplicated)                   Distributed Buffer(Duplicated)
             |                                                 |
       TilingCopy(CMX2DDR)                               NonDistributedCast
              |                       ==>                      |
         [PureViewLikeOps]                                [PureViewLikeOps]
              |
         Copy(DDR2CMX)
              |

*/

class DuplicatedCopyWithCMXCopy : public mlir::OpRewritePattern<VPUIP::NCEClusterTilingOp> {
public:
    DuplicatedCopyWithCMXCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTilingOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTilingOp tilingCopy,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DuplicatedCopyWithCMXCopy::matchAndRewrite(VPUIP::NCEClusterTilingOp tilingCopy,
                                                               mlir::PatternRewriter& rewriter) const {
    if (!tilingCopy->hasOneUse()) {
        return mlir::failure();
    }
    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(tilingCopy.getInnerTaskOp());
    if (copyOp == nullptr || VPUIP::isCopyFromDDR(copyOp) || !VPUIP::isCopyToDDR(copyOp)) {
        return mlir::failure();
    }

    auto tilingCopyInput = tilingCopy.getInputs()[0];
    auto inType = VPUIP::extractDataType(tilingCopyInput).dyn_cast<VPUIP::DistributedBufferType>();
    if (inType == nullptr) {
        return mlir::failure();
    }
    if (inType.getDistribution().getNumTiles() == nullptr) {
        return mlir::failure();
    }
    auto mode = inType.getDistribution().getMode().getValue();
    if (!VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    auto inStrides = inType.getStrides();
    auto outStrides = VPUIP::extractDataType(tilingCopy.getOutputs()[0]).dyn_cast<vpux::NDTypeInterface>().getStrides();
    if (inStrides != outStrides) {
        return mlir::failure();
    }

    SmallVector<mlir::Operation*> viewLikeOps;
    auto userOp = *tilingCopy->getUsers().begin();
    while (mlir::isa<VPUIP::GenericReshapeOp, VPUIP::ShapeCastOp, VPUIP::PermuteCastOp>(userOp) &&
           userOp->hasOneUse()) {
        viewLikeOps.push_back(userOp);
        userOp = *userOp->getUsers().begin();
    }
    auto userCopy = mlir::dyn_cast<VPUIP::CopyOp>(userOp);
    if (userCopy == nullptr || VPUIP::isCopyToDDR(userCopy)) {
        return mlir::failure();
    }

    auto userInType = VPUIP::extractDataType(userCopy.getInput()).cast<vpux::NDTypeInterface>();
    auto userOutType = VPUIP::extractDataType(userCopy.getOutput()).cast<vpux::NDTypeInterface>();
    if (userInType.changeMemSpace(userOutType.getMemSpace()) != userOutType) {
        return mlir::failure();
    }
    auto symbolAttr = userOutType.getMemSpace();
    auto innerOutputType = VPUIP::extractDataType(copyOp.getOutput()).dyn_cast<vpux::NDTypeInterface>();
    auto newOutType = innerOutputType.changeMemSpace(symbolAttr);
    rewriter.setInsertionPointAfter(tilingCopy);
    auto castOp =
            rewriter.create<VPUIP::NonDistributedCastOp>(tilingCopy->getLoc(), newOutType, tilingCopy.getInputs()[0]);

    _log.trace("Create NonDistributedCast op at '{0}'", tilingCopy->getLoc());
    auto newOutput = castOp.getOutput();
    for (auto viewLikeOp : viewLikeOps) {
        mlir::IRMapping mapper;
        mapper.map(viewLikeOp->getOperands(), ArrayRef({newOutput}));
        auto* newViewLikeOp = rewriter.clone(*viewLikeOp, mapper);
        auto viewLikeOutType = VPUIP::extractDataType(viewLikeOp->getResult(0)).cast<vpux::NDTypeInterface>();
        auto newViewLikeOutType = viewLikeOutType.changeMemSpace(newOutType.getMemSpace());
        newViewLikeOp->getResult(0).setType(newViewLikeOutType);
        newOutput = newViewLikeOp->getResult(0);
    }
    for (auto viewLikeOp : viewLikeOps) {
        viewLikeOp->dropAllUses();
        rewriter.eraseOp(viewLikeOp);
    }
    tilingCopy->dropAllUses();
    rewriter.eraseOp(tilingCopy);
    rewriter.replaceOp(userCopy, {newOutput});

    return mlir::success();
}

//
// FuseCopiesThroughReshape
//

/*
  Fuse copy(with strided input) with clusterTiling copy through reshape

    SubView(Strided input)                            SubView(Strided input)
              |                                                |
        Copy(DDR2DDR)                               ClusterTilingCopy(DDR2CMX)
              |                       ==>                      |
         GenericReshape                                GenericReshape
              |
     ClusterTilingCopy(DDR2CMX)

*/

class FuseCopiesThroughReshape : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    FuseCopiesThroughReshape(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseCopiesThroughReshape::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("FuseCopiesThroughReshape: Copy at {0}", copyOp->getLoc());

    auto wrapperOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    if (wrapperOp) {
        return mlir::failure();
    }

    auto copyOpInput = copyOp.getInput();
    auto copyOpInputType = copyOpInput.getType().cast<vpux::NDTypeInterface>();
    const auto inReqs = StrideReqs::compact(copyOpInputType.getRank());
    if (inReqs.checkStrides(copyOpInputType)) {
        _log.trace("The input has no strides");
        return mlir::failure();
    }
    if (!copyOp->hasOneUse()) {
        return mlir::failure();
    }

    auto reshapeOp = mlir::dyn_cast<VPUIP::GenericReshapeOp>(*copyOp.getOutput().getUsers().begin());
    if (reshapeOp == nullptr) {
        return mlir::failure();
    }
    if (!reshapeOp->hasOneUse()) {
        return mlir::failure();
    }

    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*reshapeOp->getResult(0).getUsers().begin());
    if (clusterTilingOp == nullptr) {
        return mlir::failure();
    }
    auto userClusterCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(clusterTilingOp.getInnerTaskOp());
    if (userClusterCopyOp == nullptr) {
        return mlir::failure();
    }

    auto origReshapeOutType = reshapeOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto origReshapeInType = reshapeOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outBuffer = clusterTilingOp.getOutputBuffs()[0];
    auto outBufAlloc = VPUIP::getRootAlloc<VPURT::AllocDistributed>(outBuffer);
    if (outBufAlloc == nullptr) {
        // The case with pure view-like ops chain is not supported yet.
        // E#122314: support the pure view-like ops chain
        return mlir::failure();
    }

    auto origDistrType = outBuffer.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto origDistrAttr = origDistrType.getDistribution();
    auto getDistributedAxesMapping = vpux::VPUIP::getDistributedAxesMappingAfterShapeChanged(
            origReshapeOutType, origReshapeInType, origDistrAttr, _log);
    if (mlir::failed(getDistributedAxesMapping)) {
        return mlir::failure();
    }
    auto axesMapping = getDistributedAxesMapping.value();
    if (axesMapping.first == -1 || axesMapping.second == -1) {
        return mlir::failure();
    }
    auto newDistributedBeforeShapeChange = vpux::VPUIP::changeDistributedAxisOnDistributionInfoAttr(
            origDistrAttr, axesMapping.first, axesMapping.second, origReshapeInType.getShape());
    auto ctx = copyOp->getContext();
    const auto newOutputElemType = origReshapeInType.getElementType();
    const auto order = mlir::AffineMapAttr::get(origReshapeInType.getDimsOrder().toAffineMap(ctx));
    auto newDistributedBufferType =
            VPUIP::DistributedBufferType::get(ctx, origReshapeInType.getShape().raw(), newOutputElemType, order,
                                              origDistrType.getMemSpace(), newDistributedBeforeShapeChange);
    if (!VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps<VPUIP::DistributedBufferType>(
                origDistrType, newDistributedBufferType)) {
        return mlir::failure();
    }
    outBuffer.setType(newDistributedBufferType);

    SmallVector<mlir::Value> inputsOutputOperands = {copyOpInput, outBuffer};
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(copyOp->getLoc(), newOperands[0], newOperands[1]);
    };
    auto newClusterTilingOp = rewriter.create<VPUIP::NCEClusterTilingOp>(copyOp->getLoc(), outBuffer.getType(),
                                                                         inputsOutputOperands, bodyBuilder);
    VPUIP::moveRootAllocBefore(outBufAlloc, newClusterTilingOp);
    rewriter.replaceOpWithNewOp<VPUIP::GenericReshapeOp>(clusterTilingOp, origDistrType,
                                                         newClusterTilingOp->getResult(0));
    auto origAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(copyOp.getOutputBuff());
    copyOp.getOutput().replaceAllUsesWith(newClusterTilingOp.getResult(0));
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(origAlloc);
    _log.trace("Successfully fused copies through reshape");
    return mlir::success();
}

class SubViewWithCopy : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    SubViewWithCopy(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool hasTrivialStrides(vpux::NDTypeInterface ndType) const;
    SmallVector<int64_t> trimTrivialDims(vpux::NDTypeInterface ndType) const;
    bool isTrivialCopy(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) const;
    mlir::Value getSuitableSubViewPatternSourceBuffer(VPUIP::CopyOp origOp, vpux::Logger log) const;

    Logger _log;
};

bool SubViewWithCopy::hasTrivialStrides(vpux::NDTypeInterface ndType) const {
    const auto elemTypeBitWidth = ndType.getElemTypeSize();
    const auto actStrides = ndType.getStrides();
    for (size_t i = 1; i < actStrides.size(); i++) {
        if (actStrides[Dim(i)] != elemTypeBitWidth) {
            return false;
        }
    }

    return true;
}

SmallVector<int64_t> SubViewWithCopy::trimTrivialDims(vpux::NDTypeInterface ndType) const {
    const auto order = ndType.getDimsOrder();
    const auto shape = order.toMemoryOrder(ndType.getShape());
    const auto isTrivialDim = [](const int64_t dim) -> bool {
        return dim != 1;
    };
    const auto firstNonTrivialDim = std::find_if(shape.begin(), shape.end(), isTrivialDim);
    return SmallVector<int64_t>(firstNonTrivialDim, shape.end());
}

bool SubViewWithCopy::isTrivialCopy(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) const {
    const auto inMemShape = trimTrivialDims(inType);
    const auto outMemShape = trimTrivialDims(outType);

    if (inMemShape.size() != outMemShape.size()) {
        return false;
    }
    for (size_t idx = 1; idx < inMemShape.size(); idx++) {
        if (inMemShape[idx] != outMemShape[idx]) {
            return false;
        }
    }

    return hasTrivialStrides(inType) && hasTrivialStrides(outType);
}

mlir::Value SubViewWithCopy::getSuitableSubViewPatternSourceBuffer(VPUIP::CopyOp copyOp, vpux::Logger log) const {
    auto maybeSubView = copyOp.getInput().getDefiningOp<VPUIP::SubViewOp>();
    if (maybeSubView == nullptr) {
        log.trace("SubViewWithCopy::getSuitableSubViewPatternSourceBuffer: input producer is not a SubView.");
        return nullptr;
    }

    auto inType = maybeSubView.getSource().getType().cast<NDTypeInterface>();
    auto outType = maybeSubView.getResult().getType().cast<NDTypeInterface>();
    // This check must be less strict.
    // The rewriter must be able to process copies of 3-d compact (non-strided) tensors.
    // However, the measurements show that the performance is even worse in that case.
    // The root cause is unclear.
    // [Track number: E#139988]
    if (!isTrivialCopy(inType, outType)) {
        log.trace("SubViewWithCopy::getSuitableSubViewPatternSourceBuffer: strided copies cannot be replaced with a "
                  "ViewOp.");
        return nullptr;
    }

    const auto inputType = copyOp.getInput().getType().cast<NDTypeInterface>();
    const auto outputType = copyOp.getOutput().getType().cast<NDTypeInterface>();
    if (inputType.getMemSpace() != outputType.getMemSpace()) {
        log.trace("SubViewWithCopy::getSuitableSubViewPatternSourceBuffer: CMX <-> DRAM transfers cannot be replaced "
                  "with a ViewOp.");
        return nullptr;
    }

    const auto staticOffsets = parseIntArrayAttr<int64_t>(maybeSubView.getStaticOffsets());
    const auto hasNonZeroOffsets = llvm::any_of(staticOffsets, [](const int64_t offset) {
        return offset != 0;
    });
    if (hasNonZeroOffsets) {
        return nullptr;
    }

    return maybeSubView.getSource();
}

mlir::LogicalResult SubViewWithCopy::matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto nestedLogger = _log.nest();
    auto subviewInput = getSuitableSubViewPatternSourceBuffer(origOp, nestedLogger);
    if (subviewInput == nullptr) {
        return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<VPUIP::ViewOp>(origOp, origOp.getType(), subviewInput);

    return mlir::success();
}

//
// OptimizeCopiesPass
//

class OptimizeCopiesPass final : public VPUIP::OptimizeCopiesBase<OptimizeCopiesPass> {
public:
    explicit OptimizeCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto cmxSize = VPU::getTotalCMXSize(module).count();

    // Note the below patterns exec order is defined by "benefitLevels" at the head
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CopyOpSequence>(&ctx, benefitLevels[0], _log);
    patterns.add<NCEClusterCopyOpSequence>(&ctx, benefitLevels[0], _log);
    patterns.add<CMXToCMXCopy>(&ctx, benefitLevels[1], _log);
    patterns.add<DDRToDDRCopyOfNCECluster>(&ctx, benefitLevels[2], _log);
    patterns.add<ConcatViewWithCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<ConcatViewWithTilingCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<FuseCopyToTheFrontOfTilingCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<FuseCopyToTheBackOfTilingCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<SubViewWithTilingCopy>(&ctx, benefitLevels[3], cmxSize, _log);
    patterns.add<DuplicatedCopyWithCMXCopy>(&ctx, benefitLevels[3], _log);
    patterns.add<FuseCopiesThroughReshape>(&ctx, benefitLevels[3], _log);
    patterns.add<SubViewWithCopy>(&ctx, benefitLevels[3], _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeCopiesPass(Logger log) {
    return std::make_unique<OptimizeCopiesPass>(log);
}
