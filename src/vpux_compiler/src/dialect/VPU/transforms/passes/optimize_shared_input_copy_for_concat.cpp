//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <llvm/ADT/SetOperations.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPU;

namespace {

bool checkMemoryKind(mlir::Value value, VPU::MemoryKind kind) {
    return value.getType().cast<vpux::NDTypeInterface>().getMemoryKind() == kind;
}

bool isCopyOp(mlir::Operation* op) {
    if (mlir::isa_and_nonnull<VPU::CopyOp>(op)) {
        return true;
    }

    auto clusterTilingOp = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(op);
    if (clusterTilingOp == nullptr) {
        return false;
    }
    return mlir::isa_and_nonnull<VPU::CopyOp>(clusterTilingOp.getInnerTaskOp());
}

bool isCopyCMX2DDR(mlir::Operation* op) {
    if (!isCopyOp(op)) {
        return false;
    }
    return checkMemoryKind(op->getOperand(0), VPU::MemoryKind::CMX_NN) &&
           checkMemoryKind(op->getResult(0), VPU::MemoryKind::DDR);
}

bool isCopyDDR2CMX(mlir::Operation* op) {
    if (!isCopyOp(op)) {
        return false;
    }
    return checkMemoryKind(op->getOperand(0), VPU::MemoryKind::DDR) &&
           checkMemoryKind(op->getResult(0), VPU::MemoryKind::CMX_NN);
}

SmallVector<int64_t> getOffsetsFromConcat(mlir::Value input, VPU::ConcatOp concatOp) {
    auto offsets = parseIntArrayOfArrayAttr<int64_t>(concatOp.getStaticOffsets().value());
    for (auto item : concatOp.getInputs() | indexed) {
        if (item.value() == input) {
            return offsets[item.index()];
        }
    }
    VPUX_THROW("input {0} is not input of ConcatOp {1}", input.getLoc(), concatOp->getLoc());
}

NDTypeInterface getConcatDistributedType(VPU::DistributedTypeInterface origType, ShapeRef shape) {
    auto distributedDataType = origType.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
    const auto typeComponents = TypeComponents().setShape(shape).setElementType(distributedDataType.getElementType());

    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedDataType.getDistribution())) {
        auto distribution = distributedDataType.getDistribution();
        if (auto sparseType = origType.dyn_cast<VPU::SparseTensorType>()) {
            distribution = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
        }

        auto newDistributedAttr =
                getConcatExplicitDistributedAttrForNewShape(distribution, shape, origType.getContext());
        return origType.changeTypeComponentsForExplicitDistribution(typeComponents, newDistributedAttr)
                .cast<NDTypeInterface>();
    }

    return origType.cast<NDTypeInterface>().changeTypeComponents(typeComponents);
}

mlir::DenseSet<int64_t> getConcatAxes(VPU::ConcatOp concat) {
    mlir::DenseSet<int64_t> concatAxes;
    const auto staticOffsets = concat.getStaticOffsets().value();
    const auto offsets = parseIntArrayOfArrayAttr<int64_t>(staticOffsets);
    for (auto& offset : offsets) {
        for (size_t axis = 0; axis < offset.size(); ++axis) {
            if (offset[axis] != 0) {
                concatAxes.insert(axis);
            }
        }
    }
    return concatAxes;
}

int64_t getSliceDimSize(VPU::SliceOp sliceOp) {
    auto sliceInShape = getShape(sliceOp.getSource());
    auto sliceOutShape = getShape(sliceOp.getResult());
    auto sliceDimSize = llvm::count_if(irange(sliceInShape.size()), [&](auto idx) {
        return sliceInShape[Dim(idx)] != sliceOutShape[Dim(idx)];
    });
    return sliceDimSize;
}

//
// SharedCopyInputRewriter
//

class SharedCopyInputRewriter final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    SharedCopyInputRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log) {
        this->setDebugName("SharedCopyInputRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool meetConcatPattern(VPU::ConcatOp) const;

    std::optional<mlir::Value> createNewBranchInput(mlir::Value concatInput, VPU::ConcatOp concat,
                                                    VPU::SliceOp sliceUser, SmallVector<int64_t>& newConcatOffset,
                                                    mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::LogicalResult SharedCopyInputRewriter::matchAndRewrite(VPU::ConcatOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto hasCopyInput = llvm::any_of(origOp.getInputs(), [&](const auto& input) {
        return isCopyCMX2DDR(input.getDefiningOp());
    });
    if (!hasCopyInput) {
        return mlir::failure();
    }

    SmallVector<VPU::ConcatOp> concatsWithSharedCopyInput;
    for (auto user : origOp->getUsers()) {
        if (auto concat = mlir::dyn_cast<VPU::ConcatOp>(user)) {
            if (meetConcatPattern(concat)) {
                concatsWithSharedCopyInput.push_back(concat);
            }
        }
    }

    if (concatsWithSharedCopyInput.size() < 2) {
        return mlir::failure();
    }

    auto ctx = origOp->getContext();
    _log.trace("process shared input copy at'{0}'", origOp->getLoc());

    /*
       Convert this subgraph:
         Copy(CMX2DDR)                       Copy(CMX2DDR)
              |                                |
            Concat                           Concat
           /     \                          /     \
      Concat0    Concat1                  Slice  Slice
         |         |             =>        |       |
       Slice     Slice                    Copy    Copy
         |         |                       |       |
       Copy(DDR2CMX)  Copy(DDR2CMX)     CMXConcat CMXConcat

    */

    for (auto concat : concatsWithSharedCopyInput) {
        _log.trace("propagate ops through concat '{0}'", concat.getLoc());
        llvm::DenseMap<mlir::Operation*, VPU::ConcatOp> newUserMapping;
        mlir::Operation* insertPoint = concat.getOperation();
        for (auto user : concat->getUsers()) {
            SmallVector<mlir::Value> newInputs;
            SmallVector<SmallVector<int64_t>> newConcatOffsets;

            auto slice = mlir::cast<VPU::SliceOp>(user);
            auto copyOp = *user->getUsers().begin();
            for (auto input : concat.getInputs()) {
                SmallVector<int64_t> newConcatOffset;
                auto newInput = createNewBranchInput(input, concat, slice, newConcatOffset, rewriter);
                if (newInput.has_value()) {
                    newInputs.push_back(newInput.value());
                    newConcatOffsets.push_back(newConcatOffset);
                    if (insertPoint->isBeforeInBlock(newInput.value().getDefiningOp())) {
                        insertPoint = newInput.value().getDefiningOp();
                    }
                }
            }
            VPUX_THROW_WHEN(newInputs.empty(), "new slice input is empty");

            rewriter.setInsertionPointAfter(insertPoint);
            auto newConcat = rewriter.create<VPU::ConcatOp>(copyOp->getLoc(), copyOp->getResult(0).getType(), newInputs,
                                                            getIntArrayOfArray(ctx, newConcatOffsets));
            newUserMapping.insert({copyOp, newConcat});
        }

        // Replace copy user with new concat op
        for (auto& item : newUserMapping) {
            rewriter.replaceOp(item.first, item.second);
        }
        // Remove concat op and its user
        for (auto user : llvm::make_early_inc_range(concat->getUsers())) {
            rewriter.eraseOp(user);
        }
        rewriter.eraseOp(concat);
    }
    return mlir::success();
}

bool SharedCopyInputRewriter::meetConcatPattern(VPU::ConcatOp concatOp) const {
    if (!concatOp.getStaticOffsets().has_value()) {
        // Only handle concat op with static offset
        return false;
    }
    auto concatAxes = getConcatAxes(concatOp);
    if (concatAxes.size() != 1) {
        // For concat on multi dims, the input and output tensor of the new copy has strides on multi dims,
        // which may cause accuracy regression when lowering to DMA op
        return false;
    }

    for (auto user : concatOp->getUsers()) {
        if (!user->hasOneUse()) {
            return false;
        }
        auto maybeSliceOp = mlir::dyn_cast<VPU::SliceOp>(user);
        if (maybeSliceOp == nullptr || !maybeSliceOp->hasOneUse()) {
            return false;
        }
        auto sliceDimSize = getSliceDimSize(maybeSliceOp);
        if (sliceDimSize != 1) {
            return false;
        }

        auto userOp = *maybeSliceOp->getUsers().begin();
        if (!isCopyDDR2CMX(userOp)) {
            return false;
        }

        if (auto clusterTilingOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(userOp)) {
            auto outDistributedType = clusterTilingOp.getResult(0).getType().cast<VPU::DistributedTensorType>();
            const auto distAttr = outDistributedType.getDistribution();
            const auto distMode = distAttr.getMode().getValue();
            if (distMode == VPU::DistributionMode::SEGMENTED || distMode == VPU::DistributionMode::OVERLAPPED) {
                const auto numTiles = distAttr.getNumTiles();
                const auto tilingScheme = parseIntArrayAttr<int64_t>(numTiles);
                auto tilingAxis = VPU::getDistributedTilingAxis(tilingScheme);
                if (concatAxes.contains(tilingAxis)) {
                    return false;
                }
            }
        }
    };

    return true;
}

std::optional<mlir::Value> SharedCopyInputRewriter::createNewBranchInput(mlir::Value concatInput, VPU::ConcatOp concat,
                                                                         VPU::SliceOp sliceUser,
                                                                         SmallVector<int64_t>& newConcatOffset,
                                                                         mlir::PatternRewriter& rewriter) const {
    const auto concatOffset = getOffsetsFromConcat(concatInput, concat);
    const auto concatSize = to_small_vector(getShape(concatInput));

    const auto sliceOffset = parseIntArrayAttr<int64_t>(sliceUser.getStaticOffsets());
    const auto sliceSize = parseIntArrayAttr<int64_t>(sliceUser.getStaticSizes());

    // check the overlapp concat input has intersection with the slice output
    SmallVector<int64_t> overlappedOffset(concatSize.size(), 0);
    SmallVector<int64_t> overlappedSize(concatSize.size(), 0);

    for (auto i : irange(sliceOffset.size())) {
        overlappedOffset[i] = std::max(concatOffset[i], sliceOffset[i]);
        auto concatEndOffset = concatOffset[i] + concatSize[i];
        auto sliceEndOffset = sliceOffset[i] + sliceSize[i];
        auto overlappedEndOffset = std::min(sliceEndOffset, concatEndOffset);
        overlappedSize[i] = overlappedEndOffset - overlappedOffset[i];
    }
    auto hasLegalOverLapped = llvm::all_of(overlappedSize, [](const auto& size) {
        return size > 0;
    });

    if (!hasLegalOverLapped) {
        return std::nullopt;
    }
    // Convert overlapped offset to slice offset for the input
    SmallVector<int64_t> newSliceOffset(overlappedSize.size(), 0);
    SmallVector<int64_t> newSliceSize = std::move(overlappedSize);
    newConcatOffset.resize(concatOffset.size());
    for (auto i : irange(concatOffset.size())) {
        newSliceOffset[i] = overlappedOffset[i] - concatOffset[i];
        newConcatOffset[i] = std::max(static_cast<int64_t>(0), concatOffset[i] - sliceOffset[i]);
    }

    rewriter.setInsertionPointAfter(concatInput.getDefiningOp());
    auto newSlice = rewriter.create<VPU::SliceOp>(sliceUser.getLoc(), concatInput, newSliceOffset, newSliceSize);

    auto userCopyOp = *sliceUser->getUsers().begin();
    auto copyOutType = userCopyOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    if (auto copyOp = mlir::dyn_cast<VPU::CopyOp>(userCopyOp)) {
        return rewriter.create<VPU::CopyOp>(userCopyOp->getLoc(), newSlice, copyOutType.getMemSpace());
    }

    auto tilingCopyOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(userCopyOp);
    VPUX_THROW_WHEN(tilingCopyOp == nullptr, "Unexpected user op type at '{0}'", userCopyOp->getLoc());

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        auto newCopyOp = builder.create<VPU::CopyOp>(loc, newOperands[0], copyOutType.getMemSpace());
        builder.create<VPU::YieldOp>(loc, newCopyOp.getOutput());
    };

    auto newShape = getShape(newSlice.getResult());
    auto newOutType = getConcatDistributedType(copyOutType.cast<VPU::DistributedTensorType>(), newShape);
    auto newCopyOp = rewriter.create<VPU::NCEClusterTilingOp>(userCopyOp->getLoc(), newOutType,
                                                              mlir::ValueRange{newSlice}, bodyBuilder);
    return newCopyOp->getResult(0);
}

//
// OptimizeSharedInputCopyForConcatPass
//

class OptimizeSharedInputCopyForConcatPass final :
        public OptimizeSharedInputCopyForConcatBase<OptimizeSharedInputCopyForConcatPass> {
public:
    explicit OptimizeSharedInputCopyForConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;

private:
};

void OptimizeSharedInputCopyForConcatPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SharedCopyInputRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeSharedInputCopyForConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeSharedInputCopyForConcatPass(Logger log) {
    return std::make_unique<OptimizeSharedInputCopyForConcatPass>(log);
}
