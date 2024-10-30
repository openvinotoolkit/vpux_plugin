//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// VerticalFusionTilingRewriter
//

class VerticalFusionTilingRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    VerticalFusionTilingRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx),
              _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    void adjustInputShape(mlir::PatternRewriter& rewriter, mlir::Operation* operation, InputTiling& inputTiling,
                          mlir::IRMapping& mapper, mlir::SetVector<mlir::Value>& slicedOperands,
                          TilingStorage& tilingStorage, int64_t tilingIndex, Dim axis) const;

    void processOffset(mlir::Value operand, mlir::IRMapping& mapper, mlir::SetVector<mlir::Value>& slicedOperands,
                       TilingStorage& tilingStorage, TileInfo& originalTiling, int64_t tilingIndex, Dim axis,
                       ShapeRef expectedShape) const;

    bool _enableVerticalFusionPipelining;
    Logger _log;
};

void VerticalFusionTilingRewriter::processOffset(mlir::Value operand, mlir::IRMapping& mapper,
                                                 mlir::SetVector<mlir::Value>& slicedOperands,
                                                 TilingStorage& tilingStorage, TileInfo& originalTiling,
                                                 int64_t tilingIndex, Dim axis, ShapeRef expectedShape) const {
    auto& offset = originalTiling.offsets[axis];
    if (offset == 0) {
        return;
    }

    const auto shiftOffsetBlockArg = [&](mlir::BlockArgument blockArg) {
        // in case previous operation is outside the block and
        // operand is block argument, correct offset on its offset from tiling info
        if (blockArg == nullptr) {
            return;
        }

        const auto storageInfo = tilingStorage.get(blockArg.getArgNumber(), tilingIndex);
        VPUX_THROW_WHEN(!storageInfo.has_value(), "Tiling info for argument {0} with index {1} not found", blockArg,
                        tilingIndex);

        auto tileInfo = storageInfo.value();

        VPUX_THROW_UNLESS(static_cast<size_t>(axis.ind()) < tileInfo.shape.size(), "Got invalid tiling shape size {0}",
                          tileInfo.shape.size());
        const auto inputOffset = tileInfo.offsets[axis];
        const auto inputDimShape = tileInfo.shape[axis];
        const auto origDimSize = originalTiling.shape[axis];

        _log.trace("Input Offset {0}, shape {1} ==> offset: {2}, shape: {3} ", inputOffset, inputDimShape, offset,
                   origDimSize);

        VPUX_THROW_WHEN((inputOffset > offset) || ((inputOffset + inputDimShape) < (offset + origDimSize)),
                        "Got invalid offsets");
        offset -= inputOffset;
    };

    if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
        shiftOffsetBlockArg(blockArg);
        return;
    }

    auto operandOp = operand.getDefiningOp();
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(operandOp);
    const auto isOne = [](auto i) {
        return i == 1;
    };

    // if the operation doesn't add offset, but its input might be affected by
    // additional offsets of operations before it, try to find all shifts until
    // the beginning of VF block
    if ((operandOp != nullptr && operandOp->hasTrait<VPU::EltwiseOp>()) ||
        (nceOp != nullptr && llvm::all_of(nceOp.getKernelSizeVal(), isOne))) {
        auto blockOperand = operand;
        std::queue<mlir::Value> operands;

        operands.push(blockOperand);
        mlir::SetVector<mlir::Operation*> passedSlice;

        while (!operands.empty()) {
            blockOperand = operands.front();
            operands.pop();

            auto tiledOperand = mapper.lookupOrNull(blockOperand);
            if (tiledOperand == nullptr) {
                continue;
            }

            mlir::Operation* sliceOperation = nullptr;
            if (slicedOperands.contains(tiledOperand)) {
                sliceOperation = tiledOperand.getDefiningOp();
            } else if (auto tiledOperation = tiledOperand.getDefiningOp()) {
                for (auto item : tiledOperation->getOperands()) {
                    if (slicedOperands.contains(item)) {
                        sliceOperation = item.getDefiningOp();
                        break;
                    }
                }
            }

            // if we have found new slice, adjust the offset on it
            if (auto sliceOp = mlir::dyn_cast_or_null<VPU::SliceOp>(sliceOperation)) {
                if (!passedSlice.contains(sliceOp)) {
                    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
                    offset -= sliceOffsets[axis.ind()];
                    passedSlice.insert(sliceOp);
                }
            }

            if (mlir::isa_and_nonnull<mlir::BlockArgument>(blockOperand)) {
                continue;
            }

            auto* blockOperation = blockOperand.getDefiningOp();
            if (blockOperation == nullptr) {
                continue;
            }

            operands.push(blockOperation->getOperand(0));
            if (blockOperation->hasTrait<VPU::EltwiseOp>() && blockOperation->getNumOperands() > 1) {
                bool isArgumentBreak = false;
                // if there is Eltwise-like operation which one of operands
                // is block argument, don't go further, no need to adjust offset on other argument
                for (auto operandEltwise : blockOperation->getOperands()) {
                    if (mlir::isa<mlir::BlockArgument>(operandEltwise) && operandEltwise.hasOneUse()) {
                        blockOperand = operandEltwise;
                        isArgumentBreak = true;
                        break;
                    }
                }
                if (isArgumentBreak) {
                    break;
                }
                operands.push(blockOperation->getOperand(1));
            }
        }

        shiftOffsetBlockArg(blockOperand.dyn_cast<mlir::BlockArgument>());
        return;
    }

    if (auto parentTilingOp = operand.getDefiningOp<VPU::TilingBuilderOpInterface>()) {
        // in case there is parent operation which has tiling info
        // restore original tiling of that op based on original tiling info
        // and correct offset on it
        auto inputOldTiling = parentTilingOp.backInferTileInfo(originalTiling, _log);

        VPUX_THROW_WHEN(inputOldTiling.tiles.empty() ||
                                static_cast<size_t>(axis.ind()) >= inputOldTiling.tiles[0].offsets.size(),
                        "Got invalid offsets");

        offset -= inputOldTiling.tiles[0].offsets[axis];
        return;
    }

    offset = expectedShape[axis] - originalTiling.shape[axis];
}

/*
 This function slice to original tile shape in case bigger tile size was chosen
 during backpropagation process.
 In this case adjust shapes to original one by slicing
*/
void VerticalFusionTilingRewriter::adjustInputShape(mlir::PatternRewriter& rewriter, mlir::Operation* operation,
                                                    InputTiling& inputTiling, mlir::IRMapping& mapper,
                                                    mlir::SetVector<mlir::Value>& slicedOperands,
                                                    TilingStorage& tilingStorage, int64_t tilingIndex, Dim axis) const {
    VPUX_THROW_WHEN(inputTiling.tiles.size() < operation->getOperands().size(),
                    "Number of operands {0} is more than number of operand tiles {1}", operation->getOperands().size(),
                    inputTiling.tiles.size());
    for (auto op : operation->getOperands() | indexed) {
        auto operand = op.value();
        auto opIndex = op.index();

        auto expectedOp = mapper.lookupOrNull(operand);
        if (expectedOp == nullptr) {
            continue;
        }

        auto originalTiling = inputTiling.tiles[opIndex];
        const auto expectedShape = getShape(expectedOp);
        const auto expectedOpSize = expectedShape.totalSize();
        const auto originalOpSize = originalTiling.shape.totalSize();
        if (expectedOpSize == originalOpSize) {
            continue;
        }

        VPUX_THROW_WHEN(
                expectedOpSize < originalOpSize,
                "Original shape size for operand {0} is bigger than current one. Current size {1}, original size {2}",
                operand, expectedOpSize, originalOpSize);

        VPUX_THROW_WHEN(expectedShape.size() != originalTiling.shape.size(),
                        "Expected shape {0} and original one {1} must have same rank", expectedShape,
                        originalTiling.shape);

        // correct offset of operations based on offsets of block argument
        // In case the output of previous operation is bigger than expected
        // which might happen when bigger tile was chosen for same block argument
        // slice operation is needed after the output with correct offsets
        // calculated based on tiling information of current operation and previous one
        _log.trace("Offset before {0}, shape {1}", originalTiling.offsets, originalTiling.shape);

        processOffset(operand, mapper, slicedOperands, tilingStorage, originalTiling, tilingIndex, axis, expectedShape);
        _log.trace("Offset after {0}", originalTiling.offsets);

        const auto valName = printToString("input {0}", opIndex);
        auto opSlice = makeTile(rewriter, operation->getLoc(), expectedOp, originalTiling, valName);

        slicedOperands.insert(opSlice);

        mapper.map(operand, opSlice);
    }
}

mlir::LogicalResult VerticalFusionTilingRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy().cast<mlir::ArrayAttr>());

    const auto numTiledAxis = llvm::count_if(tilingStrategy, [](auto num) {
        return num > 1;
    });

    VPUX_THROW_WHEN(numTiledAxis != 1, "VF tiling is supported only for one axis");

    auto maxTiledLen = std::max_element(tilingStrategy.begin(), tilingStrategy.end());

    if (maxTiledLen == tilingStrategy.end()) {
        return mlir::failure();
    }

    VPUX_THROW_WHEN(*maxTiledLen <= 1, "There is no tiling for VF");

    auto operationStorage = std::make_unique<TilingOperationStorage>();
    auto tilingStorage = restoreTilingRegions(vfOp, _log, operationStorage);

    auto vfConfig = VFConfig(vfOp, _enableVerticalFusionPipelining);

    SmallVector<mlir::Value> resultTileVals;
    resultTileVals.reserve(*maxTiledLen);
    SmallVector<Shape> resultTileOffsets;
    mlir::IRMapping mapper;
    llvm::SetVector<mlir::Value> slicedOperands;

    auto dim = Dim(std::distance(tilingStrategy.begin(), maxTiledLen));
    for (auto index : irange(*maxTiledLen)) {
        mlir::Value currentResult;
        slicedOperands.clear();
        Shape currentTile;
        DenseMap<size_t, mlir::Operation*> argMapper;
        for (auto* op : vfConfig.getVFOperations()) {
            for (auto operand : op->getOperands()) {
                if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                    const auto valName = printToString("input {0}", index);
                    auto origInput = vfOp.getOperand(blockArg.getArgNumber());
                    auto tileInfo = tilingStorage.get(blockArg.getArgNumber(), index);

                    VPUX_THROW_WHEN(!tileInfo.has_value(),
                                    "Couldn't find tile information for argument {0} and tile {1}",
                                    blockArg.getArgNumber(), index);
                    auto operandTile = VPU::makeTile(rewriter, op->getLoc(), origInput, tileInfo.value(), valName);

                    mapper.map(operand, operandTile);
                }
            }

            auto inputTiling = operationStorage->get(op, index);

            VPUX_THROW_WHEN(!inputTiling.has_value(), "Couldn't find tile information for operation {0} and tile {1}",
                            *op, index);

            const auto inputTilingPair = inputTiling.value();
            auto inputTilingInfo = inputTilingPair.first;
            adjustInputShape(rewriter, op, inputTilingInfo, mapper, slicedOperands, tilingStorage, index, dim);

            auto* copiedOp = rewriter.clone(*op, mapper);
            currentResult = copiedOp->getResult(0);

            currentTile = inputTilingPair.second.offsets;
            const auto baseResType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
            if (auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(copiedOp)) {
                tiledBuilderOp.adjustAttrs(inputTilingInfo, inputTilingPair.second);
            } else if (auto tiledViewOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(copiedOp)) {
                tiledViewOp.adjustAttrs(inputTilingInfo, inputTilingPair.second, baseResType.getShape());
            }
            const auto tiledResType =
                    baseResType.extractDenseTile(inputTilingPair.second.offsets, inputTilingPair.second.shape);

            currentResult.setType(tiledResType);
            mapper.map(op->getResult(0), currentResult);
        }

        resultTileVals.push_back(currentResult);
        resultTileOffsets.push_back(currentTile);
    }

    if (vfConfig.isPipelined()) {
        // For VF region
        //      DPU_0_0 -> SW_0 -> DPU_0_1 ->
        //      DPU_1_0 -> SW_1 -> DPU_1_1 -> ...
        // Reorder from
        //      DPU_0_0, SW_0, DPU_0_1, DPU_1_0, SW_1, ...
        // to
        //      DPU_0_0, SW_0, DPU_1_0, SW_1, DPU_0_1, ...
        // to support the parallelization of [SW_0, DPU_1_0] [SW_1, DPU_0_1]
        // The new order is aligned with the scheduler,
        // which follows IR order in terms of compute operations
        for (auto index = 0; index < *maxTiledLen - 1; ++index) {
            auto nextOp = resultTileVals[index + 1].getDefiningOp()->getOperand(0).getDefiningOp();
            resultTileVals[index].getDefiningOp()->moveAfter(nextOp);
        }
    }

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(vfOp, vfOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                               ArrayRef(resultTileOffsets));

    return mlir::success();
}

//
// VfTilingPass
//

class VfTilingPass final : public VfTilingBase<VfTilingPass> {
public:
    explicit VfTilingPass(bool enableVerticalFusionPipelining, Logger log)
            : _enableVerticalFusionPipelining(enableVerticalFusionPipelining) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableVerticalFusionPipelining = false;
};

mlir::LogicalResult VfTilingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableVerticalFusionPipelining.hasValue()) {
        _log.trace("Overloading VfTilingPass argument by MLIR variable");
        _enableVerticalFusionPipelining = enableVerticalFusionPipelining;
    }
    return mlir::success();
}

//
// safeRunOnModule
//

void VfTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::VerticalFusionOp>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<VerticalFusionTilingRewriter>(&ctx, _enableVerticalFusionPipelining, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createVfTilingPass
//

std::unique_ptr<mlir::Pass> VPU::createVfTilingPass(bool enableVerticalFusionPipelining, Logger log) {
    return std::make_unique<VfTilingPass>(enableVerticalFusionPipelining, log);
}
