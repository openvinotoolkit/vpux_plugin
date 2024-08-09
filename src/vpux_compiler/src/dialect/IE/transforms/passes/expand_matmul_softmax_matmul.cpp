//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fft_ops_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

using AnyRankedTensor = mlir::TypedValue<mlir::RankedTensorType>;

bool opHasOneUse(mlir::Operation* op) {
    return op != nullptr && op->hasOneUse();
}

bool isValidPattern(IE::MatMulOp op) {
    // Pattern should have output 49x32
    auto outType = op.getType().cast<NDTypeInterface>();
    if (outType.getShape()[Dims4D::Act::W] != 32 || outType.getShape()[Dims4D::Act::H] != 49) {
        return false;
    }

    // Pattern should be MatMul1 -> reshapeOp1 -> SoftMax -> reshapeOp2 -> MatMul2
    // Verify -> reshapeOp2
    auto reshapeOp2 = op.getInput1().getDefiningOp<IE::ReshapeOp>();
    if (!opHasOneUse(reshapeOp2)) {
        return false;
    }

    // Verify -> SoftMax
    auto softmaxOp = reshapeOp2.getInput().getDefiningOp<IE::SoftMaxOp>();
    if (!opHasOneUse(softmaxOp)) {
        return false;
    }

    // Verify -> reshapeOp1
    auto reshapeOp1 = softmaxOp.getInput().getDefiningOp<IE::ReshapeOp>();
    if (!opHasOneUse(reshapeOp1)) {
        return false;
    }

    // Verify MatMul1
    auto matMulOp1 = reshapeOp1.getInput().getDefiningOp<IE::MatMulOp>();
    if (!opHasOneUse(matMulOp1)) {
        return false;
    }

    return true;
}

//
// MatMulOpConverter
//

class MatMulOpConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
        setDebugName("MatMulOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp op, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got Operation: '{1}'", getDebugName(), op);

        if (!isValidPattern(op)) {
            return mlir::failure();
        }
        _log.nest().trace("[{0}] MatMul -> Reshape -> SoftMax -> Reshape -> MatMul Pattern found", getDebugName());
        auto reshapeOp2 = op.getInput1().getDefiningOp<IE::ReshapeOp>();
        auto softmaxOp = reshapeOp2.getInput().getDefiningOp<IE::SoftMaxOp>();
        auto reshapeOp1 = softmaxOp.getInput().getDefiningOp<IE::ReshapeOp>();
        auto matMulOp1 = reshapeOp1.getInput().getDefiningOp<IE::MatMulOp>();

        // Pads for expand
        const std::array<int64_t, 4> padsBegin = {0, 0, 0, 0};
        const std::array<int64_t, 4> padsEnd15 = {0, 0, 0, 15};
        const std::array<int64_t, 4> padsEnd32 = {0, 0, 0, 32};
        auto newPadsBeginAttr = getIntArrayAttr(op->getContext(), padsBegin);
        auto newPadsEndAttr15 = getIntArrayAttr(op->getContext(), padsEnd15);
        auto newPadsEndAttr32 = getIntArrayAttr(op->getContext(), padsEnd32);

        // Expand second input of first MatMul 32x49 -> 32x64
        auto newExpand1 = rewriter.create<IE::ExpandOp>(matMulOp1->getLoc(), matMulOp1.getInput2(), newPadsBeginAttr,
                                                        newPadsEndAttr15);
        // Create new MatMul1
        auto newMatmul1 = rewriter.create<IE::MatMulOp>(matMulOp1->getLoc(), matMulOp1.getInput1(), newExpand1,
                                                        matMulOp1.getTransposeA(), matMulOp1.getTransposeB());

        // Create new Reshape1
        auto prevReshapeType = reshapeOp1.getType().cast<NDTypeInterface>();
        auto newReshapeShape = prevReshapeType.getShape();
        auto newShape = Shape{newReshapeShape[Dims4D::Act::N], newReshapeShape[Dims4D::Act::C],
                              newReshapeShape[Dims4D::Act::H], 64};
        auto newType = prevReshapeType.changeShape(newShape);
        auto newShape1Attr = getIntArrayAttr(op->getContext(), newType.getShape().raw());
        auto newReshape1 = rewriter.create<IE::ReshapeOp>(reshapeOp1->getLoc(), newType, newMatmul1, nullptr, false,
                                                          newShape1Attr);

        // Create new SoftMax
        auto newPadSize = 15;  // new dim - old dim (64-49)
        auto newPadSizeAttr = getIntAttr(op->getContext(), newPadSize);
        auto newSoftmaxOp = rewriter.create<IE::SoftMaxOp>(softmaxOp->getLoc(), newReshape1, softmaxOp.getAxisIndAttr(),
                                                           newPadSizeAttr);

        // Expand second input of MatMul2 49x32 -> 49x64
        auto newExpand2 =
                rewriter.create<IE::ExpandOp>(op->getLoc(), op.getInput2(), newPadsBeginAttr, newPadsEndAttr32);

        // Create new Reshape2
        auto prevReshapeType2 = reshapeOp2.getType().cast<NDTypeInterface>();
        auto newReshapeShape2 = prevReshapeType2.getShape();
        auto newShape2 = Shape{newReshapeShape2[Dims4D::Act::N], newReshapeShape2[Dims4D::Act::C],
                               newReshapeShape2[Dims4D::Act::H], 64};
        auto newType2 = prevReshapeType2.changeShape(newShape2);
        auto newShape2Attr = getIntArrayAttr(op->getContext(), newType2.getShape().raw());
        auto newReshape2 = rewriter.create<IE::ReshapeOp>(reshapeOp2->getLoc(), newType2, newSoftmaxOp, nullptr, false,
                                                          newShape2Attr);

        // Create new MatMul2
        auto newMatmul2 = rewriter.create<IE::MatMulOp>(op->getLoc(), newReshape2, newExpand2, op.getTransposeA(),
                                                        op.getTransposeB());

        // Slice MatMul back to 49x32
        auto shape = getShape(op.getOutput());
        auto offsetsSize = shape.size();
        auto offsets = SmallVector<int64_t>(offsetsSize, 0);
        auto slice = rewriter.create<IE::SliceOp>(newMatmul2->getLoc(), newMatmul2.getOutput(), offsets, shape);

        rewriter.replaceOp(op, slice);
        return mlir::success();
    }

private:
    Logger _log;
};

//
// ExpandMatMulSoftMaxMatMulPass
//

class ExpandMatMulSoftMaxMatMulPass final : public IE::ExpandMatMulSoftMaxMatMulBase<ExpandMatMulSoftMaxMatMulPass> {
public:
    explicit ExpandMatMulSoftMaxMatMulPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ExpandMatMulSoftMaxMatMulPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MatMulOpConverter>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createExpandMatMulSoftMaxMatMulPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createExpandMatMulSoftMaxMatMulPass(Logger log) {
    return std::make_unique<ExpandMatMulSoftMaxMatMulPass>(log);
}
