//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

class UnrollFullyConnected final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    UnrollFullyConnected(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
        setDebugName("UnrollFullyConnected");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkConcat(IE::ConcatOp concat) const;
    SmallVector<mlir::Value> findMatMulInputs(IE::FullyConnectedOp matMulOp) const;
    SmallVector<mlir::Value> splitLeftInput(const mlir::Value lhs, const int64_t groups,
                                            mlir::PatternRewriter& rewriter) const;
    SmallVector<mlir::Value> buildMatMuls(const mlir::Location loc, const mlir::ValueRange lhsInputs,
                                          const mlir::ValueRange rhsInputs, mlir::PatternRewriter& rewriter) const;
    SmallVector<mlir::Value> accumulateMatMuls(const mlir::ValueRange matMuls, mlir::PatternRewriter& rewriter) const;
    SmallVector<mlir::Value> reshapeTo2d(mlir::ValueRange values, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

bool UnrollFullyConnected::checkConcat(IE::ConcatOp concat) const {
    const auto concatInputs = concat.getInputs();
    const auto concatOutput = concat.getOutput();
    const auto outputShape = getShape(concatOutput);
    const Shape expectedInputShape = {1, outputShape[Dim(1)], outputShape[Dim(2)]};
    const auto isShapeCompatible = [&expectedInputShape](const mlir::Value in) -> bool {
        const auto inputShape = getShape(in);
        return inputShape == expectedInputShape;
    };
    return std::all_of(concatInputs.begin(), concatInputs.end(), isShapeCompatible);
}

SmallVector<mlir::Value> UnrollFullyConnected::findMatMulInputs(IE::FullyConnectedOp matMulOp) const {
    // Left-hand matrix must have exactly two dimensions.
    const auto lhs = matMulOp.getInput();
    const auto lhsType = lhs.getType().dyn_cast<vpux::NDTypeInterface>();
    if (lhsType.getRank() != 2) {
        return {};
    }
    // Right-hand matrix must have exactly two dimensions.
    const auto rhs = matMulOp.getWeights();
    const auto rhsType = rhs.getType().dyn_cast<vpux::NDTypeInterface>();
    if (rhsType.getRank() != 2) {
        return {};
    }
    // Right-hand matrix must have IE.Transpose producer
    auto rhsProducer = rhs.getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::TransposeOp>(rhsProducer)) {
        return {};
    }
    auto transpose = mlir::cast<IE::TransposeOp>(rhsProducer);
    // Check that transpose transforms [d0, d1] shape into [d1, d0]
    const auto transposeInShape = getShape(transpose.getInput());
    const auto transposeOutShape = getShape(transpose.getOutput());
    const auto expectedTransposeOutShape = Shape{transposeInShape[Dim(1)], transposeInShape[Dim(0)]};
    if (expectedTransposeOutShape != transposeOutShape) {
        return {};
    }
    // IE.Transpose must have IE.AffineReshape producer
    auto transposeProducer = transpose.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::AffineReshapeOp>(transposeProducer)) {
        return {};
    }
    auto reshape = mlir::cast<IE::AffineReshapeOp>(transposeProducer);
    // Check that reshape collapses [d0, d1, d2] shape into [d0 * d1, d2]
    const auto reshapeInputDims = getShape(reshape.getInput());
    const Shape expectedOutputShape = {reshapeInputDims[Dim(0)] * reshapeInputDims[Dim(1)], reshapeInputDims[Dim(2)]};
    const auto reshapeOutputDims = getShape(reshape.getOutput());
    if (reshapeOutputDims != expectedOutputShape) {
        return {};
    }
    // IE.Reshape must have IE.Concat producer
    auto reshapeProducer = reshape.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::ConcatOp>(reshapeProducer)) {
        return {};
    }
    // The concat must concatenate [1xHxW, ..., 1xHxW] inputs into CxHxW shape.
    auto concat = mlir::cast<IE::ConcatOp>(reshapeProducer);
    if (!checkConcat(concat)) {
        return {};
    }
    return concat.getInputs();
}

SmallVector<mlir::Value> UnrollFullyConnected::splitLeftInput(const mlir::Value lhs, const int64_t groups,
                                                              mlir::PatternRewriter& rewriter) const {
    const auto lhsShape = getShape(lhs);
    const auto blockSize = lhsShape[Dim(1)] / groups;
    const SmallVector<int64_t> staticSizes = {lhsShape[Dim(0)], blockSize};
    const auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), staticSizes);
    SmallVector<mlir::Value> inputChunks;
    for (const auto& idx : irange(groups)) {
        const auto loc = appendLoc(lhs.getLoc(), "_slice_{0}", idx);
        const SmallVector<int64_t> offsets = {0, idx * blockSize};
        const auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), offsets);
        auto slice = rewriter.create<IE::SliceOp>(loc, lhs, offsetsAttr, staticSizesAttr);
        inputChunks.push_back(slice.getResult());
    }
    return inputChunks;
}

SmallVector<mlir::Value> UnrollFullyConnected::buildMatMuls(const mlir::Location loc, const mlir::ValueRange lhsInputs,
                                                            const mlir::ValueRange rhsInputs,
                                                            mlir::PatternRewriter& rewriter) const {
    VPUX_THROW_UNLESS(lhsInputs.size() == rhsInputs.size(),
                      "The number of left-hand matrices does not match the number of right-hand matrices");
    SmallVector<mlir::Value> matMuls;
    for (const auto& idx : irange(lhsInputs.size())) {
        const auto transposeLoc = appendLoc(loc, "_transpose_{0}", idx);
        SmallVector<unsigned> transPerm = {1, 0};
        const auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transPerm, rewriter.getContext()));
        auto transpose = rewriter.create<IE::TransposeOp>(transposeLoc, rhsInputs[idx],
                                                          /*order=*/nullptr, orderAttr);
        const auto matMulLoc = appendLoc(loc, "_slice_{0}", idx);
        auto newMatMul = rewriter.create<IE::FullyConnectedOp>(matMulLoc, lhsInputs[idx], transpose.getOutput(),
                                                               /*bias=*/nullptr);
        matMuls.push_back(newMatMul.getOutput());
    }
    return matMuls;
}

SmallVector<mlir::Value> UnrollFullyConnected::accumulateMatMuls(const mlir::ValueRange matMuls,
                                                                 mlir::PatternRewriter& rewriter) const {
    const auto numGroups = matMuls.size();
    VPUX_THROW_UNLESS(numGroups >= 2, "The group must contain at least two IE.MatMul operations, got {0}", numGroups);
    SmallVector<mlir::Value> addOps;
    // Add up the first two matrix multiplications.
    // Next iterations will add each MatMul to the previous result.
    const auto addLoc = appendLoc(matMuls[0].getLoc(), "_add");
    auto addMatMuls = rewriter.create<IE::AccumulateOp>(addLoc, matMuls[0], matMuls[1],
                                                        /*lhsScale=*/nullptr,
                                                        /*rhsScale=*/nullptr);
    addOps.push_back(addMatMuls.getOutput());
    for (const auto& idx : irange(numGroups - 2)) {
        // idx + 2 because the first two MatMul operations have already been summed up.
        const auto& matMul = matMuls[idx + 2];
        const auto loc = appendLoc(matMul.getLoc(), "_add");
        auto newAddOp = rewriter.create<IE::AccumulateOp>(loc, addOps.back(), matMuls[idx + 2],
                                                          /*lhsScale=*/nullptr,
                                                          /*rhsScale=*/nullptr);
        addOps.push_back(newAddOp.getOutput());
    }
    return addOps;
}

SmallVector<mlir::Value> UnrollFullyConnected::reshapeTo2d(mlir::ValueRange values,
                                                           mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> reshapedValues;
    const auto to2d = [&rewriter](const mlir::Value val) -> mlir::Value {
        const ShapeRef sliceShape = getShape(val);
        const SmallVector<int64_t> target2dShape = {sliceShape[Dim(1)], sliceShape[Dim(2)]};
        const auto target2dShapeAttr = getIntArrayAttr(rewriter.getContext(), target2dShape);
        auto reshape = rewriter.create<IE::ReshapeOp>(val.getLoc(), val, nullptr, false, target2dShapeAttr);
        return reshape.getOutput();
    };
    std::transform(values.begin(), values.end(), std::back_inserter(reshapedValues), to2d);
    return reshapedValues;
}

mlir::LogicalResult UnrollFullyConnected::matchAndRewrite(IE::FullyConnectedOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto matMulInputs = findMatMulInputs(origOp);
    if (matMulInputs.empty()) {
        return matchFailed(rewriter, origOp, "IE.MatMul at {0} is not supported", origOp->getLoc());
    }
    const auto rhsChunks = reshapeTo2d(matMulInputs, rewriter);
    const auto numChunks = checked_cast<int64_t>(rhsChunks.size());
    // Split left input into the number of chunks:
    const auto lhsChunks = splitLeftInput(origOp.getInput(), numChunks, rewriter);
    // Multiply lhs by rhs in pairs
    const auto matMuls = buildMatMuls(origOp.getLoc(), lhsChunks, rhsChunks, rewriter);
    // Sum up MatMul results
    const auto addOps = accumulateMatMuls(matMuls, rewriter);
    VPUX_THROW_WHEN(addOps.empty(), "The group must contain at least one IE.Accumulate operation, got 0.");
    // The last IE.Accumulate operation in the list will contain the total sum.
    rewriter.replaceOp(origOp, addOps.back());
    return mlir::success();
}

class UnrollFullyConnectedPass final : public IE::UnrollFullyConnectedBase<UnrollFullyConnectedPass> {
public:
    explicit UnrollFullyConnectedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollFullyConnectedPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnrollFullyConnected>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollFullyConnectedPass(Logger log) {
    return std::make_unique<UnrollFullyConnectedPass>(log);
}
