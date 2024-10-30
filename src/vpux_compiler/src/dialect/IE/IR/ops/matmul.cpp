//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// MatMul operation takes two tensors and performs usual matrix-matrix multiplication, matrix-vector multiplication or
// vector-matrix multiplication depending on argument shapes. Input tensors can have any rank >= 1. Two right-most axes
// in each tensor are interpreted as matrix rows and columns dimensions while all left-most axes (if present) are
// interpreted as multi-dimensional batch: [BATCH_DIM_1, BATCH_DIM_2,..., BATCH_DIM_K, ROW_INDEX_DIM, COL_INDEX_DIM].
// The operation supports usual broadcast semantics for batch dimensions. It enables multiplication of batch of pairs of
// matrices in a single shot. Before matrix multiplication, there is an implicit shape alignment for input arguments. It
// consists of the following steps:
// 1. Applying transpositions specified by optional transpose_a and transpose_b attributes. Only the two right-most
// dimensions are transposed, other dimensions remain the same. Transpose attributes are ignored for 1D tensors.
// 2. One-dimensional tensors unsqueezing is applied for each input independently. The axes inserted in this step are
// not included in the output shape.
//    - If rank of the first input is equal to 1, it is always unsqueezed to 2D tensor row vector (regardless of
//    transpose_a) by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape. For example [S] will be
//    reshaped to [1, S].
//    - If rank of the second input is equal to 1, it is always unsqueezed to 2D tensor column vector (regardless of
//    transpose_b) by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape. For example [S] will be
//    reshaped to [S, 1].
//    - Temporary axes inserted in this step are removed from the final output shape after multiplying. After
//    vector-matrix multiplication, the temporary axis inserted at ROW_INDEX_DIM is removed. After matrix-vector
//    multiplication, the temporary axis inserted at COL_INDEX_DIM is removed. Output shape of two 1D tensors
//    multiplication [S] x [S] is squeezed to scalar.
// Example)
//   [M, K] * [K, N] => [M, N]
//   [K]    * [K, N] => [1, K] * [K, N] => [1, N] => [N]
//   [M, K] * [K]    => [M, K] * [K, 1] => [M, 1] => [M]
//   [M]    * [M]    => [1, M] * [M, 1] => [1, 1] => [1]
//
// 3. If ranks of input arguments are different after steps 1 and 2, the tensor with a smaller rank is unsqueezed from
// the left side of the shape by necessary number of axes to make both shapes of the same rank.
// 4. Usual rules of the broadcasting are applied for batch dimensions.
//

#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

mlir::LogicalResult vpux::IE::MatMulOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MatMulOpAdaptor matMul(operands, attrs, prop);
    if (mlir::failed(matMul.verify(loc))) {
        return mlir::failure();
    }

    const auto inType1 = matMul.getInput1().getType().cast<mlir::ShapedType>();
    const auto inType2 = matMul.getInput2().getType().cast<mlir::ShapedType>();
    const auto inShape1 = inType1.getShape();
    const auto inShape2 = inType2.getShape();

    const auto inRank1 = inShape1.size();
    const auto inRank2 = inShape2.size();
    const auto transA = matMul.getTransposeA();
    const auto transB = matMul.getTransposeB();

    // Rightmost two axes are row & col. Remaining left axes are batch
    constexpr int kRowColIdxRange = 2;

    SmallVector<int64_t> outShape;
    outShape.reserve(std::max(inRank1, inRank2));

    // Temporally transformed shapes
    auto inShape1Trans = to_small_vector(inShape1);
    auto inShape2Trans = to_small_vector(inShape2);
    std::reverse(inShape1Trans.begin(), inShape1Trans.end());
    std::reverse(inShape2Trans.begin(), inShape2Trans.end());

    // Apply transpose only when rank >= 2
    if (transA && (inRank1 > 1)) {
        std::swap(inShape1Trans[0], inShape1Trans[1]);
    }
    if (transB && (inRank2 > 1)) {
        std::swap(inShape2Trans[0], inShape2Trans[1]);
    }

    // Only use the dim when it is Mat
    if (inRank2 >= kRowColIdxRange) {
        outShape.push_back(inShape2Trans[0]);
    }
    if (inRank1 >= kRowColIdxRange) {
        outShape.push_back(inShape1Trans[1]);
    }

    // Process batch axes
    uint32_t idx1 = kRowColIdxRange;
    uint32_t idx2 = kRowColIdxRange;

    while (idx1 < inRank1 || idx2 < inRank2) {
        if (idx1 < inRank1 && idx2 < inRank2) {
            outShape.push_back(std::max(inShape1Trans[idx1], inShape2Trans[idx2]));
            ++idx1;
            ++idx2;
        } else if (idx2 >= inRank2) {
            outShape.push_back(inShape1Trans[idx1]);
            ++idx1;
        } else if (idx1 >= inRank1) {
            outShape.push_back(inShape2Trans[idx2]);
            ++idx2;
        }
    }
    std::reverse(std::begin(outShape), std::end(outShape));

    inferredReturnShapes.emplace_back(outShape, inType1.getElementType());
    return mlir::success();
}

//
// UseFullyConnected
//

namespace {

class UseFullyConnected final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    using mlir::OpRewritePattern<IE::MatMulOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult UseFullyConnected::matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const {
    auto inShape = getShape(matmulOp.getInput1());
    auto weightsShape = getShape(matmulOp.getInput2());

    auto transIn = matmulOp.getTransposeA();
    auto transWeights = matmulOp.getTransposeB();

    static const auto IC = Dim(1);

    if (inShape.size() != 2 || weightsShape.size() != 2) {
        return mlir::failure();
    }

    if (transIn || (!transWeights) || inShape[IC] != weightsShape[IC]) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::FullyConnectedOp>(matmulOp, matmulOp.getType(), matmulOp.getInput1(),
                                                      matmulOp.getInput2(), nullptr);

    return mlir::success();
}

}  // namespace

//
// TransposeInputs
//

namespace {

class TransposeInputs final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    using mlir::OpRewritePattern<IE::MatMulOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult TransposeInputs::matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const {
    // E-122051:
    // MatMul can be converted to IE::FullyConnectedOp or to a specialized software implementation VPU::MatMulOp.
    // Both of these paths has separate restrictions on the input format. IE::FullyConnected requires transposeA ==
    // false and transposeB == true. VPU::MatMulOp requires transposeA = false and transposeB == false. I think the
    // canonical form of IE::MatMul should have transposeA==false and transposeB==false, and we should legalize it for
    // IE::FullyConnected in a new `ConvertMatMulToFullyConnected` pass.
    if (VPU::MatMulOp::isSupported(matmulOp)) {
        return mlir::failure();
    }

    if (IE::isMatmulWithRHSTransposition(matmulOp)) {
        return mlir::failure();
    }

    auto ctx = matmulOp.getContext();
    auto transposedOrderAttr = [&](int64_t inputRank) -> mlir::AffineMapAttr {
        SmallVector<unsigned> perm(inputRank, 0);
        std::iota(perm.begin(), perm.end(), 0);
        if (inputRank < 2) {
            return nullptr;
        }
        std::iter_swap(perm.end() - 1, perm.end() - 2);
        return mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, ctx));
    };

    auto getInput = [&](auto isTranspose, auto input, StringRef locSuffix) -> mlir::Value {
        if (isTranspose) {
            return input;
        }
        const auto inputShape = getShape(input);
        const auto inputRank = inputShape.size();

        if (transposedOrderAttr(inputRank) != nullptr) {
            auto orderAttr = transposedOrderAttr(inputRank);
            auto transpose = rewriter.create<IE::TransposeOp>(appendLoc(matmulOp->getLoc(), locSuffix), input, nullptr,
                                                              orderAttr);
            return transpose.getOutput();
        }

        return nullptr;
    };

    auto input1 = getInput(!matmulOp.getTransposeA(), matmulOp.getInput1(), "input_1");
    if (input1 == nullptr) {
        return mlir::failure();
    }
    auto input2 = getInput(matmulOp.getTransposeB(), matmulOp.getInput2(), "input_2");
    if (input2 == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::MatMulOp>(matmulOp, input1, input2, /*transpose_a=*/false,
                                              /*transpose_b=*/true);

    return mlir::success();
}

}  // namespace

void vpux::IE::MatMulOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<TransposeInputs>(context);
    patterns.add<UseFullyConnected>(context);
}
