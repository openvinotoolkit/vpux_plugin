//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::YuvToRgbOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::YuvToRgbOpAdaptor colorConv(operands, attrs);
    if (mlir::failed(colorConv.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = colorConv.getInput1().getType().cast<mlir::ShapedType>();
    const auto shape = inType.getShape();
    if (shape[3] != 1) {
        return errorAt(loc, "Incorrect input shape format: '{0}'", shape);
    }

    SmallVector<int64_t> outShape{shape[0], shape[1], shape[2], 3};

    if (colorConv.getInput2() == nullptr) {
        VPUX_THROW_UNLESS(colorConv.getInput3() == nullptr, "1xPlane config error");
        VPUX_THROW_UNLESS(((outShape[1] * 2) % 3) == 0, "Invalid height");
        outShape[1] = outShape[1] * 2 / 3;
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// ConvertToMultiInputs
//

namespace {

class ConvertToMultiInputs final : public mlir::OpRewritePattern<IE::YuvToRgbOp> {
public:
    using mlir::OpRewritePattern<IE::YuvToRgbOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::YuvToRgbOp yuvToRgbOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToMultiInputs::matchAndRewrite(IE::YuvToRgbOp yuvToRgbOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (yuvToRgbOp.getInput2() == nullptr) {
        const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(yuvToRgbOp).to<Byte>().count();

        const auto outputByteSize =
                yuvToRgbOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElemTypeSize().to<Byte>().count();
        const auto outputSizeBytes =
                yuvToRgbOp.getOutput().getType().cast<NDTypeInterface>().getShape().totalSize() * outputByteSize;

        const auto inputByteSize =
                yuvToRgbOp.getInput1().getType().cast<vpux::NDTypeInterface>().getElemTypeSize().to<Byte>().count();
        const auto inputSizeBytes =
                yuvToRgbOp.getInput1().getType().cast<NDTypeInterface>().getShape().totalSize() * inputByteSize;
        auto requiredCMX = outputSizeBytes + inputSizeBytes;
        if (requiredCMX < cmxAvailableBytes) {
            return mlir::success();
        }

        auto inputShape = yuvToRgbOp.getInput1().getType().cast<NDTypeInterface>().getShape();
        const auto inShapeType = yuvToRgbOp.getInput1().getType().cast<mlir::ShapedType>().getShape();
        const auto sliceOpLoc = yuvToRgbOp.getLoc();
        auto* ctx = rewriter.getContext();
        enum { N = 0, H = 1, W = 2, C = 3 };

        if (yuvToRgbOp.getInFmt() == IE::ColorFmt::NV12) {
            auto input1_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto input2_offsets = SmallVector<int64_t>(inputShape.size(), 0);

            input2_offsets[H] = inShapeType[H] / 3 * 2;

            SmallVector<int64_t> input1_sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> input2_sizes(inputShape.begin(), inputShape.end());

            input1_sizes[H] = inShapeType[H] / 3 * 2;
            input2_sizes[H] = inShapeType[H] / 3;

            auto input1_slice = rewriter.create<IE::SliceOp>(sliceOpLoc, yuvToRgbOp.getInput1(),
                                                             getIntArrayAttr(ctx, input1_offsets),
                                                             getIntArrayAttr(ctx, input1_sizes));
            auto input2_slice = rewriter.create<IE::SliceOp>(sliceOpLoc, yuvToRgbOp.getInput1(),
                                                             getIntArrayAttr(ctx, input2_offsets),
                                                             getIntArrayAttr(ctx, input2_sizes));

            input2_sizes[W] = input2_sizes[W] / 2;
            input2_sizes[C] = 2;
            auto shapeEndAttr = getIntArrayAttr(ctx, input2_sizes);
            auto input2_slice_reshape =
                    rewriter.create<IE::ReshapeOp>(sliceOpLoc, input2_slice.getResult(), nullptr, false, shapeEndAttr);

            rewriter.replaceOpWithNewOp<IE::YuvToRgbOp>(yuvToRgbOp, input1_slice.getResult(), input2_slice_reshape,
                                                        nullptr, yuvToRgbOp.getInFmt(), yuvToRgbOp.getOutFmt());
            return mlir::success();

        } else {
            auto inputY_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto inputUV_offsets = SmallVector<int64_t>(inputShape.size(), 0);

            inputUV_offsets[H] = inShapeType[H] / 3 * 2;

            SmallVector<int64_t> inputY_sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> inputUV_sizes(inputShape.begin(), inputShape.end());

            inputY_sizes[H] = inShapeType[H] / 3 * 2;
            inputY_sizes[W] = inShapeType[W];
            inputUV_sizes[H] = inShapeType[H] / 3;
            inputUV_sizes[W] = inShapeType[W];

            auto inputY_slice = rewriter.create<IE::SliceOp>(sliceOpLoc, yuvToRgbOp.getInput1(),
                                                             getIntArrayAttr(ctx, inputY_offsets),
                                                             getIntArrayAttr(ctx, inputY_sizes));
            auto inputUV_slice = rewriter.create<IE::SliceOp>(sliceOpLoc, yuvToRgbOp.getInput1(),
                                                              getIntArrayAttr(ctx, inputUV_offsets),
                                                              getIntArrayAttr(ctx, inputUV_sizes));

            inputUV_sizes[H] = inShapeType[H] / 3 * 2;
            inputUV_sizes[W] = inShapeType[W] / 2;

            auto shape2EndAttr = getIntArrayAttr(ctx, inputUV_sizes);
            auto inputUV_slice_reshape = rewriter.create<IE::ReshapeOp>(sliceOpLoc, inputUV_slice.getResult(), nullptr,
                                                                        false, shape2EndAttr);

            SmallVector<int64_t> inputU_sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> inputV_sizes(inputShape.begin(), inputShape.end());

            inputU_sizes[H] = inputUV_sizes[H] / 2;
            inputU_sizes[W] = inputUV_sizes[W];

            inputV_sizes[H] = inputUV_sizes[H] / 2;
            inputV_sizes[W] = inputUV_sizes[W];

            auto inputU_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto inputV_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            inputV_offsets[H] = inputUV_sizes[H] / 2;

            auto inputU_slice = rewriter.create<IE::SliceOp>(sliceOpLoc, inputUV_slice_reshape,
                                                             getIntArrayAttr(ctx, inputU_offsets),
                                                             getIntArrayAttr(ctx, inputU_sizes));
            auto inputV_slice = rewriter.create<IE::SliceOp>(sliceOpLoc, inputUV_slice_reshape,
                                                             getIntArrayAttr(ctx, inputV_offsets),
                                                             getIntArrayAttr(ctx, inputV_sizes));

            rewriter.replaceOpWithNewOp<IE::YuvToRgbOp>(yuvToRgbOp, inputY_slice.getResult(), inputU_slice.getResult(),
                                                        inputV_slice.getResult(), yuvToRgbOp.getInFmt(),
                                                        yuvToRgbOp.getOutFmt());
            return mlir::success();
        }
    }
    return mlir::success();
}

}  // namespace

void vpux::IE::YuvToRgbOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertToMultiInputs>(context);
}
