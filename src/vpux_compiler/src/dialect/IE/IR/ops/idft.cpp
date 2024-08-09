//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/fft_ops_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::IDFTOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::IDFTOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }
    const auto inType = op.getInput().getType().cast<mlir::ShapedType>();
    auto outShape = to_small_vector(inType.getShape());
    auto params = fftExtractParams(loc, op);
    if (mlir::failed(params)) {
        return mlir::failure();
    }
    auto axes = params.value().axes;
    auto signalSize = params.value().signalSize;
    for (size_t i = 0; i < axes.size(); ++i) {
        const int64_t currentAxis = axes[i];
        if (signalSize[i] != -1) {
            outShape[currentAxis] = signalSize[i];
        }
    }
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    return mlir::success();
}

//
// getCanonicalizationPatterns
//

void vpux::IE::IDFTOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<IE::FftConvertConstToAttrAndNormalize<IE::IDFTOp>>(context);
}
