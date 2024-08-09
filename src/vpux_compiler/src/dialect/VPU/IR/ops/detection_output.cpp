//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DetectionOutputOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputOpAdaptor detectionOutput(operands, attrs, prop);
    if (mlir::failed(detectionOutput.verify(loc))) {
        return mlir::failure();
    }

    const auto boxLogitsType = detectionOutput.getInBoxLogits().getType().cast<vpux::NDTypeInterface>();

    auto origN{0}, origC{1};
    const auto numImages = boxLogitsType.getShape().raw()[origN];
    const auto numLocClasses = detectionOutput.getAttr().getShareLocation().getValue()
                                       ? 1
                                       : detectionOutput.getAttr().getNumClasses().getInt();

    if (numLocClasses <= 0) {
        return errorAt(loc, "Number of classes should be a natural number");
    }

    if (boxLogitsType.getShape().raw()[origC] % (numLocClasses * 4) != 0) {
        return errorAt(loc, "C dimension should be divisible by numLocClasses * 4");
    }

    const auto numPriorBoxes = boxLogitsType.getShape().raw()[origC] / (numLocClasses * 4);
    const auto keepTopK = detectionOutput.getAttr().getKeepTopK()[0].cast<mlir::IntegerAttr>().getInt();
    const auto topK = detectionOutput.getAttr().getTopK().getInt();
    const auto numClasses = detectionOutput.getAttr().getNumClasses().getInt();

    SmallVector<int64_t> outputShape{1, 1};
    if (keepTopK > 0) {
        outputShape.push_back(numImages * keepTopK);
    } else if (topK > 0) {
        outputShape.push_back(numImages * topK * numClasses);
    } else {
        outputShape.push_back(numImages * numPriorBoxes * numClasses);
    }
    outputShape.push_back(7);

    const auto outType = boxLogitsType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
