//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::RMSOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::RMSOpAdaptor rms(operands, attrs, prop);
    if (mlir::failed(rms.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = rms.getInput().getType().cast<mlir::ShapedType>();
    const auto gammaType = rms.getGamma().getType().cast<mlir::ShapedType>();
    const auto inputRank = inType.getRank();

    if (inputRank < 3) {
        return errorAt(loc, "Input tensor rank should be 3 or greater. Got {0}D tensor.", inputRank);
    }

    const auto inputWidth = inType.getDimSize(inputRank - 1);
    const auto gammaWidth = gammaType.getDimSize(0);

    if (inputWidth != gammaWidth) {
        return errorAt(loc, "Input width should be the same as gamma. Got input width = {0} and gamma width = {1}",
                       inputWidth, gammaWidth);
    }

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());
    return mlir::success();
}
