//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MaxPoolOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MaxPoolOpAdaptor maxPool(operands, attrs, prop);
    if (mlir::failed(maxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool.getStrides());
    const auto roundingType = maxPool.getRoundingType();

    const auto inType = maxPool.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();

    const auto shapeI64 = inferMaxPoolOutputShape(inShape, windowStrides, dataPaddingBelow, dataPaddingAbove,
                                                  windowShape, roundingType);

    const auto outType = inType.changeShape(Shape(shapeI64));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::MaxPoolOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto origInputShape = getShape(getInput());
    const auto origPadding = PadInfo(getPadsBegin(), getPadsEnd());

    return backInferPoolTile(outputTile, origInputShape, getKernelSize(), getStrides(), origPadding);
}

void vpux::VPU::MaxPoolOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    IE::adjustPaddings(this, inputTiling);
}

mlir::FailureOr<OutputTiling> vpux::VPU::MaxPoolOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
