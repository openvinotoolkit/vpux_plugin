//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/validation_util.hpp>

#include <mlir/IR/BlockAndValueMapping.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::MaxPoolOpAdaptor maxPool(operands, attrs);
    if (mlir::failed(maxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool.pads_begin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool.kernel_size());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool.strides());
    const auto roundingType = maxPool.rounding_type().getValue();

    const auto inType = maxPool.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto inShape = maxPool.input().getType().cast<mlir::ShapedType>().getShape();

    const auto outputShape = ngraph::infer_batched_pooling_forward(
            nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
            ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ngraph::Shape(windowShape.begin(), windowShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()), true,
            roundingType == vpux::IE::RoundingType::CEIL);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

InputTiling vpux::IE::MaxPoolOp::backInferTileInfo(const vpux::TileInfo& outputTile) {
    const auto origInputShape = getShape(input());

    return backInferPoolTile(outputTile, origInputShape, kernel_size(), strides(), pads_begin(), pads_end());
}

void vpux::IE::MaxPoolOp::adjustAttrs(const TilingInfo& inputTiling) {
    IE::adjustPaddings(this, inputTiling);
}

std::unique_ptr<ngraph::Node> vpux::IE::MaxPoolOp::toNgraph(ngraph::OutputVector &outputs)
{
    VPUX_THROW_WHEN(post_opAttr() != nullptr, "post_op attribute for '{0}' is not supported", IE::MaxPoolOp::getOperationName());
    const auto strides = parseIntArrayAttr<size_t>(stridesAttr());
    const auto padsBegin = parseIntArrayAttr<size_t>(pads_begin());
    const auto padsEnd = parseIntArrayAttr<size_t>(pads_end());
    const auto kernel = parseIntArrayAttr<size_t>(kernel_size());

    return std::make_unique<opset_latest::MaxPool>(outputs.at(0), ngraph::Strides(strides.begin(), strides.end()),
        ngraph::Shape(padsBegin.begin(), padsBegin.end()), ngraph::Shape(padsEnd.begin(), padsEnd.end()),
        ngraph::Shape(kernel.begin(), kernel.end()), exportRoundingType(rounding_type()));
}
