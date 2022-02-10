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
#include "vpux/utils/core/range.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::AvgPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::AvgPoolOpAdaptor avgPool(operands, attrs);
    if (mlir::failed(avgPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(avgPool.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(avgPool.pads_begin());
    const auto windowShape = parseIntArrayAttr<int64_t>(avgPool.kernel_size());
    const auto windowStrides = parseIntArrayAttr<int64_t>(avgPool.strides());
    const auto roundingType = avgPool.rounding_type().getValue();

    const auto inType = avgPool.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto inShape = avgPool.input().getType().cast<mlir::ShapedType>().getShape();

    const auto outputShape = ngraph::infer_batched_pooling_forward(
            nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
            ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ngraph::Shape(windowShape.begin(), windowShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()),
            true, /* It is only used during assertion. True will make it pass */
            roundingType == vpux::IE::RoundingType::CEIL);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

std::unique_ptr<ngraph::Node> vpux::IE::AvgPoolOp::toNgraph(ngraph::OutputVector &outputs)
{
    const auto strides = parseIntArrayAttr<size_t>(stridesAttr());
    const auto padsBegin = parseIntArrayAttr<size_t>(pads_begin());
    const auto padsEnd = parseIntArrayAttr<size_t>(pads_end());
    const auto kernel = parseIntArrayAttr<size_t>(kernel_size());
    const auto excludePads = exclude_pads();
    const auto roundingType = exportRoundingType(rounding_type());

    return std::make_unique<opset_latest::AvgPool>(outputs.at(0), ngraph::Strides(strides.begin(),strides.end()),
        ngraph::Shape(padsBegin.begin(), padsBegin.end()), ngraph::Shape(padsEnd.begin(), padsEnd.end()),
        ngraph::Shape(kernel.begin(), kernel.end()), excludePads, roundingType);
}
