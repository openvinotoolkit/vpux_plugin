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
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReduceSumOpAdaptor reduceSum(operands, attrs);
    if (mlir::failed(reduceSum.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceSum.input();
    const auto keepDims = reduceSum.keep_dims() != nullptr;
    auto axes = IE::constInputToData(loc, reduceSum.axes()).getValue();

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axes, inferredReturnShapes);
}

std::shared_ptr<ngraph::Node> vpux::IE::ReduceSumOp::toNgraph(ngraph::OutputVector &outputs)
{
    return std::make_shared<opset_latest::ReduceSum>(outputs.at(0), outputs.at(1), keep_dims());
}
