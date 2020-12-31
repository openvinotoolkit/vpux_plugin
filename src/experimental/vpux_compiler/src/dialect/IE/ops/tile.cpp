//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::TileOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TileOpAdaptor tile(operands, attrs);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = tile.input().getType().cast<mlir::ShapedType>();

    auto inRepeats = tile.repeats().getDefiningOp<mlir::ConstantOp>();
    if (inRepeats == nullptr) {
        return mlir::failure();
    }

    const auto denseElementArray = inRepeats.value().dyn_cast<mlir::DenseElementsAttr>();
    if (denseElementArray == nullptr) {
        return mlir::failure();
    }

    const auto elementsRange = denseElementArray.getValues<int64_t>();

    auto outShape = to_small_vector(inType.getShape());
    auto elementsIter = elementsRange.begin();
    for (size_t i = 0; i < outShape.size(); ++i) {
        if (elementsIter == elementsRange.end()) {
            return mlir::failure();
        }

        outShape[i] *= *elementsIter++;
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    return mlir::success();
}
