//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"

namespace vpux {
namespace IE {

DimArr getDiffInOutSizeDims(ShapeRef inShape, ShapeRef outShape) {
    VPUX_THROW_UNLESS(inShape.size() == outShape.size(),
                      "The size of the input '{0}' and output '{1}' tensors does not match", inShape.size(),
                      outShape.size());
    SmallVector<Dim> diffInOutSizeDims;
    const auto ioShapes = zip(inShape, outShape);
    for (const auto& ioShape : ioShapes | indexed) {
        const auto inSize = std::get<0>(ioShape.value());
        const auto outSize = std::get<1>(ioShape.value());
        if (inSize != outSize) {
            diffInOutSizeDims.push_back(Dim(ioShape.index()));
        }
    }
    return diffInOutSizeDims;
}

std::optional<vpux::Dim> getSingleDiffAxis(ShapeRef inShape, ShapeRef outShape) {
    const auto layerAxes = getDiffInOutSizeDims(inShape, outShape);
    return layerAxes.size() == 1 ? layerAxes.front() : std::optional<vpux::Dim>{};
}

}  // namespace IE
}  // namespace vpux
