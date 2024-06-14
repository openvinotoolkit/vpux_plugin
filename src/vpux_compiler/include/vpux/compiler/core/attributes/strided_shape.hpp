//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"

namespace vpux {

//
// StridedShape
//

struct StridedShape {
    StridedShape(ShapeRef shape, StridesRef strides): shape(shape.raw()), strides(strides.raw()) {
    }
    StridedShape() = default;

    Shape shape;
    Strides strides;
};

}  // namespace vpux
