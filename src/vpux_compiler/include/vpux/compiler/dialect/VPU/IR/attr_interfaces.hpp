//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace vpux {

namespace VPU {
struct SETileInfo {
    mlir::ArrayAttr offsets;
    mlir::ArrayAttr sizes;
};

}  // namespace VPU

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPU/attr_interfaces.hpp.inc>
