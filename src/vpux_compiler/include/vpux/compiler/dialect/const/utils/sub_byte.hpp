//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/format.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/utils/types.hpp"

#include <openvino/core/type/element_type.hpp>

#include <mlir/IR/MLIRContext.h>

namespace vpux {
namespace Const {

bool isSubByte(const size_t bitWidth);

SmallVector<char> getConstBuffer(const char* sourceData, const size_t bitWidth, const int64_t numElems);

}  // namespace Const
}  // namespace vpux
