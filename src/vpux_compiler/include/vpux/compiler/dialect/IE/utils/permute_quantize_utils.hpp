//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace IE {

bool isLegalReorderAddPattern(IE::ReorderOp origOp);
bool isLegalReorderAvgPoolPattern(IE::ReorderOp origOp);
bool isBeneficialConvertToPermuteQuantize(ShapeRef shape);
std::optional<SmallVector<int64_t>> getAdjustHW(int64_t alignment, int64_t width, int64_t height);
bool isODUPermuteEffectiveForShape(const ShapeRef shape, const int64_t alignment);
bool isShapeCompatibleWithODUPermute(const ShapeRef shape, const int64_t alignment);

}  // namespace IE
}  // namespace vpux
