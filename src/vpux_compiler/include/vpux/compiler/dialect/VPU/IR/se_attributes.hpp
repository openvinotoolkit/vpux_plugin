//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"

#include <cstdint>

namespace vpux {
namespace VPU {

// SEInterpolateAttr
constexpr int64_t SE_INTERPOLATE_FACTOR_H = 0;
constexpr int64_t SE_INTERPOLATE_FACTOR_W = 1;

constexpr int64_t SE_INTERPOLATE_KERNEL_Y = 0;
constexpr int64_t SE_INTERPOLATE_KERNEL_X = 1;

constexpr int64_t SE_INTERPOLATE_STRIDE_Y = 0;
constexpr int64_t SE_INTERPOLATE_STRIDE_X = 1;

// SEUpsamplingAttr
constexpr int64_t SE_UPSAMPLING_FACTOR_H = 0;
constexpr int64_t SE_UPSAMPLING_FACTOR_W = 1;

constexpr int64_t SE_PAD_LEFT = 0;
constexpr int64_t SE_PAD_TOP = 1;
constexpr int64_t SE_PAD_RIGHT = 2;
constexpr int64_t SE_PAD_BOTTOM = 3;

// SERollAttr
constexpr int64_t SE_ROLL_SPATIAL_H = 0;
constexpr int64_t SE_ROLL_SPATIAL_W = 1;

}  // namespace VPU
}  // namespace vpux
