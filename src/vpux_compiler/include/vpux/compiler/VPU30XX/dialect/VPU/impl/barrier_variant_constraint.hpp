//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/barrier_variant_constraint.hpp"

namespace vpux::VPU::arch30xx {

// Given firmwareVariantCount are constant we need some multiplier(ratio) to get the right barrierMaxVariantSum and
// barrierMaxVariantCount.
// TODO: E#78647 refactor to use api/vpu_cmx_info_{arch}.h
constexpr double barrierMaxVariantCountRatio = 1;
constexpr double barrierMaxVariantSumRatio = 1;
constexpr size_t firmwareVariantCount = 512;

struct PerBarrierVariantConstraint final {
    size_t getPerBarrierMaxVariantSum() const;
    size_t getPerBarrierMaxVariantCount() const;
};
}  // namespace vpux::VPU::arch30xx
