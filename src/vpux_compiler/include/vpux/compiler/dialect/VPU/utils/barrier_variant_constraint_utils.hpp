//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/compiler/utils/passes.hpp>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/barrier_variant_constraint.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef BARR_MAX_VARIANT_SUM = "VPU.BarrierMaxVariantSum";
constexpr StringRef BARR_MAX_VARIANT_COUNT = "VPU.BarrierMaxVariantCount";
constexpr StringRef PIPELINE_OPTIONS = "Options";

size_t getPerBarrierVariantConstraint(mlir::Operation* op, StringRef attrName);

}  // namespace VPU
}  // namespace vpux
