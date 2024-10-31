//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/compiler/utils/passes.hpp>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef FP16_COMPRESSED_CONV = "VPU.FP16CompressedConv";

bool hasFP16CompressedConv(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
