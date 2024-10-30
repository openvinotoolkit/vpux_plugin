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

constexpr StringRef AUTO_PADDING_ODU = "VPU.AutoPaddingODU";
constexpr StringRef AUTO_PADDING_IDU = "VPU.AutoPaddingIDU";

// Hardware limitation
constexpr int64_t WIDTH16_CHANNEL_LIMIT = 10;
constexpr int64_t FP16_WIDTH = 16;

bool hasAutoPaddingODU(mlir::ModuleOp module);
bool hasAutoPaddingIDU(mlir::ModuleOp module);
bool inputCompatibleWithAutoPad(vpux::NDTypeInterface type);
bool hasOnlyOutPadding(mlir::ModuleOp module);
bool hasOnlyInPadding(mlir::ModuleOp module);

}  // namespace VPU
}  // namespace vpux
