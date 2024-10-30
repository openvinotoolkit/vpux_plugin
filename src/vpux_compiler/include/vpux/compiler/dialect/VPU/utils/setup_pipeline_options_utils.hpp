//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/compiler/utils/passes.hpp>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef PIPELINE_OPTIONS = "Options";

vpux::IE::PipelineOptionsOp getPipelineOptionsOp(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp);

}  // namespace VPU
}  // namespace vpux
