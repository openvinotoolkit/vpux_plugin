//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

bool VPU::isSprLUTEnabled(mlir::Operation* op) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch SprLUTEnabled");

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(VPU::SPRLUT_ENABLED);
    VPUX_THROW_WHEN(attrValue == nullptr, "Failed to find ReduceOpSupported IE.OptionOp attribute");
    return static_cast<bool>(attrValue.getOptionValue());
}
