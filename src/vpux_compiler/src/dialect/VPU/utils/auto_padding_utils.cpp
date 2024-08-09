//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

bool VPU::hasAutoPaddingODU(mlir::Operation* op) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch AutoPaddingODU");

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(AUTO_PADDING_ODU);
    VPUX_THROW_WHEN(attrValue == nullptr, "Failed to find ODU Auto Padding IE.OptionOp attribute");
    return static_cast<bool>(attrValue.getOptionValue());
}
