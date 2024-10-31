//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>
#include <cstddef>

using namespace vpux;

bool VPU::hasMaxKernelSize(mlir::Operation* op) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp != nullptr) {
        auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(VPU::MAX_KERNEL_SIZE);
        if (attrValue != nullptr) {
            return true;
        }
    }
    return false;
}

int64_t VPU::getMaxKernelSize(mlir::Operation* op) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch maxKernelSize");

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(VPU::MAX_KERNEL_SIZE);
    VPUX_THROW_WHEN(attrValue == nullptr, "Failed to find IE.OptionOp attribute", VPU::MAX_KERNEL_SIZE);

    return static_cast<int64_t>(attrValue.getOptionValue());
}
