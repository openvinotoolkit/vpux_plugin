//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/compressed_convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

bool VPU::hasFP16CompressedConv(mlir::Operation* op) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch EnableFP16CompressedConv");

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(FP16_COMPRESSED_CONV);
    VPUX_THROW_WHEN(attrValue == nullptr, "Failed to find FP16 Compressed Conv IE.OptionOp attribute");
    return static_cast<bool>(attrValue.getOptionValue());
}
