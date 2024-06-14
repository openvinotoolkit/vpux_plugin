//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/barrier_variant_constraint_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

size_t VPU::getPerBarrierVariantConstraint(mlir::Operation* op, StringRef attrName) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch PerBarrierVariantConstarint");

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(attrName);
    VPUX_THROW_WHEN(attrValue == nullptr, "Failed to find IE.OptionOp attribute", attrName);
    return static_cast<size_t>(attrValue.getOptionValue());
}
