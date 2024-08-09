//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

IE::PipelineOptionsOp VPU::getPipelineOptionsOp(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp) {
    auto pipelineOptionsOp = moduleOp.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    const auto hasPipelineOptions = pipelineOptionsOp != nullptr;
    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());

    if (!hasPipelineOptions) {
        pipelineOptionsOp =
                optionsBuilder.create<IE::PipelineOptionsOp>(mlir::UnknownLoc::get(&ctx), VPU::PIPELINE_OPTIONS);
        pipelineOptionsOp.getOptions().emplaceBlock();
    }

    return pipelineOptionsOp;
}
