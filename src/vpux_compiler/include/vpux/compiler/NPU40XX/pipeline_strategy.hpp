//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/pipeline_strategy.hpp"
#include "vpux/compiler/utils/options.hpp"

namespace vpux {

//
// PipelineStrategy40XX
//

class PipelineStrategy40XX final : public IPipelineStrategy {
public:
    void buildPipeline(mlir::PassManager& pm, const intel_npu::Config& config, mlir::TimingScope& rootTiming,
                       Logger log) override;
    void buildELFPipeline(mlir::PassManager& pm, const intel_npu::Config& config, mlir::TimingScope& rootTiming,
                          Logger log, bool useWlm) override;
};

}  // namespace vpux
