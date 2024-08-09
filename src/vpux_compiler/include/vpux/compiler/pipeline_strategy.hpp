//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/core/logger.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {
//
// IPipelineStrategy
//

class IPipelineStrategy {
public:
    virtual void buildPipeline(mlir::PassManager& pm, const intel_npu::Config& config, mlir::TimingScope& rootTiming,
                               Logger log) = 0;

    virtual void buildELFPipeline(mlir::PassManager& pm, const intel_npu::Config& config, mlir::TimingScope& rootTiming,
                                  Logger log) = 0;

    virtual ~IPipelineStrategy() = default;
};

}  // namespace vpux
