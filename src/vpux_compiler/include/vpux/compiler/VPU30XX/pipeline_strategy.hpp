//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/pipeline_strategy.hpp"

namespace vpux {

//
// PipelineStrategy30XX
//

class PipelineStrategy30XX final : public IPipelineStrategy {
public:
    void buildPipeline(mlir::PassManager& pm, const Config& config, mlir::TimingScope& rootTiming, Logger log) override;
    void buildELFPipeline(mlir::PassManager& pm, const Config& config, mlir::TimingScope& rootTiming,
                          Logger log) override;
};

}  // namespace vpux
