//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::arch30xx {

class ConvertLayers2VPUStrategy : public IGreedilyPassStrategy {
public:
    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override;
};

}  // namespace vpux::arch30xx
