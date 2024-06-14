//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::VPU::arch30xx {

/*
   Class for getting WrapVPUOpsInNCEClusterTilingStrategy patterns for NPU30XX
*/
class WrapVPUOpsInNCEClusterTilingStrategy : public IGreedilyPassStrategy {
public:
    WrapVPUOpsInNCEClusterTilingStrategy(bool enableExplicitDistributedTensorAttr)
            : _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr) {
    }
    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
};

}  // namespace vpux::VPU::arch30xx
