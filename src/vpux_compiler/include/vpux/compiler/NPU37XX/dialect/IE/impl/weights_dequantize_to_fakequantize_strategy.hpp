//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::IE::arch37xx {

/*
   Class for getting WeightsDequantizeToFakeQuantizeStrategy patterns for VPU37XX
*/
class WeightsDequantizeToFakeQuantizeStrategy : public IGreedilyPassStrategy {
public:
    WeightsDequantizeToFakeQuantizeStrategy(bool enableWDBlockArgumentInput) noexcept;

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;

private:
    const bool _enableWDBlockArgumentInput;
};

}  // namespace vpux::IE::arch37xx
