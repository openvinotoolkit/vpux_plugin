//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_outstanding_quant_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_outstanding_quant.hpp"

namespace vpux::IE::arch37xx {

//
// FuseOutstandingQuantStrategy
//

void FuseOutstandingQuantStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::AddOp>>(ctx, isMixPrecisionSupported, log);
    patterns.add<vpux::IE::QuantizeWithAvgPool>(ctx, isMixPrecisionSupported, log);
}

}  // namespace vpux::IE::arch37xx
