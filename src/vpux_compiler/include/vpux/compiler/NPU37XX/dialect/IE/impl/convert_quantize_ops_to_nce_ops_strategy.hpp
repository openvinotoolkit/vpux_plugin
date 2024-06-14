//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/convert_quantize_ops_to_nce_ops_strategy.hpp"

namespace vpux::IE::arch37xx {
class ConvertQuantizeOpsToNceOpsStrategy final : public vpux::IE::IConvertQuantizeOpsToNceOpsStrategy {
public:
    void prepareAvgPool(mlir::ConversionTarget& toAvgPoolTarget, mlir::RewritePatternSet& toAvgPoolPatterns,
                        mlir::MLIRContext& ctx, Logger& log) final override;
    void prepareEltwise(mlir::ConversionTarget& toEltwiseTarget, mlir::RewritePatternSet& toEltwisePatterns,
                        mlir::MLIRContext& ctx, Logger& log) final override;
    void prepareQuantToDw(mlir::ConversionTarget& quantToDwTarget, mlir::RewritePatternSet& quantToDwPatterns,
                          mlir::MLIRContext& ctx, Logger& log) final override;

private:
    const bool _canUseCMajor = false;
};

}  // namespace vpux::IE::arch37xx
