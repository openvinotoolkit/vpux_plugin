//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/conversion.hpp"

namespace vpux {
namespace vpuipdpu2npureg40xx {

class DPUVariantRewriter final : public mlir::OpRewritePattern<VPUIPDPU::DPUVariantOp> {
public:
    DPUVariantRewriter(mlir::MLIRContext* ctx, Logger log, VPU::DPUDryRunMode dryRunMode);

public:
    mlir::LogicalResult matchAndRewrite(VPUIPDPU::DPUVariantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::DPUDryRunMode _dryRunMode;

    mlir::LogicalResult verifyDPUVariant(VPUIPDPU::DPUVariantOp op) const;

    void fillDPUConfigs(
            mlir::Region& DPURegion,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;

    void fillBarrierCfg(
            VPUIPDPU::DPUVariantOp op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
    void fillProfilingCfg(
            VPUIPDPU::DPUVariantOp origOp,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
    void fillStubCfg(std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
};

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
