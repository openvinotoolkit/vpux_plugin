//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/conversion.hpp"

namespace vpux {
namespace vpuipdpu2npureg40xx {

class DPUInvariantRewriter final : public mlir::OpRewritePattern<VPUIPDPU::DPUInvariantOp> {
public:
    DPUInvariantRewriter(mlir::MLIRContext* ctx, Logger log, VPU::DPUDryRunMode dryRunMode);

public:
    mlir::LogicalResult matchAndRewrite(VPUIPDPU::DPUInvariantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::DPUDryRunMode _dryRunMode;

    void fillIDUCfg(mlir::Region& DPURegion,
                    std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
    void fillMPECfg(mlir::Region& DPURegion,
                    std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
    void fillPPECfg(mlir::Region& DPURegion,
                    std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
    void fillODUCfg(mlir::Region& DPURegion,
                    std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
    void fillBarrierCfg(
            VPUIPDPU::DPUInvariantOp origOp,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
    void fillProfilingCfg(
            VPUIPDPU::DPUInvariantOp origOp,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
    void fillStubCfg(std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const;
};

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
