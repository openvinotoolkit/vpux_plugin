//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/attributes.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
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

    void fillIDUCfg(mlir::Region& DPURegion, vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillMPECfg(mlir::Region& DPURegion, vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillPPECfg(mlir::Region& DPURegion, vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillODUCfg(mlir::Region& DPURegion, vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillBarrierCfg(VPUIPDPU::DPUInvariantOp origOp,
                        vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillProfilingCfg(VPUIPDPU::DPUInvariantOp origOp,
                          vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillStubCfg(vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const;
};

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
