//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/utils/traits_utils.hpp"

#include <mlir/IR/Operation.h>

using namespace vpux;

namespace {

class MPECfgOpInterfaceModel final :
        public VPUIPDPU::MPECfgOpInterface::ExternalModel<MPECfgOpInterfaceModel, VPUIPDPU::MPECfgOp> {
public:
    mlir::LogicalResult verifyInnerOps(mlir::Operation* op) const {
        return VPUIPDPU::arch37xx::verifyMPECfgOp(mlir::cast<VPUIPDPU::MPECfgOp>(op));
    }
};

class ODUCfgOpInterfaceModel final :
        public VPUIPDPU::ODUCfgOpInterface::ExternalModel<ODUCfgOpInterfaceModel, VPUIPDPU::ODUCfgOp> {
public:
    mlir::LogicalResult verifyInnerOps(mlir::Operation* op) const {
        return VPUIPDPU::arch40xx::verifyODUCfgOp(mlir::cast<VPUIPDPU::ODUCfgOp>(op));
    }
};

class DPUVariantOpInterfaceModel final :
        public VPUIPDPU::DPUVariantOpInterface::ExternalModel<DPUVariantOpInterfaceModel, VPUIPDPU::DPUVariantOp> {
public:
    mlir::LogicalResult verifyInnerOps(mlir::Operation* op) const {
        return VPUIPDPU::arch37xx::verifyDPUVariantOp(mlir::cast<VPUIPDPU::DPUVariantOp>(op));
    }
};

}  // namespace

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::verifyODUCfgOp(VPUIPDPU::ODUCfgOp op) {
    auto verifier37XX = VPUIPDPU::arch37xx::verifyODUCfgOp(op);
    if (verifier37XX.failed()) {
        return verifier37XX;
    }

    if (!hasOptionalSingleInstanceChildren<VPUIPDPU::ODUCfgOp, VPUIPDPU::ODUCmxPortsOp,
                                           VPUIPDPU::ODUWriteCombineBufferOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: too many optional child ops", op.getOperationName());
    }

    return ::mlir::success();
}

void vpux::VPUIPDPU::arch40xx::registerVerifiersOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPUIPDPU::VPUIPDPUDialect*) {
        VPUIPDPU::MPECfgOp::attachInterface<MPECfgOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUCfgOp::attachInterface<ODUCfgOpInterfaceModel>(*ctx);
        VPUIPDPU::DPUVariantOp::attachInterface<DPUVariantOpInterfaceModel>(*ctx);
    });
}
