//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops_interfaces.hpp"
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
        return VPUIPDPU::arch37xx::verifyODUCfgOp(mlir::cast<VPUIPDPU::ODUCfgOp>(op));
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

mlir::LogicalResult vpux::VPUIPDPU::arch37xx::verifyMPECfgOp(VPUIPDPU::MPECfgOp op) {
    if (!hasOptionalSingleInstanceChildren<VPUIPDPU::MPECfgOp, VPUIPDPU::MPEDenormalOperandsFTZOp,
                                           VPUIPDPU::MPEActivationBiasOp, VPUIPDPU::MPEWeightsBiasOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: too many optional child ops", op.getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult vpux::VPUIPDPU::arch37xx::verifyODUCfgOp(VPUIPDPU::ODUCfgOp op) {
    if (!hasMandatorySingleInstanceChildren<VPUIPDPU::ODUCfgOp, VPUIPDPU::ODUOutTensorSizeOp,
                                            VPUIPDPU::ODUOutActivationsOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: missing mandatory child ops", op.getOperationName());
    }
    if (!hasOptionalSingleInstanceChildren<VPUIPDPU::ODUCfgOp, VPUIPDPU::ODUDataReuseOp, VPUIPDPU::ODUPermuteDataOp,
                                           VPUIPDPU::ODUSparsityOp, VPUIPDPU::ODUSwizzleDataOp,
                                           VPUIPDPU::ODUMemoryModeOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: too many optional child ops", op.getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult vpux::VPUIPDPU::arch37xx::verifyDPUVariantOp(VPUIPDPU::DPUVariantOp op) {
    if (!hasMandatorySingleInstanceChildren<VPUIPDPU::DPUVariantOp, VPUIPDPU::ODUOutSubtensorOp>(op)) {
        return errorAt(op.getLoc(), "Operation {0}: missing mandatory child ops", op.getOperationName());
    }

    return ::mlir::success();
}

void vpux::VPUIPDPU::arch37xx::registerVerifiersOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPUIPDPU::VPUIPDPUDialect*) {
        VPUIPDPU::MPECfgOp::attachInterface<MPECfgOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUCfgOp::attachInterface<ODUCfgOpInterfaceModel>(*ctx);
        VPUIPDPU::DPUVariantOp::attachInterface<DPUVariantOpInterfaceModel>(*ctx);
    });
}
