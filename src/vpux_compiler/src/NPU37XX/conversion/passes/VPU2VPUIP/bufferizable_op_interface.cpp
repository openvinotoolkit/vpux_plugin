//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_sw_ops_interface.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"

using namespace vpux;

namespace {

// ConvertOp is always bufferized to VPUIP Software Kernel Operation in arch37XX
void registerConvertOpBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::ConvertOp::attachInterface<SoftwareLayerOpBufferizeModel<VPU::ConvertOp>>(*ctx);
    });
}

}  // namespace

//
// registerBufferizableOpInterfaces
//

void vpux::arch37xx::registerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    vpux::registerConstDeclareBufferizableOpInterfaces(registry);
    vpux::registerFuncAndReturnBufferizableOpInterfaces(registry);
    vpux::registerSoftwareLayerBufferizableOpInterfaces(registry);
    vpux::registerVpuNceBufferizableOpInterfaces(registry);
    vpux::registerVPUBufferizableOpInterfaces(registry);
    vpux::registerNCEClusterTilingBufferizableOpInterfaces(registry);
    registerConvertOpBufferizableOpInterfaces(registry);
}
