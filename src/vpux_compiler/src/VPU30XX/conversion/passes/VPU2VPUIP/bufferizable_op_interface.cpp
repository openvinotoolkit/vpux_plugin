//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"

//
// registerBufferizableOpInterfaces
//

void vpux::arch30xx::registerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    vpux::registerConstDeclareBufferizableOpInterfaces(registry);
    vpux::registerFuncAndReturnBufferizableOpInterfaces(registry);
    vpux::registerVpuNceBufferizableOpInterfaces(registry);
    vpux::registerVPUBufferizableOpInterfaces(registry);
    vpux::registerNCEClusterTilingBufferizableOpInterfaces(registry);
}
