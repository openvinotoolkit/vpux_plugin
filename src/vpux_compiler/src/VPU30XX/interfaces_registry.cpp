//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/interfaces_registry.hpp"
#include <mlir/IR/DialectRegistry.h>

#include "vpux/compiler/VPU30XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"
#include "vpux/compiler/VPU30XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/IR/ops_interfaces.hpp"

namespace vpux {

void InterfacesRegistry30XX::registerInterfaces(mlir::DialectRegistry& registry) {
    IE::arch30xx::registerElemTypeInfoOpInterfaces(registry);
    VPU::arch30xx::registerLayerWithPostOpModelInterface(registry);
    VPU::arch30xx::registerLayoutInfoOpInterfaces(registry);
    VPUIP::arch30xx::registerAlignedChannelsOpInterfaces(registry);
    vpux::arch30xx::registerBufferizableOpInterfaces(registry);
}

}  // namespace vpux
