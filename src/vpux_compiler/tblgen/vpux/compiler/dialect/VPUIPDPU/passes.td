//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_PASSES
#define VPUX_COMPILER_DIALECT_VPUIPDPU_PASSES

include "mlir/Pass/PassBase.td"

//
// Expand DPU configuration
//

def ExpandDPUConfig : PassBase<"expand-dpu-config", "vpux::FunctionPass"> {
    let summary = "Convert VPUASM.DPUInvariant/Variant to VPUIPDPU.DPUInvariant/Variant";

    let description = [{
        VPUASM level represents ops in their logical form. They have inputs, outputs, associated barriers and so on.
        VPUIPDPU is still a logical representation of the DPU config., but a more detailed one and also HW arch sensitive.
    }];

    let constructor = "vpux::VPUIPDPU::createExpandDPUConfigPass()";
    let dependentDialects = [
        "vpux::VPUASM::VPUASMDialect",
        "vpux::VPUIPDPU::VPUIPDPUDialect",
        "vpux::ELF::ELFDialect"
    ];
}

#endif
