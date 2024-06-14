//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU
#define VPUX_COMPILER_DIALECT_VPUIPDPU

include "mlir/IR/OpBase.td"

def VPUIPDPU_Dialect : Dialect {
    let summary = "VPU NN DPU Execution RunTime Dialect";

    let description = [{
The **VPUIPDPU Dialect** represents NN DPU RunTime IR in terms of MLIR framework.

It specializes the NN IR describing the DPU configuration to a hierarchy of modular operations,
based on a functional-wise view of the DPU sub-blocks and their parameters.

[./VPUIP/_ops_interfaces.md]
    }];

    let name = "VPUIPDPU";

    let cppNamespace = "vpux::VPUIPDPU";
}

#endif
