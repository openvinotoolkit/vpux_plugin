//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUReg37XX
#define VPUX_COMPILER_DIALECT_NPUReg37XX

include "mlir/IR/OpBase.td"

def NPUReg37XX_Dialect : Dialect {
    let summary = "NPU 37XX HW specifics";

    let description = [{
        The **37XX** encapsulates NPU37XX specific information.

    }];

    let name = "NPUReg37XX";

    let cppNamespace = "vpux::NPUReg37XX";

    let dependentDialects = [
        "vpux::VPURegMapped::VPURegMappedDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

#endif
