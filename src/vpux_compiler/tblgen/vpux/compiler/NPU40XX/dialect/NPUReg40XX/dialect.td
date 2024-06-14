//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUReg40XX
#define VPUX_COMPILER_DIALECT_NPUReg40XX

include "mlir/IR/OpBase.td"

def NPUReg40XX_Dialect : Dialect {
    let summary = "NPU 40XX HW specifics";

    let description = [{
        The **40XX** encapsulates NPU40XX specific information.

    }];

    let name = "NPUReg40XX";

    let cppNamespace = "vpux::NPUReg40XX";

    let dependentDialects = [
        "vpux::VPURegMapped::VPURegMappedDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultAttributePrinterParser = 1;
}

#endif
