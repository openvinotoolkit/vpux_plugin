//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELF
#define VPUX_COMPILER_DIALECT_ELF

include "mlir/IR/OpBase.td"

// Provide a definition of the ELF dialect in the ODS framework
def ELF_Dialect : Dialect {
    let name = "ELF";

    let cppNamespace = "::vpux::ELF";

    let extraClassDeclaration = [{
        void registerAttributes();
    }];

    let useDefaultAttributePrinterParser = 1;
}

#endif
