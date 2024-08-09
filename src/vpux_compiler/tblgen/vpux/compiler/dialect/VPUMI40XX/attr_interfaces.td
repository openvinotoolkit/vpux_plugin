//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI40XX_ATTR_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUMI40XX_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

//
// DMATransactionAttrInterface
//

def VPUMI40XX_DMATransactionAttrInterface : AttrInterface<"DMATransactionAttr"> {
    let description = [{
        Interface for DMA Transaction attributes.
    }];

    let cppNamespace = "vpux::VPUMI40XX";

    let methods = [
        InterfaceMethod<[{
            Returns the reduced DMA transaction dimensions.
            }],
            "vpux::DMATransaction", "getDMATransaction",
            (ins)
        >
    ];
}

#endif
