//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUReg40XX_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_NPUReg40XX_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// SingleOutputAsIndexOp
//

def NPUReg40XX_SingleOutputAsIndexOp : NativeOpTrait<"SingleOutputAsIndexOp"> {
    let cppNamespace = "vpux::NPUReg40XX";
}

#endif  // VPUX_COMPILER_DIALECT_NPUReg40XX_OPS_INTERFACES
