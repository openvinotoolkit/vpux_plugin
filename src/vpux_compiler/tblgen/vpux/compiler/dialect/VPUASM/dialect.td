//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUASM
#define VPUX_COMPILER_DIALECT_VPUASM

include "mlir/IR/OpBase.td"

def VPUASM_Dialect : Dialect {
    let summary = "VPU NN Assembly dialect";

    let description = [{
The **VPUMI Dialect** represents the Inference with the view and semantics of the NN RunTime.
Additionally to the **VPUMI Dialect** it represents the tasks with **symbolic** relationships between them.

It handles such VPU-specifics as:

* Memory layout of tasks and operations.
* Memory relationships as exposed by symbolic references.
* As with **VPUMI** it represents the operational view of the NN RunTime.

Main intent is to represent logic and passes that solve memory layout and relationship problems

[./VPUASM/_ops_interfaces.md]
    }];

    let name = "VPUASM";

    let cppNamespace = "vpux::VPUASM";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::ELF::ELFDialect",
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPUMI40XX::VPUMI40XXDialect",
        "mlir::func::FuncDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];

    let useDefaultTypePrinterParser = 1;
}

#endif
