//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI40XX
#define VPUX_COMPILER_DIALECT_VPUMI40XX

include "mlir/IR/OpBase.td"

def VPUMI40XX_Dialect : Dialect {
    let summary = "VPU NN Mapped Inference RunTime Dialect for arch NPU40XX";

    let description = [{
The **VPUMI40XX Dialect** represents NN RunTime IR together with MappedInference
    constructs in terms of the MLIR framework.
    This is specific for NPU40XX HW architecture.

[./VPUMI40XX/_ops_interfaces.md]
    }];

    let name = "VPUMI40XX";

    let cppNamespace = "vpux::VPUMI40XX";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "mlir::func::FuncDialect",
        "mlir::quant::QuantizationDialect",
        "vpux::VPURegMapped::VPURegMappedDialect",
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultAttributePrinterParser = 1;
}

#endif
