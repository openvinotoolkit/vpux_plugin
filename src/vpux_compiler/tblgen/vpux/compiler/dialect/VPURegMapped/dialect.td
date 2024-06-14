//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED

include "mlir/IR/OpBase.td"

def VPURegMapped_Dialect : Dialect {
    let summary = "VPU NN Register Mapped Dialect";

    let description = [{
The **VPURegMapped Dialect** defines attributes and types common
    for all supported VPU HW architectures. It is used combined with
    arch specific register map dialects (e.g. NPU37XX, NPU40XX)
    to represents NN RunTime IR constructs in terms of the MLIR framework.

[./VPURegMapped/_ops_interfaces.md]
    }];

    let name = "VPURegMapped";

    let cppNamespace = "vpux::VPURegMapped";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect",
        "mlir::func::FuncDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

#endif
