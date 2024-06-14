//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef VPUX_COMPILER_DIALECT_VPUASM_ATTRS
#define VPUX_COMPILER_DIALECT_VPUASM_ATTRS

include "mlir/IR/OpBase.td"

/// A TypeAttr for memref types.
def VPUASM_BufferTypeAttr
    : TypeAttrBase<"::vpux::VPUASM::BufferType", "VPU ASM Buffer type attribute"> {
    let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

def VPUASM_IndexTypeAttr
    : TypeAttrBase<"::vpux::VPURegMapped::IndexType", "TypeAttr for Mapped Inference IndexType" > {
    let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

#endif
