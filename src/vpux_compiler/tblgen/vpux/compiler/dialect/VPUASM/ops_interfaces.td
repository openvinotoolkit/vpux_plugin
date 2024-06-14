//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUASM_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUASM_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def VPUASM_SymbolicRefInterface : OpInterface<"SymbolicRefInterface"> {
    let description = "Interface to describe the nature that an OP symbolically references another OP in VPUASM diallect";

    let cppNamespace = "vpux::VPUASM";

    let methods = [
        InterfaceMethod<
            "Interface to get the binary offset of the binary address that is represented by the symbolic reference",
            "uint64_t", "offsetOfSymRef",
            (ins "mlir::SymbolRefAttr":$symRef)
        >
    ];
}

def VPUASM_IndexedTaskInterface : OpInterface<"IndexedTaskInterface"> {
    let description = [{
        Interface to encapsulate task-list semantics into ops. In VPUASM the ops don't have a type-system, as such
        the task-list information will be stored as attributes. IndexedTaskInterface will provide common methods
        and verifiers for such ops.

        Ops that define this interface *MUST* define a "VPUASM_IndexTypeAttr" with the name "task_index"
    }];

    let cppNamespace = "vpux::VPUASM";

    let methods = [
        StaticInterfaceMethod<
            "Name of the listIndex attribute",
            "llvm::StringRef",
            "getListIndexAttrName", (ins), [{}],
            /*defaultImplementation=*/ [{
                return "task_index";
            }]
        >,
        InterfaceMethod<
            "Get index type",
            "VPURegMapped::IndexType",
            "getIndexType", (ins), [{}],
            /*defaultImplementation=*/ [{
                auto attr = this->getOperation()->template getAttrOfType<mlir::TypeAttr>(getListIndexAttrName());
                return attr.getValue().template cast<VPURegMapped::IndexType>();
            }]
        >,
        InterfaceMethod<
            "Get the tile index of the current element",
            "size_t", "getTileIndex", (ins ), [{}],
            /*defaultImplementation=*/ [{
                auto indexType = getIndexType();
                return indexType.getTileIdx();
            }]
        >,
        InterfaceMethod<
            "Get the list index of the current element",
            "size_t", "getListIndex", (ins ), [{}],
            /*defaultImplementation=*/ [{
                auto indexType = getIndexType();
                return indexType.getListIdx();
            }]
        >,
        InterfaceMethod<
            "get the task index of the current element",
            "size_t", "getIndex", (ins), [{}],
            /*defaultImplementation=*/ [{
                auto indexType = getIndexType();
                return indexType.getValue();
            }]
        >
    ];
}

#endif
