//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/dialect.td"

//
// Base classes
//

class VPURegMapped_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::VPURegMapped";
    let genSpecializedAttr = 0;
}

class VPURegMapped_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<VPURegMapped_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class VPURegMapped_Attr<string name, list<Trait> traits = []> : AttrDef<VPURegMapped_Dialect, name, traits> {
    let mnemonic = name;
}

//
// RegFieldDataType
//

def VPURegMapped_RegFieldDataType :
    VPURegMapped_I64EnumAttr<
        "RegFieldDataType",
        "RegFieldDataType that the regField supports",
        [
            I64EnumAttrCase<"UINT", 0>,
            I64EnumAttrCase<"SINT", 1>,
            I64EnumAttrCase<"FP",   2>,
        ]
    > {
}

def VPURegMapped_RegFieldDataTypeAttr : VPURegMapped_EnumAttr<VPURegMapped_RegFieldDataType, "reg_field_data_type">;

//
// RegisterFieldAttr
//

def VPURegMapped_RegisterFieldAttr : VPURegMapped_Attr<"RegisterField"> {
    let description = "This object represents closely a RegisterField Attr";
    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins "vpux::VPURegMapped::RegFieldType":$regField);

    let genVerifyDecl = 1;
    let assemblyFormat = "`<` $regField `>`";
}

//
// RegisterFieldArrayAttr
//

def VPURegMapped_RegisterFieldArrayAttr : TypedArrayAttrBase<VPURegMapped_RegisterFieldAttr,
    "array of RegisterFields">{
        string cppType = "::mlir::ArrayAttr";
    }

//
// RegisterAttr
//

def VPURegMapped_RegisterAttr : VPURegMapped_Attr<"Register"> {
    let description = "This object represents closely a Register Attr";
    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins "vpux::VPURegMapped::RegisterType":$reg);

    let genVerifyDecl = 1;
    let assemblyFormat = "`<` $reg `>`";
}

//
// RegisterArrayAttr
//

def VPURegMapped_RegisterArrayAttr : TypedArrayAttrBase<VPURegMapped_RegisterAttr,
    "array of Registers">{
        string cppType = "::mlir::ArrayAttr";
    }

//
// RegisterMappedAttr
//

def VPURegMapped_RegisterMappedAttr : VPURegMapped_Attr<"RegisterMapped"> {
    let description = "This object represents closely a RegisterMapped Attr";
    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins "vpux::VPURegMapped::RegMappedType":$regMapped);

    let genVerifyDecl = 1;
    let assemblyFormat = "`<` $regMapped `>`";
}

//
// TaskType
//

def VPURegMapped_TaskType :
    VPURegMapped_I64EnumAttr<
        "TaskType",
        "The type of the relevant processing code of the VPU for the task",
        [
            I64EnumAttrCase<"DMA",                 0>,
            I64EnumAttrCase<"ActKernelInvocation", 1>,
            I64EnumAttrCase<"ActKernelRange",      2>,
            I64EnumAttrCase<"DPUInvariant",        3>,
            I64EnumAttrCase<"DPUVariant",          4>,
            I64EnumAttrCase<"M2I",                 5>
        ]
    > {
}

def VPURegMapped_TaskTypeAttr : VPURegMapped_EnumAttr<VPURegMapped_TaskType, "task_type">;

#endif
