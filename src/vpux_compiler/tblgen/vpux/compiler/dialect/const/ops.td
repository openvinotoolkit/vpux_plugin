//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONST_OPS
#define VPUX_COMPILER_DIALECT_CONST_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/dialect.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/ELFNPU37XX/ops_interfaces.td"


include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//
// Base classes
//

class Const_Op<string mnemonic, list<Trait> traits = []> :
    Op<Const_Dialect, mnemonic, traits>;

//
// DeclareOp
//

def Const_DeclareOp :
        Const_Op<
            "Declare",
            [
                IsolatedFromAbove,
                ConstantLike,
                Pure,
                DeclarationOp,
                DotInterface,
                OpAsmOpInterface,
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>,
                // we can be a user of symbols through SymElementsAttr
                // we verify the validity of the reference through verifySymbolUses()
                DeclareOpInterfaceMethods<SymbolUserOpInterface>
            ]
        > {
    let summary = "Constant tensor/buffer declaration";

    let description = [{
        This operation can perform extra lazy constant folding transformations for constant content.
    }];

    let arguments = (ins
        Const_ContentAttr:$content
    );

    let results = (outs
        AnyTypeOf<[AnyStaticShapeTensor, AnyStaticShapeMemRef]>:$output
    );

    let hasCanonicalizer = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
    let assemblyFormat = "attr-dict type($output) `=` $content";

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::AQUAMARINE;
        }

        void getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
            setNameFn(getResult(), "cst");
        }
    }];
}

class Const_HasChild<list<string> ops> : PredOpTrait<
    "all operations in the body are of types '" # !interleave(ops, ",") # "'",
    CPred<"::llvm::all_of(getBody().getOps(), [](::mlir::Operation& op) { return ::llvm::isa<" # !interleave(ops, ",")  # ">(op); })">
>;

def Const_DataOp :
    Const_Op<
            "Data",
            [
                IsolatedFromAbove,
                OpAsmOpInterface,
                SingleBlock,
                // allow an empty body and don't require a terminator op
                NoTerminator,
                // must be directly under ModuleOp
                HasParent<"mlir::ModuleOp">,
                // disallow any other ops other than const.Rodata in body
                Const_HasChild<["vpux::Const::RodataOp", "vpux::Const::RefOp"]>,
                // make this op identifiable by a symbol
                DeclareOpInterfaceMethods<Symbol>,
                // body region contains ops identified by symbols
                SymbolTable
            ]
        > {
    let summary = "A container for const.Rodata operations";

    let description = [{
        This operation is a container for const.Rodata operations. It does not represent any computation
        or data layouting. It's purely virtual and is used to organize the IR.

        Example:

        ```MLIR
        const.Data @ov_bin {
            const.Rodata @weights1 dense<2.000000e+00> : tensor<48x3x3x3xf32>
            const.Rodata @weights2 dense<1.000000e+00> : tensor<48x3x3x3xf32>
        }
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region AnyRegion:$body);

    let assemblyFormat = [{
        $sym_name $body attr-dict
    }];
}

def IsNotSymElementsAttr : AttrConstraint<
    CPred<"!::llvm::isa<::vpux::Const::SymElementsAttr>($_self)">
>;

def Const_RodataOp :
    Const_Op<
            "Rodata",
            [
                IsolatedFromAbove,
                OpAsmOpInterface,
                // disallow naked Rodata ops
                HasParent<"vpux::Const::DataOp">,
                // make this op identifiable by a symbol
                DeclareOpInterfaceMethods<Symbol>
            ]
        > {
    let summary = "Contains weight information or points to a blob that contains weight information";

    let description = [{
        This operation contains weight information or points to a blob that contains it.
        It can be referenced throughout the IR from multiple places via const.DeclareOp.

        Example:

        ```MLIR
        const.Rodata @weights1 dense<2.000000e+00> : tensor<48x3x3x3xf32>
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        // disallow SymElementsAttr in order to prohibit RodataOp referencing other RodataOp
        ConfinedAttr<ElementsAttr, [IsNotSymElementsAttr]>:$content
    );

    let assemblyFormat = [{
        $sym_name $content attr-dict
    }];
}

def Const_RefOp :
    Const_Op<
            "Ref",
            [
                IsolatedFromAbove,
                OpAsmOpInterface,
                // disallow naked Rodata ops
                HasParent<"vpux::Const::DataOp">,
                // make this op identifiable by a symbol
                DeclareOpInterfaceMethods<Symbol>
            ]
        > {
    let summary = "Represents a reference to a location that contains a folded constant";

    let description = [{
        This operation represents a location in which the result of folding a constant can be stored.
        It is mainly intended to be used for storing the results of the @init function in the context
        of weights separation.

        Example:

        ```MLIR
        const.Data @init_res {
            const.Ref @cst_folded : tensor<48x3x3x3xf32>
        }
        func.func @init() {
            %0 = ...
            const.Store %0, @init_res::@cst_folded : tensor<48x3x3x3xf32>
        }
        func.func @main() {
            %cst = const.Load @init_res::@cst_folded -> tensor<48x3x3x3xf32>
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type
    );

    let assemblyFormat = [{
        $sym_name `:` $type attr-dict
    }];
}

def Const_LoadOp :
        Const_Op<
            "Load",
            [
                DeclarationOp,
                DotInterface,
                IsolatedFromAbove,
                MemoryEffects<[MemRead]>
            ]
        > {
    let summary = "Load a constant identified by a symbol";

    let description = [{
        Reads the value of a constant that is placed in a dedicated section of the IR
        (e.g. const.Data) and identified by a symbol
    }];

    let arguments = (ins
        SymbolRefAttr:$sym_name
    );

    let results = (outs
        AnyTypeOf<[AnyStaticShapeTensor, AnyStaticShapeMemRef]>:$output
    );

    let assemblyFormat = [{
        $sym_name attr-dict `->` type(results)
    }];
}

def Const_StoreOp :
        Const_Op<
            "Store",
            [
                DotInterface,
                IsolatedFromAbove,
                MemoryEffects<[MemWrite]>
            ]
        > {
    let summary = "Store a value in a buffer identified by a symbol";

    let description = [{
        Writes the input value into a buffer identified by a symbol found a dedicated
        section of the IR (e.g. const.Data)
    }];

    let arguments = (ins
        AnyTypeOf<[AnyStaticShapeTensor, AnyStaticShapeMemRef]>:$input,
        SymbolRefAttr:$sym_name
    );

    let assemblyFormat = [{
        $input `,` $sym_name attr-dict `:` type(operands)
    }];
}

#endif
