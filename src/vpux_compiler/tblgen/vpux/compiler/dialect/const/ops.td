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

//
// Base classes
//

class Const_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            Const_Dialect,
            mnemonic,
            !listconcat(
              traits,
              [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>
              ]
            )
        >;

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
                OpAsmOpInterface
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

#endif
