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

#ifndef VPUX_COMPILER_DIALECT_VPUASM_TYPES
#define VPUX_COMPILER_DIALECT_VPUASM_TYPES

include "vpux/compiler/dialect/VPUASM/dialect.td"
include "vpux/compiler/dialect/VPURT/attributes.td"
include "mlir/IR/OpBase.td"


class VPUASM_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
        : TypeDef<VPUASM_Dialect, name, traits, baseCppClass
        > {
    let mnemonic = name;
}

def BufferSectionTypeParam : TypeParameter<"vpux::VPURT::BufferSection", "Buffer Section Enum as type parameter"> {
    let printer = [{ $_printer << "\"" << vpux::VPURT::stringifyBufferSection($_self) << "\"" }];
    let parser = [{ [&] () -> mlir::FailureOr<vpux::VPURT::BufferSection> {
        std::string value;
        if (odsParser.parseString(&value))
            return mlir::failure();

        auto bufferSec = VPURT::symbolizeBufferSection(value);

        if (bufferSec.has_value()) {
            return mlir::FailureOr<vpux::VPURT::BufferSection>(std::move(bufferSec.value()));
        } else {
            return mlir::failure();
        }
    }()
    }];
}

//
// VPU_MemLocationType
//

def VPUASM_MemLocationType : VPUASM_Type<"MemLocation"> {

    let summary = "Generic type to represent a VPU logical memory location";

    let parameters = (ins BufferSectionTypeParam:$section,
                            "uint64_t":$sectionIndex,
                            "uint64_t":$byteOffset
    );

    // let assemblyFormat = "$section `[` $sectionIndex `]` ` ` `<` $byteOffset `>`";
    let assemblyFormat = "$section `[` $sectionIndex `]``<` $byteOffset `>`"; // `[` $sectionIndex `]` ` ` `<` $byteOffset `>`";
}

//
// VPU_BufferTraitsType
//

def VPUASM_BufferTraitsType : VPUASM_Type<"BufferTraits"> {
    let summary = "Generic type to represents discrete VPU specific traits of a buffer";

    let parameters = (ins "uint64_t":$swizzlingKey);

    let assemblyFormat = "`swizzling` `(` $swizzlingKey `)`";
}


//
// BufferType
//

def VPUASM_BufferType : VPUASM_Type<"Buffer"> {

    let summary = "Generic type to represent memory Buffers";

    let parameters = (ins VPUASM_MemLocationType:$location,
                            "mlir::MemRefType":$memref,
                            VPUASM_BufferTraitsType:$traits
    );

    let assemblyFormat = "`<` $location `:` $memref `:` $traits `>`";
}

#endif // VPUX_COMPILER_DIALECT_VPUASM_TYPES
