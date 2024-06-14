//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELFNPU37XX_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_ELFNPU37XX_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// ELFSectionInterface
//

def ELFNPU37XX_SectionInterface : OpInterface<"ElfSectionInterface"> {
    let description = "Interface for the ELF section";

    let cppNamespace = "vpux::ELFNPU37XX";

    let methods = [
        InterfaceMethod<
            "Serialize the Section into an ELF entity",
            "void",
            "serialize", (ins "elf::Writer&":$writer,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Section*>&":$sectionMap,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap)
        >,
        InterfaceMethod<
            "Get the block of the region",
            "mlir::Block*",
            "getBlock", (ins), [{}], [{
                return vpux::ELFNPU37XX::getSectionBlock($_op);
            }]
        >,
        InterfaceMethod<
            "Get the name of the section",
            "llvm::StringRef",
            "getName", (ins), [{}], [{
                return $_op.getSecName();
            }]
        >
    ];
}

def ELFNPU37XX_SymbolObjectInterface : OpInterface<"ElfSymbolObjectInterface"> {
    let description = "Interface for the ELF symbol objects";

    let cppNamespace = "vpux::ELFNPU37XX";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an ELF symbol",
            "void",
            "serialize", (ins "elf::writer::Symbol*":$symbol,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Section*>&":$sectionMap)
        >,
    ];
}

def ELFNPU37XX_RelocationObjectInterface : OpInterface<"ElfRelocationObjectInterface"> {
    let description = "Interface for the ELF relocation objects";

    let cppNamespace = "vpux::ELFNPU37XX";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an ELF relocation object",
            "void",
            "serialize", (ins "elf::writer::Relocation*":$relocation,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap,
                              "vpux::DenseMap<mlir::Value, mlir::DenseMap<mlir::Value, size_t>>&":$sectionsCache)
        >,
    ];
}

//
// OpInterface
//

def ELFNPU37XX_OpInterface : OpInterface<"ElfObjectInterface"> {
    let description = "Interface for the ELF object";

    let cppNamespace = "vpux::ELFNPU37XX";

    let methods = [
        InterfaceMethod<
            "Serialize the Op to the BLOB format",
            "void",
            "serialize", (ins "elf::Writer&":$writer)
        >,
    ];
}

//
// BinaryOpInterface
//

def ELFNPU37XX_BinaryOpInterface : OpInterface<"BinaryOpInterface"> {
    let description = "Interface for the ELF binary serialization of operations";
    let cppNamespace = "vpux::ELFNPU37XX";

    let methods = [
        InterfaceMethod<
            "Serialize the Op to the BLOB format",
            "void",
            "serialize", (ins "elf::writer::BinaryDataSection<uint8_t>&":$binDataSection)
        >,

        InterfaceMethod<
            "Get the size in bytes of the serialized form of this object",
            "size_t",
            "getBinarySize", (ins)
        >,

        StaticInterfaceMethod<
            "Get the alignment requirements for the serialization of this op",
            "size_t",
            "getAlignmentRequirements", (ins)
        >,

        InterfaceMethod<
            "Get the accessing processors of the Op",
            "::vpux::ELFNPU37XX::SectionFlagsAttr",
            "getAccessingProcs", (ins)
        >,

        InterfaceMethod<
            "Get the accessing processors of the Op user(s)",
            "::vpux::ELFNPU37XX::SectionFlagsAttr",
            "getUserProcs", (ins)
        >,

        InterfaceMethod<
            "Get the memory space in which the serialization of this op will reside",
            "vpux::VPURT::BufferSection",
            "getMemorySpace", (ins)
        >,
    ];
}

def ELFNPU37XX_GetOffsetOfOpInterface : OpInterface<"GetOffsetOfOpInterface"> {
    let description = "Interface for the offset computation of different fields of inside serializable operations";
    let cppNamespace = "vpux::ELFNPU37XX";

    let methods = [
        InterfaceMethod<
            "Returns the offset of a field (specified by the 'val' parameter) inside a serializable op",
            "size_t",
            "getOffsetOfWithinOperation", (ins "mlir::Value":$val)
        >,
    ];
}

#endif
