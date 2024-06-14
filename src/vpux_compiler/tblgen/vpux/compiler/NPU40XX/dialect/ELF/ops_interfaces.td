//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELF_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_ELF_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// ELFSectionInterface
//

def ELF_SectionInterface : OpInterface<"ElfSectionInterface"> {
    let description = "Interface for the ELF section";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Section into an ELF entity",
            "void",
            "serialize", (ins "elf::Writer&":$writer,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Section*>&":$sectionMap,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap,
                              "vpux::ELF::SymbolReferenceMap&":$symRefMap)
        >,
        InterfaceMethod<
            "Get the block of the region",
            "mlir::Block*",
            "getBlock", (ins), [{}], [{
                return vpux::ELF::getSectionBlock($_op);
            }]
        >,
        InterfaceMethod<
            "Get the name of the section",
            "llvm::StringRef",
            "getSectionName", (ins), [{}], [{
                return $_op.getSymName();
            }]
        >,
        InterfaceMethod<
            "Get the type of the section",
            "std::optional<::vpux::ELF::SectionTypeAttr>",
            "getSectionType", (ins), [{}], [{
                auto op = this->getOperation();
                if (op->hasAttr("secType")) {
                    auto secTypeAttr = (op->getAttr("secType")).template cast<::vpux::ELF::SectionTypeAttrAttr>();
                    return std::optional<::vpux::ELF::SectionTypeAttr>(secTypeAttr.getValue());
                } else {
                    return std::nullopt;
                }
            }]
        >
    ];
}

def ELF_SymbolObjectInterface : OpInterface<"ElfSymbolObjectInterface"> {
    let description = "Interface for the ELF symbol objects";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an ELF symbol",
            "void",
            "serialize", (ins "elf::writer::Symbol*":$symbol,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Section*>&":$sectionMap)
        >,
    ];
}

def ELF_RelocationObjectInterface : OpInterface<"ElfRelocationObjectInterface"> {
    let description = "Interface for the ELF relocation objects";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an ELF relocation object",
            "void",
            "serialize", (ins "elf::writer::Relocation*":$relocation,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap)
        >,
    ];
}

//
// BinaryOpInterface
//

def ELF_BinaryOpInterface : OpInterface<"BinaryOpInterface"> {
    let description = "Interface for the ELF binary serialization of operations";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Op to the BLOB format",
            "void",
            "serialize", (ins "elf::writer::BinaryDataSection<uint8_t>&":$binDataSection), [{}],
            /*defaultImplementation*/ [{
                VPUX_THROW("Unexpected call to interface implementation.");
            }]
        >,

        InterfaceMethod<
            "Get the size in bytes of the serialized form of this object",
            "size_t",
            "getBinarySize", (ins), [{}],
            /*defaultImplementation*/ [{
                VPUX_THROW("Unexpected call to interface implementation.");
            }]
        >,

        InterfaceMethod<
            "Serialize the Op to the BLOB format, using a symbol reference map for faster look-up",
            "void",
            "serializeCached", (ins "elf::writer::BinaryDataSection<uint8_t>&":$binDataSection, "ELF::SymbolReferenceMap&":$symRefMap), [{}],
            /*defaultImplementation*/ [{
                $_op.serialize(binDataSection);
            }]
        >,

        InterfaceMethod<
            "Get the size in bytes of the serialized form of this object, using a symbol reference map for faster look-up",
            "size_t",
            "getBinarySizeCached", (ins "ELF::SymbolReferenceMap&":$symRefMap), [{}],
            /*defaultImplementation*/ [{
                return $_op.getBinarySize();
            }]
        >
    ];
}

//
// WrappableOpInterface
//

def ELF_WrappableOpInterface : OpInterface<"WrappableOpInterface"> {
    let description = "Interface provided by ELF dialect that the dialect passes will interact with";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Get the signature name for this op to be placed in",
            "std::optional<vpux::ELF::SectionSignature>",
            "getSectionSignature", (ins)
        >,

        InterfaceMethod<
            "Check if an op is allocatable in memory",
            "bool",
            "hasMemoryFootprint", (ins)
        >,

        InterfaceMethod<
            "Get the alignment requirements for the serialization of this op",
            "size_t",
            "getAlignmentRequirements", (ins)
        >,

        InterfaceMethod<
            "Get the accessing processors of the Op",
            "::vpux::ELF::SectionFlagsAttr",
            "getAccessingProcs", (ins "mlir::SymbolUserMap&":$symbolUserMap)
        >,

        InterfaceMethod<
            "Get the accessing processors of the Op user(s)",
            "::vpux::ELF::SectionFlagsAttr",
            "getUserProcs", (ins)
        >,

        InterfaceMethod<
            "Set the offset of an operation inside an ELF section op",
            "void",
            "setMemoryOffset", (ins "mlir::IntegerAttr":$offset), [{}],
            /*defaultImplementation*/ [{
                this->getOperation()->setAttr(WrappableOpInterface::elfMemOffsetAttrName(), offset);
            }]
        >,

        InterfaceMethod<
            "Get the offset of an operation inside an ELF section op",
            "uint64_t",
            "getMemoryOffset", (ins), [{}],
            /*defaultImplementation*/ [{
                auto op = this->getOperation();
                if(op->hasAttr(WrappableOpInterface::elfMemOffsetAttrName())) {
                    auto attr = op->template getAttrOfType<mlir::IntegerAttr>(WrappableOpInterface::elfMemOffsetAttrName());
                    return attr.getUInt();
                }
                else {
                    return 0;
                }
            }]
        >
    ];

    let extraClassDeclaration = [{
        static llvm::StringLiteral elfMemOffsetAttrName() {
            return "elfMemOffsetAttrKey";
        }
    }];
}

//
// RelocatableOpInterface
//

def ELF_RelocatableOpInterface : OpInterface<"RelocatableOpInterface"> {
    let description = "ELF Interface for the operations that need relocations";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Get all necesarry information in order to construct the op's relocations",
            "std::vector<vpux::ELF::RelocationInfo>",
            "getRelocationInfo", (ins "ELF::SymbolReferenceMap&":$symRefMap)
        >
    ];
}

#endif
