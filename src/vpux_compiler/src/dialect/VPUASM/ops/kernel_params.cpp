//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/SymbolTable.h>
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <kernels/inc/common_types.h>

using namespace vpux;

//
// KernelParamsOp
//

vpux::NDTypeInterface getNdTypeFromBufferSymRef(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef) {
    auto buffOp = symRefMap.lookupSymbol(symRef);

    if (mlir::isa<VPUASM::DeclareBufferOp>(buffOp)) {
        auto buffer = mlir::cast<VPUASM::DeclareBufferOp>(buffOp);
        return buffer.getBufferType().getMemref().cast<vpux::NDTypeInterface>();
    } else if (mlir::isa<VPUASM::ConstBufferOp>(buffOp)) {
        auto buffer = mlir::cast<VPUASM::ConstBufferOp>(buffOp);
        return buffer.getBufferType().getMemref().cast<vpux::NDTypeInterface>();
    }
    VPUX_THROW("Could not find symbol name entry for input {0}", symRef);
}

vpux::NDTypeInterface getNdTypeFromBufferSymRef(mlir::Operation* op, mlir::SymbolRefAttr symRef) {
    auto buffOp = mlir::SymbolTable::lookupNearestSymbolFrom(op, symRef);

    if (mlir::isa<VPUASM::DeclareBufferOp>(buffOp)) {
        auto buffer = mlir::cast<VPUASM::DeclareBufferOp>(buffOp);
        return buffer.getBufferType().getMemref().cast<vpux::NDTypeInterface>();
    } else if (mlir::isa<VPUASM::ConstBufferOp>(buffOp)) {
        auto buffer = mlir::cast<VPUASM::ConstBufferOp>(buffOp);
        return buffer.getBufferType().getMemref().cast<vpux::NDTypeInterface>();
    }
    VPUX_THROW("Could not find symbol name entry for input {0}", symRef);
}

void vpux::VPUASM::KernelParamsOp::serializeCached(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                   ELF::SymbolReferenceMap& symRefMap) {
    // E#81501: this logic should be present either at VPUMI lowering or VPUASM lowering... doing it here temporarily

    std::vector<uint8_t> inputDimsVector, outputDimsVector;
    std::vector<uint8_t> inputStridesVector, outputStridesVector;

    auto inputBuffs = getInputs();
    auto outputBuffs = getOutputs();

    auto insertDimsIntoVector = [](std::vector<uint8_t>& dimsVector, vpux::NDTypeInterface ndType) {
        auto shape = ndType.getShape();
        const auto dimsOrder = ndType.getDimsOrder();
        const auto memShape = dimsOrder.toMemoryOrder(shape);

        for (auto& memDim : memShape | reversed) {
            auto dim = checked_cast<int32_t>(memDim);
            ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&dim), sizeof(dim));
            dimsVector.insert(dimsVector.end(), valueAsArray.begin(), valueAsArray.end());
        }
    };

    auto insertStridesIntoVector = [](std::vector<uint8_t>& stridesVector, vpux::NDTypeInterface ndType) {
        auto strides = ndType.getMemStrides();
        for (auto&& stride : strides | reversed) {
            ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&stride), sizeof(stride));
            stridesVector.insert(stridesVector.end(), valueAsArray.begin(), valueAsArray.end());
        }
    };

    // input Dims & Strides
    for (auto inputBuff : inputBuffs) {
        auto inputSymRef = inputBuff.cast<mlir::SymbolRefAttr>();
        auto inputNdType = getNdTypeFromBufferSymRef(symRefMap, inputSymRef);

        insertDimsIntoVector(inputDimsVector, inputNdType);
        insertStridesIntoVector(inputStridesVector, inputNdType);
    }

    // output Dims & Strides
    for (const auto outputBuff : outputBuffs) {
        auto outputSymRef = outputBuff.cast<mlir::SymbolRefAttr>();
        auto outputNdType = getNdTypeFromBufferSymRef(symRefMap, outputSymRef);

        insertDimsIntoVector(outputDimsVector, outputNdType);
        insertStridesIntoVector(outputStridesVector, outputNdType);
    }

    auto params = getKernelParams();

    auto dense_elem_data = params.getValues<uint8_t>();

    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    // serialize actual kernel params
    binDataSection.appendData(data_vector.data(), data_vector.size());

    // serialize IO dims/strides
    binDataSection.appendData(inputDimsVector.data(), inputDimsVector.size());
    binDataSection.appendData(inputStridesVector.data(), inputStridesVector.size());
    binDataSection.appendData(outputDimsVector.data(), outputDimsVector.size());
    binDataSection.appendData(outputStridesVector.data(), outputStridesVector.size());
    return;
}

size_t vpux::VPUASM::KernelParamsOp::getBinarySizeCached(ELF::SymbolReferenceMap& symRefMap) {
    auto params = getKernelParams();
    auto dense_elem_data = params.getValues<uint8_t>();
    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    auto inputBuffs = getInputs();
    auto outputBuffs = getOutputs();

    size_t inputDimsSize = 0;
    size_t inputStridesSize = 0;

    for (const auto inputBuff : inputBuffs) {
        auto inputSymRef = inputBuff.cast<mlir::SymbolRefAttr>();
        auto ndType = getNdTypeFromBufferSymRef(symRefMap, inputSymRef);

        inputDimsSize += sizeof(int32_t) * ndType.getShape().size();
        inputStridesSize += sizeof(int64_t) * ndType.getMemStrides().size();
    }

    size_t outputDimsSize = 0;
    size_t outputStridesSize = 0;

    for (const auto outputBuff : outputBuffs) {
        auto outputSymRef = outputBuff.cast<mlir::SymbolRefAttr>();
        auto ndType = getNdTypeFromBufferSymRef(symRefMap, outputSymRef);

        outputDimsSize += sizeof(int32_t) * ndType.getShape().size();
        outputStridesSize += sizeof(int64_t) * ndType.getMemStrides().size();
    }

    return data_vector.size() + inputDimsSize + outputDimsSize + inputStridesSize + outputStridesSize;
}

size_t vpux::VPUASM::KernelParamsOp::getParamsStructSize() {
    auto params = getKernelParams();
    auto dense_elem_data = params.getValues<uint8_t>();
    auto data_vector = std::vector<uint8_t>(dense_elem_data.begin(), dense_elem_data.end());

    return data_vector.size();
}

// The parameter structs for the sw layers must be 64Byte aligned as an ActShave requirement
size_t vpux::VPUASM::KernelParamsOp::getAlignmentRequirements() {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::KernelParamsOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::KernelParamsOp::getUserProcs() {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::KernelParamsOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "params"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::KernelParamsOp::hasMemoryFootprint() {
    return true;
}

std::vector<ELF::RelocationInfo> vpux::VPUASM::KernelParamsOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    // To be removed after E#100513
    std::vector<ELF::RelocationInfo> relocs;

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    auto kernelInputs = getInputs();

    for (auto input : kernelInputs | indexed) {
        auto inputSymRef = input.value().cast<mlir::SymbolRefAttr>();
        auto relocType = VPUASM::getBufferLocation(symRefMap, inputSymRef) == VPURT::BufferSection::CMX_NN
                                 ? ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET
                                 : ELF::RelocationType::R_VPU_32;

        relocs.push_back(ELF::RelocationInfo(
                inputSymRef, targetSection,
                input.index() * sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, dataAddr), relocType,
                ELF::getOffsetOfSymRef(symRefMap, inputSymRef)));
    }

    auto kernelOutputs = getOutputs();

    for (auto output : kernelOutputs | indexed) {
        auto outputSymRef = output.value().cast<mlir::SymbolRefAttr>();
        auto relocType = VPUASM::getBufferLocation(symRefMap, outputSymRef) == VPURT::BufferSection::CMX_NN
                                 ? ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET
                                 : ELF::RelocationType::R_VPU_32;

        relocs.push_back(ELF::RelocationInfo(outputSymRef, targetSection,
                                             (kernelInputs.size() + output.index()) * sizeof(sw_params::MemRefData) +
                                                     offsetof(sw_params::MemRefData, dataAddr),
                                             relocType, ELF::getOffsetOfSymRef(symRefMap, outputSymRef)));
    }

    auto getNDTypeIfFromSymRef = [&symRefMap](mlir::SymbolRefAttr symRef) {
        auto memoryOp = symRefMap.lookupSymbol(symRef);
        auto bufferTypeAttr = memoryOp->getAttrOfType<mlir::TypeAttr>("buffer_type");
        VPUX_THROW_UNLESS(bufferTypeAttr, "Operation is not a memory-descriptive op");

        auto bufferType = bufferTypeAttr.getValue().cast<VPUASM::BufferType>();
        auto NDTypeIf = bufferType.getMemref().cast<vpux::NDTypeInterface>();
        return NDTypeIf;
    };

    auto baseOffset = getMemoryOffset();
    auto sizeOfParamsStruct = getParamsStructSize();
    auto addend = baseOffset + sizeOfParamsStruct;
    auto fullSourceSymRef = ELF::composeSectionObjectSymRef(targetSection, this->getOperation());

    for (auto kernelInputIt : kernelInputs | indexed) {
        relocs.push_back(ELF::RelocationInfo(
                fullSourceSymRef, targetSection,
                kernelInputIt.index() * sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, dimsAddr),
                ELF::RelocationType::R_VPU_32, addend));

        auto inputSymRef = kernelInputIt.value().cast<mlir::SymbolRefAttr>();
        addend += sizeof(int32_t) * getNDTypeIfFromSymRef(inputSymRef).getShape().size();
    }

    for (auto kernelInputIt : kernelInputs | indexed) {
        relocs.push_back(ELF::RelocationInfo(
                fullSourceSymRef, targetSection,
                kernelInputIt.index() * sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, stridesAddr),
                ELF::RelocationType::R_VPU_32, addend));

        auto inputSymRef = kernelInputIt.value().cast<mlir::SymbolRefAttr>();
        addend += sizeof(int64_t) * getNDTypeIfFromSymRef(inputSymRef).getMemStrides().size();
    }

    for (auto kernelOutputIt : kernelOutputs | indexed) {
        relocs.push_back(
                ELF::RelocationInfo(fullSourceSymRef, targetSection,
                                    (kernelInputs.size() + kernelOutputIt.index()) * sizeof(sw_params::MemRefData) +
                                            offsetof(sw_params::MemRefData, dimsAddr),
                                    ELF::RelocationType::R_VPU_32, addend));

        auto outputSymRef = kernelOutputIt.value().cast<mlir::SymbolRefAttr>();
        addend += sizeof(int32_t) * getNDTypeIfFromSymRef(outputSymRef).getShape().size();
    }

    for (auto kernelOutputIt : kernelOutputs | indexed) {
        relocs.push_back(
                ELF::RelocationInfo(fullSourceSymRef, targetSection,
                                    (kernelInputs.size() + kernelOutputIt.index()) * sizeof(sw_params::MemRefData) +
                                            offsetof(sw_params::MemRefData, stridesAddr),
                                    ELF::RelocationType::R_VPU_32, addend));

        auto outputSymRef = kernelOutputIt.value().cast<mlir::SymbolRefAttr>();
        addend += sizeof(int64_t) * getNDTypeIfFromSymRef(outputSymRef).getMemStrides().size();
    }

    return relocs;
}
