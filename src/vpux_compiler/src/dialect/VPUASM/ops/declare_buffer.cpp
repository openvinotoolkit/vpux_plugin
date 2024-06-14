//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/utils/core/logger.hpp"

using namespace vpux;

//
// DeclareBufferOp
//

void VPUASM::DeclareBufferOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
    return;
}

size_t VPUASM::DeclareBufferOp::getBinarySize() {
    const auto type = getBufferType().getMemref().cast<vpux::NDTypeInterface>();
    return type.getTotalAllocSize().count();
}

size_t VPUASM::DeclareBufferOp::getAlignmentRequirements() {
    // DeclareBuffers are addressed by the mem-schedulers, so can't override anything
    return ELF::VPUX_NO_ALIGNMENT;
}

ELF::SectionFlagsAttr VPUASM::DeclareBufferOp::getAccessingProcs(mlir::SymbolUserMap& symbolUserMap) {
    auto flags = ELF::SectionFlagsAttr::SHF_NONE;
    const auto users = symbolUserMap.getUsers(getOperation());
    for (auto user : users) {
        flags = flags | mlir::cast<ELF::WrappableOpInterface>(user).getUserProcs();
    }

    return flags;
}

ELF::SectionFlagsAttr VPUASM::DeclareBufferOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

void VPUASM::DeclareBufferOp::setMemoryOffset(mlir::IntegerAttr) {
    // declareBufferOp's offset is implicit in it's memLocation
    return;
}

uint64_t VPUASM::DeclareBufferOp::getMemoryOffset() {
    auto location = getBufferType().getLocation();
    return location.getByteOffset();
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DeclareBufferOp::getSectionSignature() {
    const auto buffType = getBufferType();
    const auto location = buffType.getLocation();
    const auto section = location.getSection();

    auto type = (section == VPURT::BufferSection::CMX_NN) ? ELF::SectionTypeAttr::VPU_SHT_CMX_WORKSPACE
                                                          : ELF::SectionTypeAttr::SHT_NOBITS;
    bool isInputOrOutputBuffer =
            section == VPURT::BufferSection::NetworkInput || section == VPURT::BufferSection::NetworkOutput;
    const auto name = vpux::ELF::generateSignature("buffer", buffType);

    if (isInputOrOutputBuffer) {
        return std::nullopt;
    }

    if (section == VPURT::BufferSection::CMX_NN || section == VPURT::BufferSection::DDR) {
        ELF::SectionFlagsAttr flags = (section == VPURT::BufferSection::CMX_NN)
                                              ? ELF::SectionFlagsAttr::SHF_NONE
                                              : ELF::SectionFlagsAttr::SHF_WRITE | ELF::SectionFlagsAttr::SHF_ALLOC;
        return ELF::SectionSignature(name, flags, type);
    }
    return std::nullopt;
}

bool vpux::VPUASM::DeclareBufferOp::hasMemoryFootprint() {
    return false;
}
