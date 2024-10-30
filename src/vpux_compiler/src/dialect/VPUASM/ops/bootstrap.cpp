//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// BootstrapOp
//

void vpux::VPUASM::BootstrapOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binaryDataSection) {
    uint32_t barId = getBarrierId();
    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&barId);
    binaryDataSection.appendData(ptrCharTmp, getBinarySize());
    return;
}

size_t vpux::VPUASM::BootstrapOp::getBinarySize() {
    return sizeof(uint32_t);
}

size_t vpux::VPUASM::BootstrapOp::getAlignmentRequirements() {
    return alignof(uint32_t);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::BootstrapOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::BootstrapOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "bootstrap"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::BootstrapOp::hasMemoryFootprint() {
    return true;
}
