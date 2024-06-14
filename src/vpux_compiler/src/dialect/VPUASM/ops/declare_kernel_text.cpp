//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/reader.hpp>

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// DeclareKernelTextOp
//

void vpux::VPUASM::DeclareKernelTextOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    const auto text = vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".text"});
    binDataSection.appendData(text.data(), text.size());
}

size_t vpux::VPUASM::DeclareKernelTextOp::getBinarySize() {
    return vpux::ELF::getKernelELF(getOperation(), getKernelPath(), {".text"}).size();
}

// The .text sections for the sw layers must be 1kB aligned as an ActShave requirement
size_t vpux::VPUASM::DeclareKernelTextOp::getAlignmentRequirements() {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DeclareKernelTextOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DeclareKernelTextOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DeclareKernelTextOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "text"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DeclareKernelTextOp::hasMemoryFootprint() {
    return true;
}
