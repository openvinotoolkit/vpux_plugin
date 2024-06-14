//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>
using namespace npu40xx;

using namespace vpux;

//
// WorkItemOp
//

void vpux::VPUASM::WorkItemOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    return;
}

size_t vpux::VPUASM::WorkItemOp::getBinarySize() {
    return sizeof(nn_public::VpuWorkItem);
}

size_t vpux::VPUASM::WorkItemOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuWorkItem);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::WorkItemOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::WorkItemOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::WorkItemOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "workItem"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::WorkItemOp::hasMemoryFootprint() {
    return true;
}

std::vector<ELF::RelocationInfo> vpux::VPUASM::WorkItemOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    auto firstTaskOffset = offsetof(nn_public::VpuWorkItem, wi_desc_ptr);
    if (auto firstTask = getFirstTask()) {
        if (getTaskType() == VPURegMapped::TaskType::DMA) {
            relocs.push_back(ELF::RelocationInfo(firstTask, targetSection, firstTaskOffset,
                                                 ELF::RelocationType::R_VPU_64,
                                                 ELF::getOffsetOfSymRef(symRefMap, firstTask)));
        } else {
            relocs.push_back(ELF::RelocationInfo(firstTask, targetSection, firstTaskOffset,
                                                 ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET,
                                                 ELF::getOffsetOfSymRef(symRefMap, firstTask)));
        }
    }

    return relocs;
}
