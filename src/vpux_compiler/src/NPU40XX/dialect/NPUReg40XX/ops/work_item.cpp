//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>
using namespace npu40xx;

using namespace vpux;

void vpux::NPUReg40XX::WorkItemOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuWorkItem workItem = {};
    auto workItemDesc = getWorkItemDescriptor().getRegMapped();
    auto serializedworkItemDesc = workItemDesc.serialize();

    memcpy(reinterpret_cast<uint8_t*>(&workItem), serializedworkItemDesc.data(), serializedworkItemDesc.size());

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&workItem);

    binDataSection.appendData(ptrCharTmp, getBinarySize(VPU::ArchKind::NPU40XX));
}

size_t vpux::NPUReg40XX::WorkItemOp::getBinarySize(VPU::ArchKind) {
    return sizeof(nn_public::VpuWorkItem);
}

size_t vpux::NPUReg40XX::WorkItemOp::getAlignmentRequirements(VPU::ArchKind) {
    // TODO: E#80148
    VPUX_THROW("WrappableInterface method should not be called at this point! E#80148");
}

std::optional<ELF::SectionSignature> vpux::NPUReg40XX::WorkItemOp::getSectionSignature() {
    // TODO: E#80148
    VPUX_THROW("WrappableInterface method should not be called at this point! E#80148");
}

bool vpux::NPUReg40XX::WorkItemOp::hasMemoryFootprint() {
    // TODO: E#80148
    VPUX_THROW("WrappableInterface method should not be called at this point! E#80148");
}

std::vector<ELF::RelocationInfo> vpux::NPUReg40XX::WorkItemOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    auto firstTaskOffset = offsetof(nn_public::VpuWorkItem, wi_desc_ptr);
    if (auto firstTask = getFirstTask()) {
        if (getTaskType() == VPURegMapped::TaskType::DMA) {
            relocs.emplace_back(firstTask, targetSection, firstTaskOffset, ELF::RelocationType::R_VPU_64,
                                ELF::getOffsetOfSymRef(symRefMap, firstTask), "First task (DMA) in work item reloc");
        } else {
            relocs.emplace_back(firstTask, targetSection, firstTaskOffset,
                                ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET,
                                ELF::getOffsetOfSymRef(symRefMap, firstTask), "First task in work item reloc");
        }
    }

    return relocs;
}
