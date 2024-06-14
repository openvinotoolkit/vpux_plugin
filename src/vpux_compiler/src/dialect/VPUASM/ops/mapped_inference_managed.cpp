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
// ManagedMappedInferenceOp
//

void vpux::VPUASM::ManagedMappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for ActKernelRangeOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUASM::ManagedMappedInferenceOp::getBinarySize() {
    return sizeof(nn_public::VpuManagedMappedInference);
}

size_t vpux::VPUASM::ManagedMappedInferenceOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuManagedMappedInference);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ManagedMappedInferenceOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ManagedMappedInferenceOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ManagedMappedInferenceOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "mapped_inference"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ManagedMappedInferenceOp::hasMemoryFootprint() {
    return true;
}

namespace {
size_t getSymRefOffsetForReloc(VPUASM::ManagedMappedInferenceOp op, mlir::SymbolRefAttr ref) {
    if (ref == op.getBarrierTasksAttr()) {
        return offsetof(nn_public::VpuManagedMappedInference, task_configs) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuTaskBarrierMap>, address);
    }

    if (ref == op.getWorkItemsAttr()) {
        return offsetof(nn_public::VpuManagedMappedInference, work_items) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuWorkItem>, address);
    }

    if (ref == op.getBootstrapTasksAttr()) {
        return offsetof(nn_public::VpuManagedMappedInference, initial_barriers) +
               offsetof(nn_public::VpuTaskReference<uint32_t>, address);
    }

    VPUX_THROW(
            "Provided SymbolRefAttr is not linked to the ManagedMappedInference Op or getSymRefOffsetForReloc does not "
            "support it");
}
}  // namespace

std::vector<ELF::RelocationInfo> VPUASM::ManagedMappedInferenceOp::getRelocationInfo(
        ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto thisMMI = *(this);
    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    if (auto workItemTasks = getWorkItems().value_or(nullptr)) {
        relocs.push_back(
                ELF::RelocationInfo(workItemTasks, targetSection, getSymRefOffsetForReloc(thisMMI, workItemTasks),
                                    ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, workItemTasks)));
    }

    if (auto barriersTasks = getBarrierTasks().value_or(nullptr)) {
        relocs.push_back(
                ELF::RelocationInfo(barriersTasks, targetSection, getSymRefOffsetForReloc(thisMMI, barriersTasks),
                                    ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, barriersTasks)));
    }

    if (auto bootstrapTasks = getBootstrapTasks().value_or(nullptr)) {
        relocs.push_back(
                ELF::RelocationInfo(bootstrapTasks, targetSection, getSymRefOffsetForReloc(thisMMI, bootstrapTasks),
                                    ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, bootstrapTasks)));
    }

    return relocs;
}
