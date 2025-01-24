//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;

//
// WorkItemOp
//

void vpux::VPUASM::WorkItemOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for work Item
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUASM::WorkItemOp::getBinarySize(VPU::ArchKind /*arch*/) {
    return sizeof(npu40xx::nn_public::VpuWorkItem);
}

size_t vpux::VPUASM::WorkItemOp::getAlignmentRequirements(VPU::ArchKind /*arch*/) {
    return alignof(npu40xx::nn_public::VpuWorkItem);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::WorkItemOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::WorkItemOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "workItem"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::WorkItemOp::hasMemoryFootprint() {
    return true;
}
