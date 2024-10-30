//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ActShaveRtOp
//

void vpux::VPUASM::ActShaveRtOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for ActShaveRTOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUASM::ActShaveRtOp::getBinarySize() {
    return sizeof(nn_public::VpuNNShaveRuntimeConfigs);
}

// The management kernel code must be 1kB aligned as an ActShave requirement
size_t vpux::VPUASM::ActShaveRtOp::getAlignmentRequirements() {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ActShaveRtOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ActShaveRtOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "runtime"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ActShaveRtOp::hasMemoryFootprint() {
    return true;
}
