//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ActKernelInvocationOp
//

void vpux::VPUASM::ActKernelInvocationOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for ActKernelRangeOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUASM::ActKernelInvocationOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelInvocation);
}

size_t vpux::VPUASM::ActKernelInvocationOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelInvocation);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ActKernelInvocationOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ActKernelInvocationOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "shave", "invocation", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ActKernelInvocationOp::hasMemoryFootprint() {
    return true;
}
