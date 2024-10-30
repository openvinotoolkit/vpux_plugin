//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ActKernelRangeOp
//

void vpux::VPUASM::ActKernelRangeOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for ActKernelRangeOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUASM::ActKernelRangeOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelRange);
}

size_t vpux::VPUASM::ActKernelRangeOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelRange);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ActKernelRangeOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ActKernelRangeOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "shave", "range", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ActKernelRangeOp::hasMemoryFootprint() {
    return true;
}
