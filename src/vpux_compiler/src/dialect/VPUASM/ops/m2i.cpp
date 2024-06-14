//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/utils/core/optional.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// M2IOp
//

void VPUASM::M2IOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for VPUASM::M2IOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage ", getOperationName());
#endif
}

size_t VPUASM::M2IOp::getBinarySize() {
    return sizeof(nn_public::VpuMediaTask);
}

size_t VPUASM::M2IOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuMediaTask);
}

ELF::SectionFlagsAttr VPUASM::M2IOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

ELF::SectionFlagsAttr VPUASM::M2IOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> VPUASM::M2IOp::getSectionSignature() {
    return ELF::SectionSignature(ELF::generateSignature("task", "m2i", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool VPUASM::M2IOp::hasMemoryFootprint() {
    return true;
}
