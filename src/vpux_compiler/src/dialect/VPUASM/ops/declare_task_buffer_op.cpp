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
// DeclareTaskBufferOp
//

void VPUASM::DeclareTaskBufferOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    return;
}

size_t VPUASM::DeclareTaskBufferOp::getBinarySize() {
    switch (getTaskType()) {
    case VPURegMapped::TaskType::DMA:
        return sizeof(nn_public::VpuDMATask);
    case VPURegMapped::TaskType::ActKernelInvocation:
        return sizeof(nn_public::VpuActKernelInvocation);
    case VPURegMapped::TaskType::ActKernelRange:
        return sizeof(nn_public::VpuActKernelRange);
    case VPURegMapped::TaskType::DPUInvariant:
        return sizeof(nn_public::VpuDPUInvariant);
    case VPURegMapped::TaskType::DPUVariant:
        return sizeof(nn_public::VpuDPUVariant);
    case VPURegMapped::TaskType::M2I:
        return sizeof(nn_public::VpuMediaTask);
    default:
        VPUX_THROW("Invalid task type for DeclareTaskBufferOp {0}", *this);
        return 0;
    }
}

size_t VPUASM::DeclareTaskBufferOp::getAlignmentRequirements() {
    switch (getTaskType()) {
    case VPURegMapped::TaskType::DMA:
        return alignof(nn_public::VpuDMATask);
    case VPURegMapped::TaskType::ActKernelInvocation:
        return alignof(nn_public::VpuActKernelInvocation);
    case VPURegMapped::TaskType::ActKernelRange:
        return alignof(nn_public::VpuActKernelRange);
    case VPURegMapped::TaskType::DPUInvariant:
        return alignof(nn_public::VpuDPUInvariant);
    case VPURegMapped::TaskType::DPUVariant:
        return alignof(nn_public::VpuDPUVariant);
    case VPURegMapped::TaskType::M2I:
        return sizeof(nn_public::VpuMediaTask);
    default:
        VPUX_THROW("Invalid task type for DeclareTaskBufferOp {0}", *this);
        return 0;
    }
}

ELF::SectionFlagsAttr VPUASM::DeclareTaskBufferOp::getAccessingProcs(mlir::SymbolUserMap&) {
    // TaskBuffers represent CMX virtual entities, whose allocation is not controlled
    return ELF::SectionFlagsAttr::SHF_NONE;
}

ELF::SectionFlagsAttr VPUASM::DeclareTaskBufferOp::getUserProcs() {
    // TaskBuffers represent CMX virtual entities, whose allocation is not controlled
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DeclareTaskBufferOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "metadata", "cmx"),
                                 ELF::SectionFlagsAttr::SHF_NONE, ELF::SectionTypeAttr::VPU_SHT_CMX_METADATA);
}

bool vpux::VPUASM::DeclareTaskBufferOp::hasMemoryFootprint() {
    return false;
}
