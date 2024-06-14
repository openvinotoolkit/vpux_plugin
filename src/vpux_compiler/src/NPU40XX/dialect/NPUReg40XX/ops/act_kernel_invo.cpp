//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include <npu_40xx_nnrt.hpp>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;
using namespace npu40xx;

//
// ActKernelInvocationOp
//

void vpux::NPUReg40XX::ActKernelInvocationOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuActKernelInvocation actKernelInvo;

    auto actKernInvoDescriptor = getActInvoDescriptorAttr().getRegMapped();
    auto serializedActKernInvoDesc = actKernInvoDescriptor.serialize();
    memcpy(reinterpret_cast<uint8_t*>(&actKernelInvo), serializedActKernInvoDesc.data(),
           serializedActKernInvoDesc.size());

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelInvo);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::NPUReg40XX::ActKernelInvocationOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelInvocation);
}

size_t vpux::NPUReg40XX::ActKernelInvocationOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelInvocation);
}

vpux::ELF::SectionFlagsAttr vpux::NPUReg40XX::ActKernelInvocationOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

vpux::ELF::SectionFlagsAttr vpux::NPUReg40XX::ActKernelInvocationOp::getUserProcs() {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
}

std::optional<ELF::SectionSignature> vpux::NPUReg40XX::ActKernelInvocationOp::getSectionSignature() {
    return {};
}

bool vpux::NPUReg40XX::ActKernelInvocationOp::hasMemoryFootprint() {
    return true;
}

namespace {
size_t getSymRefOffsetForReloc(NPUReg40XX::ActKernelInvocationOp op, mlir::SymbolRefAttr ref) {
    constexpr auto ptrOffset = offsetof(nn_public::VpuPtr<void>, ptr);

    if (ref == op.getKernelRange()) {
        return offsetof(nn_public::VpuActKernelInvocation, range) + ptrOffset;
    } else if (ref == op.getKernelParams()) {
        return offsetof(nn_public::VpuActKernelInvocation, kernel_args) + ptrOffset;
    } else if (ref == op.getKernelData()) {
        return offsetof(nn_public::VpuActKernelInvocation, data_window_base) + ptrOffset;
    } else if (ref == op.getProfilingData()) {
        return offsetof(nn_public::VpuActKernelInvocation, perf_packet_out) + ptrOffset;
    }

    VPUX_THROW("Provided SymbolRefAttr is not linked to the ActKernelInvocation Op or getSymRefOffsetForReloc does not "
               "support "
               "it");
}
}  // namespace

std::vector<ELF::RelocationInfo> vpux::NPUReg40XX::ActKernelInvocationOp::getRelocationInfo(
        ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto thisInvo = *(this);
    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    relocs.push_back(ELF::RelocationInfo(
            getKernelRange(), targetSection, getSymRefOffsetForReloc(thisInvo, getKernelRange()),
            ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET, ELF::getOffsetOfSymRef(symRefMap, getKernelRange())));

    if (auto kernelData = getKernelData().value_or(nullptr)) {
        relocs.push_back(ELF::RelocationInfo(kernelData, targetSection, getSymRefOffsetForReloc(thisInvo, kernelData),
                                             ELF::RelocationType::R_VPU_64,
                                             ELF::getOffsetOfSymRef(symRefMap, kernelData)));
    }

    relocs.push_back(
            ELF::RelocationInfo(getKernelParams(), targetSection, getSymRefOffsetForReloc(thisInvo, getKernelParams()),
                                ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, getKernelParams())));

    if (auto profilingData = getProfilingData().value_or(nullptr)) {
        relocs.push_back(ELF::RelocationInfo(
                profilingData, targetSection, getSymRefOffsetForReloc(thisInvo, profilingData),
                ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET, ELF::getOffsetOfSymRef(symRefMap, profilingData)));
    }

    return relocs;
}
