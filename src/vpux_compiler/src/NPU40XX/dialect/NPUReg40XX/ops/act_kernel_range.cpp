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
// ActKernelRangeOp
//

void vpux::NPUReg40XX::ActKernelRangeOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuActKernelRange actKernelRange;

    auto actKernRangeDescriptor = getActRangeDescriptorAttr().getRegMapped();
    auto serializedActKernRangeDesc = actKernRangeDescriptor.serialize();
    memcpy(reinterpret_cast<uint8_t*>(&actKernelRange), serializedActKernRangeDesc.data(),
           serializedActKernRangeDesc.size());

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelRange);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::NPUReg40XX::ActKernelRangeOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelRange);
}

size_t vpux::NPUReg40XX::ActKernelRangeOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelRange);
}

std::optional<ELF::SectionSignature> vpux::NPUReg40XX::ActKernelRangeOp::getSectionSignature() {
    return {};
}

bool vpux::NPUReg40XX::ActKernelRangeOp::hasMemoryFootprint() {
    return true;
}

std::vector<ELF::RelocationInfo> vpux::NPUReg40XX::ActKernelRangeOp::getRelocationInfo(
        ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    if (auto kernelText = getKernelText().value_or(nullptr)) {
        relocs.push_back(ELF::RelocationInfo(
                kernelText, targetSection,
                offsetof(nn_public::VpuActKernelRange, text_window_base) + offsetof(nn_public::VpuPtr<void>, ptr),
                ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, kernelText),
                "Kernel text (ptr in text_window_base) for act kernel range reloc"));
    }
    return relocs;
}
