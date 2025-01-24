//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ManagedBarrierOp
//

void NPUReg40XX::ManagedBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuTaskBarrierMap barrier = {};

    auto barrierDescriptor = getBarrierDescriptorAttr().getRegMapped();
    auto serializedBarrierDesc = barrierDescriptor.serialize();
    memcpy(reinterpret_cast<uint8_t*>(&barrier), serializedBarrierDesc.data(), serializedBarrierDesc.size());

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&barrier);
    binDataSection.appendData(ptrCharTmp, getBinarySize(VPU::ArchKind::NPU40XX));
}

size_t NPUReg40XX::ManagedBarrierOp::getBinarySize(VPU::ArchKind) {
    return sizeof(nn_public::VpuTaskBarrierMap);
}

size_t vpux::NPUReg40XX::ManagedBarrierOp::getAlignmentRequirements(VPU::ArchKind) {
    // TODO: E#80148
    VPUX_THROW("WrappableInterface method should not be called at this point! E#80148");
}

std::optional<ELF::SectionSignature> vpux::NPUReg40XX::ManagedBarrierOp::getSectionSignature() {
    // TODO: E#80148
    VPUX_THROW("WrappableInterface method should not be called at this point! E#80148");
}

bool vpux::NPUReg40XX::ManagedBarrierOp::hasMemoryFootprint() {
    // TODO: E#80148
    VPUX_THROW("WrappableInterface method should not be called at this point! E#80148");
}
