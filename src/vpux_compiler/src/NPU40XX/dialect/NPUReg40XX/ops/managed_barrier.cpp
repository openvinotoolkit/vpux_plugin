//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

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
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t NPUReg40XX::ManagedBarrierOp::getBinarySize() {
    return sizeof(nn_public::VpuTaskBarrierMap);
}
