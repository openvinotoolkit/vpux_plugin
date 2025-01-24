///
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/utils/convert_to_dma_utils.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

size_t VPU::getGatherDMAMaxIndicesListLength(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU40XX: {
        return VPUIP::arch40xx::DMA_MAX_INDICES_LIST_LENGTH;
    }
    case VPU::ArchKind::UNKNOWN:
    case VPU::ArchKind::NPU37XX: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    default: {
        return VPUIP::arch40xx::DMA_MAX_INDICES_LIST_LENGTH;
    }
    }
};

size_t VPU::getGatherDMAMaxElementSize(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU40XX: {
        return VPUIP::arch40xx::GATHER_DMA_MAX_ELEMENT_SIZE;
    }
    case VPU::ArchKind::UNKNOWN:
    case VPU::ArchKind::NPU37XX: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    default: {
        return VPUIP::arch40xx::GATHER_DMA_MAX_ELEMENT_SIZE;
    }
    }
};
