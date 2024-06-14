//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/nce_sparsity_converters.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPU/impl/nce_sparsity_converters.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::NCESparsity::PPEConverterCb VPU::NCESparsity::getPPEConverterCb(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU30XX: {
        return VPU::arch30xx::getScale;
    }
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX: {
        return VPU::arch37xx::getScale;
    }
    case VPU::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}

VPU::NCESparsity::BiasConverterCb VPU::NCESparsity::getBiasConverterCb(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU30XX: {
        return VPU::arch30xx::getBias;
    }
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX: {
        return VPU::arch37xx::getBias;
    }
    case VPU::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}
