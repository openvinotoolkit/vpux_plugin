//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/factories/convert_layers_to_vpu_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/conversion/impl/convert_layers_to_vpu_strategy.hpp"
#include "vpux/compiler/VPU30XX/conversion/impl/convert_layers_to_vpu_strategy.hpp"

namespace vpux {

std::unique_ptr<IGreedilyPassStrategy> CreateConvertLayers2VPUStrategy(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU30XX:
        return std::make_unique<arch30xx::ConvertLayers2VPUStrategy>();
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX:
        return std::make_unique<arch37xx::ConvertLayers2VPUStrategy>();
    default:
        VPUX_THROW("Arch '{0}' is not supported", arch);
    }
}

}  // namespace vpux
