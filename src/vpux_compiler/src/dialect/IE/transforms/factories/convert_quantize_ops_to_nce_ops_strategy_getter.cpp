//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/convert_quantize_ops_to_nce_ops_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/VPU30XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"

using namespace vpux;

namespace vpux::IE {

std::unique_ptr<IConvertQuantizeOpsToNceOpsStrategy> createConvertQuantizeOpsToNceOpsStrategy(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU30XX:
        return std::make_unique<IE::arch30xx::ConvertQuantizeOpsToNceOpsStrategy>();
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX:
        return std::make_unique<IE::arch37xx::ConvertQuantizeOpsToNceOpsStrategy>();
    default:
        VPUX_THROW("Arch '{0}' is not supported", arch);
    }
}

}  // namespace vpux::IE
