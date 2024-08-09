//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/interfaces/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"

namespace vpux {
namespace IE {

//
// D2SToTransposedConvVerifierBase
//

bool D2SToTransposedConvVerifierBase::isBeneficialConversion(IE::DepthToSpaceOp) const {
    return true;
}

std::unique_ptr<D2SToTransposedConvVerifierBase> createD2SToTransposedConvVerifier(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX: {
        return std::make_unique<IE::arch37xx::D2SToTransposedConvVerifier>();
    }
    case VPU::ArchKind::NPU40XX: {
        return std::make_unique<IE::arch40xx::D2SToTransposedConvVerifier>();
    }
    case VPU::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}

}  // namespace IE
}  // namespace vpux
