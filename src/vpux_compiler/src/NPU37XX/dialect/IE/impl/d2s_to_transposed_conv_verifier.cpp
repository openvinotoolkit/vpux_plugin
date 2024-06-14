//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux::IE::arch37xx;

//
// D2SToTransposedConvVerifier
//

// Larger block_size and fewer number of output channels make DPU solution less efficient, see E#113159
bool D2SToTransposedConvVerifier::isBeneficialConversion(IE::DepthToSpaceOp d2sOp) const {
    const auto blockSize = d2sOp.getBlockSize();
    if (blockSize >= 4) {
        return false;
    }

    auto outputType = d2sOp.getOutput().getType().cast<NDTypeInterface>();
    auto outputShape = outputType.getShape();
    auto outputChannels = outputShape[Dims4D::Act::C];
    auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());

    return outputChannels >= alignment;
}
