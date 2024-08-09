//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux::IE::arch40xx;

//
// D2SToTransposedConvVerifier
//

// If not converted to TransposedConv, D2S is implemented by DMA on NPU40XX, it's
// more efficient to convert to TransposedConv if with relatively
// a big output channel (it's 8 from experiment result), see E#125463
constexpr int64_t BENEFICIAL_OUTPUT_CHANNEL_NUM = 8;
bool D2SToTransposedConvVerifier::isBeneficialConversion(IE::DepthToSpaceOp d2sOp) const {
    auto outputType = d2sOp.getOutput().getType().cast<NDTypeInterface>();
    auto outputShape = outputType.getShape();
    auto outputChannels = outputShape[Dims4D::Act::C];

    return outputChannels >= BENEFICIAL_OUTPUT_CHANNEL_NUM;
}
