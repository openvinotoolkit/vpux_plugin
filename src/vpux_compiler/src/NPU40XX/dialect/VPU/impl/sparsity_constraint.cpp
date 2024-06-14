//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux::VPU::arch40xx;

// In order for a channel size to be compatible with being the storage element size, it must be in the allowed limits
// [16-8192] and must be aligned to 16.
bool SparsityConstraint::areChannelsFitForSESize(int64_t channels) const {
    auto channelsInRange =
            channels >= VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT && channels <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    auto channelsAligned = (channels % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) == 0;
    return channelsInRange && channelsAligned;
}

// E#102555: IDU errata, incorrect SE pointers for BF16/FP16 (DENSE_SE=1, IC=8K; SE size=4K)
// As this only applies for DENSE_SE=1, the method is meant to be used only for activation sparsity
bool SparsityConstraint::areChannelsFitForSESize(mlir::Type inputType, int64_t channels) const {
    const auto ndType = inputType.cast<vpux::NDTypeInterface>();
    const auto inputElemType = ndType.getElementType();
    const auto inputShape = ndType.getShape();
    VPUX_THROW_WHEN(inputShape.size() != 4, "Expected 4D input, got {0}D", inputShape.size());
    const auto inputChannels = inputShape[Dims4D::Act::C];

    auto floatInput = inputElemType.isa<mlir::Float16Type>() || inputElemType.isa<mlir::BFloat16Type>();
    if (floatInput && inputChannels == VPU::NCEInvariant::VPU_DIMENSION_LIMIT && channels == 4096) {
        return false;
    }

    return areChannelsFitForSESize(channels);
}
