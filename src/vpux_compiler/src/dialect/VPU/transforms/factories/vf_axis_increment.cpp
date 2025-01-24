//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/vf_axis_increment.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_axis_increment.hpp"

using namespace vpux::VPU;

std::unique_ptr<IVFAxisIncrement> vpux::VPU::getVFAxisIncrement(Dim axis) {
    if (axis == Dims4D::Act::C) {
        return std::make_unique<ChannelsVFAxisIncrement>();
    }

    return std::make_unique<DefaultVFAxisIncrement>();
}
