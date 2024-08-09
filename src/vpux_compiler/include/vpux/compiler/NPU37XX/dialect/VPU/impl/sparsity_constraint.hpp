//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/sparsity_constraint.hpp"

namespace vpux::VPU::arch37xx {

struct SparsityConstraint final {
    bool areChannelsFitForSESize(int64_t channels) const;
    bool areChannelsFitForSESize(mlir::Type inputType, int64_t channels) const;
};

}  // namespace vpux::VPU::arch37xx
