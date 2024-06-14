//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/sparsity_constraint.hpp"

using namespace vpux::VPU;

bool SparsityConstraint::areChannelsFitForSESize(int64_t channels) const {
    return self->areChannelsFitForSESize(channels);
}

bool SparsityConstraint::areChannelsFitForSESize(mlir::Type inputType, int64_t channels) const {
    return self->areChannelsFitForSESize(inputType, channels);
}
