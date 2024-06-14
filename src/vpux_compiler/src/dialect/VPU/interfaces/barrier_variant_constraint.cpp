//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/barrier_variant_constraint.hpp"

using namespace vpux::VPU;

size_t PerBarrierVariantConstraint::getPerBarrierMaxVariantSum() const {
    return self->getPerBarrierMaxVariantSum();
}

size_t PerBarrierVariantConstraint::getPerBarrierMaxVariantCount() const {
    return self->getPerBarrierMaxVariantCount();
}
