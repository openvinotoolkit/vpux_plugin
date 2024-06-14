//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/sparsity_constraint.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPU/impl/sparsity_constraint.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::SparsityConstraint VPU::getSparsityConstraint(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU30XX:
    case VPU::ArchKind::NPU37XX: {
        return VPU::arch30xx::SparsityConstraint{};
    }
    case VPU::ArchKind::NPU40XX: {
        return VPU::arch40xx::SparsityConstraint{};
    }
    case VPU::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}
