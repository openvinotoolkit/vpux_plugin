//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/max_lstm_hidden_size_constant.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/max_lstm_hidden_size_constant.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

constexpr int64_t maxLstmHiddenSizeConstant = 0;

int64_t VPU::getMaxLstmHiddenSizeConstant(VPU::ArchKind arch, bool sequenceEnabled) {
    switch (arch) {
    case VPU::ArchKind::NPU40XX: {
        return VPU::arch40xx::getMaxLstmHiddenSizeConstant(sequenceEnabled);
    }
    case VPU::ArchKind::NPU37XX: {
        return maxLstmHiddenSizeConstant;
    }
    case VPU::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}
