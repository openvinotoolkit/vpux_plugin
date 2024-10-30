//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/mc_strategy_getter.hpp"

namespace vpux::VPU::arch40xx {

/*
   Class for getting strategies for VPU40XX
*/

class StrategyGetter : public arch37xx::StrategyGetter {
public:
    StrategyGetter(const int64_t numClusters);

    void getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const override;

private:
    int64_t _numTiles;
};

}  // namespace vpux::VPU::arch40xx
