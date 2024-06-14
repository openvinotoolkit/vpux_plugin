//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/mc_strategy_getter.hpp"

using namespace vpux::VPU::arch40xx;

//
// StrategyGetter
//

StrategyGetter::StrategyGetter(const int64_t numClusters): _numTiles(numClusters) {
    VPUX_THROW_WHEN(_numTiles <= 0, "Invalid number of clusters {0}", numClusters);
}

void StrategyGetter::getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const {
    arch37xx::StrategyGetter::getMCStrategies(strategies);
    strategies.push_back(MultiClusterStrategy::SplitOverWidth);

    // there is computation across 2 axes, so 3 tiles is minimum
    if (_numTiles > 2) {
        strategies.push_back(MultiClusterStrategy::SplitOverHeightKernel);
        strategies.push_back(MultiClusterStrategy::SplitOverHeightWidth);
    }
}
