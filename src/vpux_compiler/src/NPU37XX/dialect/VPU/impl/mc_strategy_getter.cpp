//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/mc_strategy_getter.hpp"

using namespace vpux::VPU::arch37xx;

//
// StrategyGetter
//

void StrategyGetter::getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const {
    StrategyGetterBase::getMCStrategies(strategies);
    strategies.push_back(MultiClusterStrategy::SplitOverHeight);
    strategies.push_back(MultiClusterStrategy::SplitOverKernel);
    strategies.push_back(MultiClusterStrategy::HKSwitch);
    strategies.push_back(MultiClusterStrategy::SplitOverHeightOverlapped);
}
