//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/mc_strategy_getter.hpp"

using namespace vpux::VPU;

void StrategyGetterBase::getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const {
    strategies.push_back(MultiClusterStrategy::Clustering);
}
