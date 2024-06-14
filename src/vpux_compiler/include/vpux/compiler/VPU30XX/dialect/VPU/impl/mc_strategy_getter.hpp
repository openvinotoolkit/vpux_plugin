//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/mc_strategy_getter.hpp"

namespace vpux::VPU::arch30xx {

/*
   Class for getting strategies for NPU30XX
*/

class StrategyGetter : public StrategyGetterBase {
public:
    void getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const override;
};

}  // namespace vpux::VPU::arch30xx
