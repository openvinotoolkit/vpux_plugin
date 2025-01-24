//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"

namespace vpux {
namespace VPU {

/*
  Factory which creates VF scheduling scenario
*/
class VFSchedulingFactory {
public:
    VFSchedulingFactory(bool prefetching);

    /*
      create scheduling scenario
    */
    std::shared_ptr<IVFScheduling> createVFScenario(VFScenario scenarioCode, Logger log) const;

private:
    bool _prefetching = true;
};

}  // namespace VPU
}  // namespace vpux
