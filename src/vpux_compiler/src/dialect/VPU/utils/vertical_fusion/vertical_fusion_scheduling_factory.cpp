//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/full_prefetch_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/minimal_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/pipelining_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/prefetch_lastop_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/weights_prefetch_vf_scheduling.hpp"

using namespace vpux;
using namespace VPU;

VFSchedulingFactory::VFSchedulingFactory(bool prefetching): _prefetching(prefetching) {
}

std::shared_ptr<IVFScheduling> VFSchedulingFactory::createVFScenario(VFScenario scenarioCode, Logger log) const {
    switch (scenarioCode) {
    case VFScenario::MINIMAL: {
        return std::make_shared<MinimalRequirementsVFScheduling>(log, _prefetching);
    }
    case VFScenario::LASTOP_PREFETCHING: {
        return std::make_shared<PrefetchingLastOpVFScheduling>(log, _prefetching);
    }
    case VFScenario::WEIGHTS_PREFETCHING: {
        return std::make_shared<WeightsPrefetchingVFScheduling>(log, _prefetching);
    }
    case VFScenario::FULL_PREFETCHING: {
        return std::make_shared<FullPrefetchingVFScheduling>(log, _prefetching);
    }
    case VFScenario::VF_PIPELINING: {
        return std::make_shared<PipeliningVFScheduling>(log, _prefetching);
    }
    default: {
        VPUX_THROW("No scheduling implemented for {0}", scenarioCode);
    }
    }
}
