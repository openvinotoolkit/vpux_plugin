//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_pipeline_container.hpp"

using namespace vpux;
using namespace VPU;

VFPipelineContainer::VFPipelineContainer(mlir::Operation* operation, const VPUNNCostParameters& tilingInfo) {
    addOperation(operation, tilingInfo);
}

void VFPipelineContainer::addOperation(mlir::Operation* operation, const VPUNNCostParameters& tilingInfo) {
    VPUX_THROW_WHEN(_containerMapper.contains(operation), "Operation {0} has already been added", operation->getLoc());
    _containerMapper.try_emplace(operation, tilingInfo);
}

StrategyCost VFPipelineContainer::maxCost(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    StrategyCost swCost = 0;
    StrategyCost dpuCost = 0;

    for (auto item : _containerMapper) {
        auto operCost = costFunction->getStrategyCost(item.first, item.second);
        if (mlir::isa<VPU::SWOpInterface>(item.first)) {
            swCost += operCost;
        } else {
            dpuCost += operCost;
        }
    }

    return std::max(swCost, dpuCost);
}

bool VFPipelineContainer::operator<(const VFPipelineContainer& o) const {
    return _containerMapper.size() < o._containerMapper.size();
}
