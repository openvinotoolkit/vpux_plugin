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

bool VFPipelineContainer::hasOperType(mlir::Operation* operation) const {
    if (_containerMapper.empty()) {
        return false;
    }

    const auto isSWOpKernel = [&](auto* oper) -> bool {
        return mlir::isa<VPU::SWOpInterface>(oper);
    };

    bool isOperSW = isSWOpKernel(operation);

    auto sameTypeFound = llvm::find_if(_containerMapper, [&](const auto& item) {
        return isSWOpKernel(item.first) == isOperSW;
    });

    return sameTypeFound != _containerMapper.end();
}

void VFPipelineContainer::addOperation(mlir::Operation* operation, const VPUNNCostParameters& tilingInfo) {
    VPUX_THROW_WHEN(hasOperType(operation), "Operation with same operation type is already in the container");
    _containerMapper.try_emplace(operation, tilingInfo);
}

StrategyCost VFPipelineContainer::maxCost(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    StrategyCost maxCost = 0;

    for (auto item : _containerMapper) {
        maxCost = std::max(maxCost, costFunction->getStrategyCost(item.first, item.second));
    }

    return maxCost;
}

bool VFPipelineContainer::operator<(const VFPipelineContainer& o) const {
    return _containerMapper.size() < o._containerMapper.size();
}
