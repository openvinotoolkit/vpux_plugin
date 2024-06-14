//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"

namespace vpux {
namespace VPU {

// container of pipelined operations
// there cannot be two operations with same type in the container

class VFPipelineContainer {
public:
    VFPipelineContainer(){};

    // add the first operation
    VFPipelineContainer(mlir::Operation* operation, const VPUNNCostParameters& tilingInfo);

    // check if there are operations with same characteristics are already there
    bool hasOperType(mlir::Operation* operation) const;

    // add new operation to the container to be pipelined with current ones
    void addOperation(mlir::Operation* operation, const VPUNNCostParameters& tilingInfo);

    // get max cost of operations from the container
    StrategyCost maxCost(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const;

    // compare two containers
    bool operator<(const VFPipelineContainer& o) const;

private:
    llvm::DenseMap<mlir::Operation*, VPUNNCostParameters> _containerMapper;
};

}  // namespace VPU
}  // namespace vpux
