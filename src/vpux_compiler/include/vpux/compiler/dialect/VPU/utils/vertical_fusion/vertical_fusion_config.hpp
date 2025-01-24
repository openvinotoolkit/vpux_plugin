//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

namespace vpux {
namespace VPU {

// the length of VF pipelining pattern
// should match the pattern DPU-SW-DPU for now
// E#95184
constexpr int64_t VF_PIPELINE_LENGTH = 3;

// structure to incapsulate all necessary objects for VF subgraph
class VFConfig {
public:
    VFConfig(VPU::VerticalFusionOp vfOp, bool enableVFPipelining = true);

    // get original subgraph
    VPU::VerticalFusionOp getSubgraph() const;

    // get the largest operation in the subgraph
    mlir::Operation* getLargestOp();

    // get all inputs
    const SmallVector<mlir::Operation*>& getInputs();

    // get all outputs
    const SmallVector<mlir::Operation*>& getOutputs();

    // get all oeprations in the subgraph
    const SmallVector<mlir::Operation*>& getVFOperations();

    // get all oeprations in the subgraph
    SmallVector<mlir::Operation*> getOperationsForTiling();

    // check if subgraph might be pipelined
    bool isPipelined() const;

    // WA before extending pipelining case
    // Track [E#95184]
    bool isPotentiallyPipelined();

    // Reset cached data
    void invalidatePointers();

    // Get cached types for operation in VF
    SmallVector<NDTypeInterface> getOperationTypes(mlir::Operation* operation, const TileInfo& outTile,
                                                   const ArrayRef<TileInfo> inputTiles);
    SmallVector<NDTypeInterface> getOperationTypes(mlir::Operation* operation);

private:
    bool isVFPipelinePattern();

    VPU::VerticalFusionOp _subgraph;
    mlir::Operation* _largestOp = nullptr;
    SmallVector<mlir::Operation*> _inputOps;
    SmallVector<mlir::Operation*> _outputOps;
    SmallVector<mlir::Operation*> _vfOps;
    bool _isVFPipelineCandidate = false;
    bool _isPipelineEnabled = false;

    DenseMap<mlir::Operation*, std::map<Shape, SmallVector<NDTypeInterface>>> _tilesCache;
};

}  // namespace VPU
}  // namespace vpux
