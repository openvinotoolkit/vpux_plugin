//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

namespace vpux {
namespace VPU {

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

    // check if subgraph might be pipelined
    bool isPipelined() const;

    // disable VF pipeline
    void disableVFPipeline();

    // restore VF pipeline
    void restoreVFPipeline();

    // WA before extending pipelining case
    // Track [E#95184]
    bool isPotentiallyPipelined() const;

private:
    bool isVFPipelinePattern();

    VPU::VerticalFusionOp _subgraph;
    mlir::Operation* _largestOp = nullptr;
    SmallVector<mlir::Operation*> _inputOps;
    SmallVector<mlir::Operation*> _outputOps;
    SmallVector<mlir::Operation*> _vfOps;
    bool _isVFPipelineCandidate = false;
    bool _isPipelineEnabled = false;
};

}  // namespace VPU
}  // namespace vpux
