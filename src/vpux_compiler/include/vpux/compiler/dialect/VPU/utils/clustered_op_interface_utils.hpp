//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"

#include "vpux/utils/core/mem_size.hpp"

namespace vpux {
namespace VPU {

constexpr int64_t SINGLE_BATCH = 1;
constexpr size_t RANK_REQUIRED_FOR_TILING = 4;

int64_t getNumTiles(mlir::Operation* op);

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool isOperationSplitOverHeightCompatible(mlir::Operation* op, const vpux::TileInfo& outputTile);

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOW
// compatible it must have an output width of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output Width must be a minimum of 4.
bool isOperationSplitOverWidthCompatible(mlir::Operation* op, ShapeRef outputShape, ShapeRef offset, ShapeRef axis);

/// Each cluster should compute at least 16 output channels. Therefore in order for a layer to be SOK
/// compatible it must have an output channel of at least the number of clusters x 16
/// specified for compilation.
/// For example for 4 cluster compilation the output channel must be a
/// minimum of 4x16=64.
/// @warning Considering SOK can use 2/3 clusters to avoid per cluster channel alignment, like
/// OC = 64, [32, 32] output channels per cluster is valid too.
/// Thus the conditions can be relaxed.
bool isOperationSplitOverKernelCompatible(mlir::Operation* op, ShapeRef outputShape, ShapeRef offset, ShapeRef axis);

// Each cluster should compute at most one output batch. Therefore, in order for a layer to be SOB compatible it must
// have an output batch dimension of at most the number of clusters specified for compilation.
// For example for 4 cluster compilation the output batch must be a maximum of 4.
bool isOperationSplitOverBatchCompatible(mlir::Operation* op, ShapeRef outputShape);

// Each cluster should compute at least one output group. Therefore, in order for a layer to be SOG compatible it must
// have output/input groups of at least the number of clusters specified for compilation.
// For example for 4 cluster compilation the input/output groups number must be a minimum of 4.
bool isOperationSplitOverGroupCompatible(mlir::Operation* op, const vpux::TileInfo& outputTile);

bool checkMCRestrictions(mlir::Operation*);

bool doesLayerFitIntoCMX(mlir::Operation* op, VPU::MultiClusterStrategy strategy, SiblingOpsAnalysis& siblingsAnalysis,
                         Byte reservedMem);

}  // namespace VPU
}  // namespace vpux
