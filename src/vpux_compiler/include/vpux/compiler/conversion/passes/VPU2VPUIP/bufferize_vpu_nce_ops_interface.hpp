//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>

namespace vpux {

// E#112397: remove this once VPU.NCE.ClusterTiling is removed
// Runs a Multi-Tile NCE Permute lowering using one-shot bufferization approach.
// This is effectively an overload of lowerMultiTileVpuNcePermute.
mlir::LogicalResult lowerMultiTileVpuNcePermuteOneShot(mlir::MLIRContext* ctx, mlir::Operation* func,
                                                       vpux::Logger& log);

}  // namespace vpux
