//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/string_ref.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <optional>

namespace vpux {

constexpr StringLiteral multiClusterStrategy = "multiClusterStrategy";  // only be used for manual strategy utils
constexpr StringLiteral tilingStrategy = "tilingStrategy";
constexpr StringLiteral defaultNoValue = "NONE";
constexpr StringLiteral verticalFusion = "verticalFusion";  // only be used for manual strategy utils
constexpr StringLiteral verticalFusionHash = "verticalFusionHash";
constexpr StringLiteral layerTypeName = "layerType";
constexpr StringLiteral updatedVFTiling = "updatedVFTiling";
constexpr StringLiteral outputPipelining = "outputPipelining";

std::optional<mlir::DenseMap<size_t, std::optional<VPU::MultiClusterStrategy>>> maybeGetStrategyFor(
        StringRef modelHash);

bool isStrategyPreConfigured(StringRef modelHash);

// Represents op2hash and reverse mapping for function layers
struct HashStageResult {
    bool succeed;
    mlir::DenseMap<mlir::Operation*, size_t> localizedHashes;
    mlir::DenseMap<size_t, mlir::Operation*> reverseMapping;
};

// Generate hashes for executable layers such as DPU and SW. These hashes are reliable, independ from IR order and don't
// change between runs. Hashes are build from layer name&type, specified by OV, output shape and hashes of neighbors.
// This functions attempts to do several rounds of hash rounds, each one increase neighborhood used for hashing. If
// hashing didn't succeed(algorithm detected collisions for function layers) succeed field of result will be false
HashStageResult hashFunctionLayers(mlir::func::FuncOp funcOp);

bool loadPreConfiguredStrategy(vpux::Logger log, mlir::func::FuncOp func, StringRef modelHash);

}  // namespace vpux
