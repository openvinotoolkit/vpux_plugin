//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <algorithm>
#include <cstdint>

namespace vpux {
namespace VPUIP {

enum class UnrollDMAAnalysisNeeded {
    UnrollDepthToSpaceDMAPass,
    UnrollSpaceToDepthDMAPass,
    UnrollUpsamplingDMAPass,
    UnrollPermuteToNNDMAPass,
    UnrollExpandDMAPass,
    UnrollPerAxisTileDMAPass,
    NumberOfAnalyzedPasses
};

class UnrollDMAAnalysis {
public:
    using StorageType = std::array<uint8_t, static_cast<size_t>(UnrollDMAAnalysisNeeded::NumberOfAnalyzedPasses)>;
    UnrollDMAAnalysis(mlir::Operation* operation);

    bool passNeeded(UnrollDMAAnalysisNeeded passTag);

private:
    mlir::Operation* _operation;
    StorageType _lookupArray{};
};

}  // namespace VPUIP
}  // namespace vpux
