//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {

// A subset of the IR intended to be extracted into a function. It contains a list of operations in topological order
struct IRSlice {
    SmallVector<mlir::Value> inputs;
    SmallVector<mlir::Value> outputs;
    std::vector<mlir::Operation*> operations;
};

// A vector of IR slices which should be outlined with the same function. This means all of these instances should be
// identical in terms of operations, attributes and types - only the data may be different (activations and constants,
// if allowed). Can have only one element if the block is not repeating
using OutliningInstance = SmallVector<IRSlice>;

//
// IFunctionOutliner
//

class IFunctionOutliner {
public:
    virtual ~IFunctionOutliner() = default;

    virtual SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) = 0;
};

//
// FunctionOutlinerNaive
//

class FunctionOutlinerNaive final : IFunctionOutliner {
public:
    FunctionOutlinerNaive(size_t numSplits, Logger log);

    // Returns a list of targets for function outlining
    // In case the intention is to split the IR into separate individual functions, each OutliningInstance will have one
    // element
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    size_t _numSplits;
    Logger _log;
};

//
// FunctionOutlinerRepeatingBlocks
//

class FunctionOutlinerRepeatingBlocks final : IFunctionOutliner {
public:
    FunctionOutlinerRepeatingBlocks(size_t minOpsInBlock, size_t maxNumIterations, Logger log);

    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    size_t _minOpsInBlock;
    size_t _maxNumIterations;
    Logger _log;
};

}  // namespace vpux
