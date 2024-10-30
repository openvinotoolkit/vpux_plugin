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
    SmallVector<std::pair<mlir::Operation*, size_t>> inputUserMapping;
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

    virtual SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp /*mainFunction*/) {
        return {};
    }
};

//
// FunctionOutlinerNaive
//

class FunctionOutlinerNaive final : public IFunctionOutliner {
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

class FunctionOutlinerRepeatingBlocks final : public IFunctionOutliner {
public:
    FunctionOutlinerRepeatingBlocks(size_t minOpsInBlock, size_t maxNumIterations, bool separateFunctions,
                                    bool weightsAsInputs, Logger log);

    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    size_t _minOpsInBlock;
    size_t _maxNumIterations;
    bool _separateFunctions;
    bool _weightsAsInputs;
    Logger _log;
};

//
// FunctionOutlinerBatching
//

class FunctionOutlinerBatching final : public IFunctionOutliner {
public:
    FunctionOutlinerBatching(Logger log);
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    Logger _log;
};

//
// Option parser
//
struct NaiveOptions {
    static constexpr size_t NUM_PARTS_DEFAULT = 2;

    size_t numParts;
};

struct RepeatingBlocksOptions {
    static constexpr size_t MIN_OPS_IN_BLOCK_DEFAULT = 30;
    static constexpr size_t MAX_NUM_ITERATIONS_DEFAULT = 100;
    static constexpr bool WEIGHTS_AS_INPUTS_DEFAULT = false;

    size_t minOpsInBlock;
    size_t maxNumIterations;
    bool weightsAsInputs;
};

struct RepeatingBlocksSeparateFunctionsOptions {
    static constexpr size_t MIN_OPS_IN_BLOCK_DEFAULT = 30;
    static constexpr size_t MAX_NUM_ITERATIONS_DEFAULT = 100;

    size_t minOpsInBlock;
    size_t maxNumIterations;
};

struct BatchingOptions {};

class OutlinerPassOptions {
    std::vector<std::variant<NaiveOptions, RepeatingBlocksOptions, RepeatingBlocksSeparateFunctionsOptions,
                             BatchingOptions>>
            _options;

public:
    template <class T>
    const T* getIf(size_t i) const {
        return std::get_if<T>(&_options[i]);
    }

    size_t count() const {
        return _options.size();
    }

    static OutlinerPassOptions createFromString(StringRef param);
};

}  // namespace vpux
