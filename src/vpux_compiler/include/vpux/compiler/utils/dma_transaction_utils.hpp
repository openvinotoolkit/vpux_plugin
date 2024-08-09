//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <cstddef>
#include <utility>

namespace vpux {
class NDTypeInterface;

struct DMAPattern {
    DMAPattern(llvm::SmallVector<size_t> dims, llvm::SmallVector<size_t> strides)
            : dims(std::move(dims)), strides(std::move(strides)) {
    }
    DMAPattern() = default;

    llvm::SmallVector<size_t> dims;
    llvm::SmallVector<size_t> strides;
};

struct DMATransaction {
    DMATransaction(llvm::SmallVector<DMAPattern> inputs, llvm::SmallVector<DMAPattern> outputs)
            : inputs(std::move(inputs)), outputs(std::move(outputs)) {
    }
    DMATransaction() = default;

    llvm::SmallVector<DMAPattern> inputs;
    llvm::SmallVector<DMAPattern> outputs;
};

DMAPattern reduceDimsForDma(vpux::NDTypeInterface ndType);
void patchDimsForNPU37XX(DMAPattern& dmaPattern);

}  // namespace vpux
