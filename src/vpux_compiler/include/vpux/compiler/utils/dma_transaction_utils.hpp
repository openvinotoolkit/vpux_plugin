//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {

struct DMATransaction {
    DMATransaction(ArrayRef<uint64_t> dims, ArrayRef<uint64_t> strides): dims(dims), strides(strides) {
    }
    DMATransaction() = default;

    llvm::SmallVector<uint64_t> dims;
    llvm::SmallVector<uint64_t> strides;
};

DMATransaction reduceDimsForDma(vpux::NDTypeInterface ndType);
void patchDimsForNPU37XX(DMATransaction& dmaTransactionDetails);

}  // namespace vpux
