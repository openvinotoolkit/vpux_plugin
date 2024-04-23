//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/Support/ThreadPool.h>
#include <mlir/IR/MLIRContext.h>

namespace vpux {
namespace Const {

SmallVector<std::shared_future<void>> initBackgroundConstantFoldingThreads(mlir::MLIRContext* ctx, size_t numThreads,
                                                                           bool collectStatistics);

void stopBackgroundConstantFoldingThreads(mlir::MLIRContext* ctx, ArrayRef<std::shared_future<void>> foldingThreads,
                                          bool collectStatistics, Logger log = Logger::global());

}  // namespace Const
}  // namespace vpux

#endif
