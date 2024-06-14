//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/Support/ThreadPool.h>
#include <mlir/IR/MLIRContext.h>

namespace vpux {
namespace Const {

class BackgroundConstantFolding {
public:
    BackgroundConstantFolding(mlir::MLIRContext* ctx, size_t maxConcurrentTasks, bool collectStatistics,
                              Logger log = Logger::global());
    ~BackgroundConstantFolding();

    BackgroundConstantFolding(const BackgroundConstantFolding&) = delete;
    BackgroundConstantFolding(BackgroundConstantFolding&&) = delete;
    BackgroundConstantFolding& operator=(const BackgroundConstantFolding&) = delete;
    BackgroundConstantFolding& operator=(BackgroundConstantFolding&&) = delete;

private:
    std::shared_future<void> initFoldingListener(llvm::ThreadPool& threadPool);
    void stopFoldingListener();
    void processFoldingRequest(const FoldingRequest& foldingRequest, ConstantFoldingCache& cache);

    bool _isEnabled = true;
    mlir::MLIRContext* _ctx;
    const size_t _maxConcurrentTasks;
    std::shared_future<void> _listenerThread;
    Logger _log;

    // The goal is to prevent overloading the thread-pool queue with background folding tasks, as the main compilation
    // process also utilizes the pool concurrently. To achieve this, a limit is imposed on the number of tasks submitted
    // to the pool from the background. Currently, the pool lacks a sophisticated mechanism to prioritize tasks from the
    // main compilation over those from the background.
    std::atomic<size_t> _activeTasks{0};
    std::mutex _mutex;
    std::condition_variable _cv;
};

}  // namespace Const
}  // namespace vpux

#endif
