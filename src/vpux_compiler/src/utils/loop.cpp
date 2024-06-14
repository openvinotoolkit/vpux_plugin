//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Threading.h>
#include <mlir/IR/Types.h>

using namespace vpux;

namespace {
typedef struct {
    int64_t begin;
    int64_t end;
    bool partition;
} LoopRange;

SmallVector<LoopRange> partitionData(ArrayRef<int64_t> dims, int64_t availableThreads, int64_t& threadsToUse) {
    SmallVector<LoopRange> ranges(dims.size());

    std::optional<size_t> partitionIdx;
    int64_t maxDim = dims[0];
    int64_t maxDimIdx = 0;
    for (size_t dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
        ranges[dimIdx] = {0, dims[dimIdx], /*partition=*/false};
        if (!partitionIdx.has_value() && dims[dimIdx] >= availableThreads) {
            partitionIdx = dimIdx;
            ranges[dimIdx].partition = true;
            threadsToUse = availableThreads;
        }

        if (dims[dimIdx] > maxDim) {
            maxDim = dims[dimIdx];
            maxDimIdx = dimIdx;
        }
    }

    // No dimension is greater than the available number of threads, partition
    // the largest dimension
    if (!partitionIdx.has_value()) {
        ranges[maxDimIdx].partition = true;
        threadsToUse = dims[maxDimIdx];
    }

    return ranges;
}

mlir::LogicalResult rangeAdjust(MutableArrayRef<LoopRange> ranges, ArrayRef<int64_t> dims, int64_t threadIdx,
                                int64_t threadsToUse) {
    for (size_t idx = 0; idx < ranges.size(); ++idx) {
        if (!ranges[idx].partition) {
            continue;
        }
        int64_t elementsPerThread = divUp(dims[idx], static_cast<int64_t>(threadsToUse));

        ranges[idx].begin = elementsPerThread * threadIdx;
        ranges[idx].end = std::min(ranges[idx].begin + elementsPerThread, dims[idx]);
        if (ranges[idx].begin >= ranges[idx].end) {
            return mlir::failure();
        }
    }
    return mlir::success();
}
}  // namespace

void vpux::loop_1d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, vpux::FuncRef<void(int64_t)> func) {
    if (dim0 <= 0) {
        return;
    }

    if (!ctx->isMultithreadingEnabled() || policy == LoopExecPolicy::Sequential) {
        for (int64_t idxDim0 = 0; idxDim0 < dim0; ++idxDim0) {
            func(idxDim0);
        }
        return;
    }

    auto& threadPool = ctx->getThreadPool();
    llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
    const auto availableThreads = threadPool.getThreadCount();
    SmallVector<int64_t> dims = {dim0};

    int64_t threadsToUse = 0;
    auto ranges = partitionData(dims, availableThreads, threadsToUse);

    for (int64_t threadIdx = 0; threadIdx < threadsToUse; ++threadIdx) {
        if (rangeAdjust(ranges, dims, threadIdx, threadsToUse).failed()) {
            break;
        }

        tasksGroup.async([ranges, func] {
            for (int64_t idxDim0 = ranges[0].begin; idxDim0 < ranges[0].end; ++idxDim0) {
                func(idxDim0);
            }
        });
    }
}

void vpux::loop_2d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, int64_t dim1,
                   FuncRef<void(int64_t, int64_t)> func) {
    if (dim0 <= 0 || dim1 <= 0) {
        return;
    }

    if (!ctx->isMultithreadingEnabled() || policy == LoopExecPolicy::Sequential) {
        for (int64_t idxDim0 = 0; idxDim0 < dim0; ++idxDim0) {
            for (int64_t idxDim1 = 0; idxDim1 < dim1; ++idxDim1) {
                func(idxDim0, idxDim1);
            }
        }
        return;
    }

    auto& threadPool = ctx->getThreadPool();
    llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
    const auto availableThreads = threadPool.getThreadCount();
    SmallVector<int64_t> dims = {dim0, dim1};

    int64_t threadsToUse = 0;
    auto ranges = partitionData(dims, availableThreads, threadsToUse);

    for (int64_t threadIdx = 0; threadIdx < threadsToUse; ++threadIdx) {
        if (rangeAdjust(ranges, dims, threadIdx, threadsToUse).failed()) {
            break;
        }

        tasksGroup.async([ranges, func] {
            for (int64_t idxDim0 = ranges[0].begin; idxDim0 < ranges[0].end; ++idxDim0) {
                for (int64_t idxDim1 = ranges[1].begin; idxDim1 < ranges[1].end; ++idxDim1) {
                    func(idxDim0, idxDim1);
                }
            }
        });
    }
}

void vpux::loop_3d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, int64_t dim1, int64_t dim2,
                   FuncRef<void(int64_t, int64_t, int64_t)> func) {
    if (dim0 <= 0 || dim1 <= 0 || dim2 <= 0) {
        return;
    }

    if (!ctx->isMultithreadingEnabled() || policy == LoopExecPolicy::Sequential) {
        for (int64_t idxDim0 = 0; idxDim0 < dim0; ++idxDim0) {
            for (int64_t idxDim1 = 0; idxDim1 < dim1; ++idxDim1) {
                for (int64_t idxDim2 = 0; idxDim2 < dim2; ++idxDim2) {
                    func(idxDim0, idxDim1, idxDim2);
                }
            }
        }
        return;
    }

    auto& threadPool = ctx->getThreadPool();
    llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
    const auto availableThreads = threadPool.getThreadCount();
    SmallVector<int64_t> dims = {dim0, dim1, dim2};

    int64_t threadsToUse = 0;
    auto ranges = partitionData(dims, availableThreads, threadsToUse);

    for (int64_t threadIdx = 0; threadIdx < threadsToUse; ++threadIdx) {
        if (rangeAdjust(ranges, dims, threadIdx, threadsToUse).failed()) {
            break;
        }

        tasksGroup.async([ranges, func] {
            for (int64_t idxDim0 = ranges[0].begin; idxDim0 < ranges[0].end; ++idxDim0) {
                for (int64_t idxDim1 = ranges[1].begin; idxDim1 < ranges[1].end; ++idxDim1) {
                    for (int64_t idxDim2 = ranges[2].begin; idxDim2 < ranges[2].end; ++idxDim2) {
                        func(idxDim0, idxDim1, idxDim2);
                    }
                }
            }
        });
    }
}

void vpux::loop_4d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, int64_t dim1, int64_t dim2,
                   int64_t dim3, FuncRef<void(int64_t, int64_t, int64_t, int64_t)> func) {
    if (dim0 <= 0 || dim1 <= 0 || dim2 <= 0 || dim3 <= 0) {
        return;
    }

    if (!ctx->isMultithreadingEnabled() || policy == LoopExecPolicy::Sequential) {
        for (int64_t idxDim0 = 0; idxDim0 < dim0; ++idxDim0) {
            for (int64_t idxDim1 = 0; idxDim1 < dim1; ++idxDim1) {
                for (int64_t idxDim2 = 0; idxDim2 < dim2; ++idxDim2) {
                    for (int64_t idxDim3 = 0; idxDim3 < dim3; ++idxDim3) {
                        func(idxDim0, idxDim1, idxDim2, idxDim3);
                    }
                }
            }
        }
        return;
    }

    auto& threadPool = ctx->getThreadPool();
    llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
    const auto availableThreads = threadPool.getThreadCount();
    SmallVector<int64_t> dims = {dim0, dim1, dim2, dim3};

    int64_t threadsToUse = 0;
    auto ranges = partitionData(dims, availableThreads, threadsToUse);

    for (int64_t threadIdx = 0; threadIdx < threadsToUse; ++threadIdx) {
        if (rangeAdjust(ranges, dims, threadIdx, threadsToUse).failed()) {
            break;
        }

        tasksGroup.async([ranges, func] {
            for (int64_t idxDim0 = ranges[0].begin; idxDim0 < ranges[0].end; ++idxDim0) {
                for (int64_t idxDim1 = ranges[1].begin; idxDim1 < ranges[1].end; ++idxDim1) {
                    for (int64_t idxDim2 = ranges[2].begin; idxDim2 < ranges[2].end; ++idxDim2) {
                        for (int64_t idxDim3 = ranges[3].begin; idxDim3 < ranges[3].end; ++idxDim3) {
                            func(idxDim0, idxDim1, idxDim2, idxDim3);
                        }
                    }
                }
            }
        });
    }
}
