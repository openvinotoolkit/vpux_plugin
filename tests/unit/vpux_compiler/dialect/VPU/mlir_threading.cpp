//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Threading.h>
#include <mlir/IR/Types.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIRThreadPool = MLIR_UnitBase;

namespace {
constexpr int64_t MAX_TEST_THREADS = 16;
}

TEST_F(MLIRThreadPool, ParallelFor_1d) {
    mlir::MLIRContext ctx(registry);
    SmallVector<int64_t> testNumElements = {1, 2, 4, 8, 16, 22, 32, 64, 1000};
    int64_t bias = 1;

    for (auto numElements : testNumElements) {
        // Prepare data
        SmallVector<int64_t> values(numElements, 0);
        SmallVector<int64_t> expected(numElements, bias);

        const auto runMLIRLoop1D = [&](const int64_t numThreads, LoopExecPolicy policy) {
            llvm::ThreadPool threadPool(llvm::optimal_concurrency(numThreads));
            ctx.disableMultithreading();
            ctx.setThreadPool(threadPool);
            SmallVector<int64_t> result(numElements, 0);

            loop_1d(policy, &ctx, values.size(), [&](int64_t i) {
                result[i] = values[i] + bias;
            });

            EXPECT_EQ(result, expected);
        };

        SmallVector<LoopExecPolicy> policy = {LoopExecPolicy::Sequential, LoopExecPolicy::Parallel};
        for (size_t numThreads = 1; numThreads <= MAX_TEST_THREADS; numThreads *= 2) {
            for (auto it : policy) {
                runMLIRLoop1D(numThreads, it);
            }
        }
    }
}

TEST_F(MLIRThreadPool, ParallelFor_2d) {
    mlir::MLIRContext ctx(registry);
    SmallVector<SmallVector<int64_t>> configurations = {{1, 64}, {100, 100}, {1, 16}, {16, 1}, {32, 32}};

    int64_t bias = 1;

    for (const auto& dims : configurations) {
        // Prepare data
        SmallVector<SmallVector<int64_t>> values(dims[0], SmallVector<int64_t>(dims[1], 0));
        SmallVector<SmallVector<int64_t>> expected(dims[0], SmallVector<int64_t>(dims[1], bias));

        const auto runMLIRLoop2D = [&](const int64_t numThreads, LoopExecPolicy policy) {
            llvm::ThreadPool threadPool(llvm::optimal_concurrency(numThreads));
            ctx.disableMultithreading();
            ctx.setThreadPool(threadPool);
            SmallVector<SmallVector<int64_t>> result(dims[0], SmallVector<int64_t>(dims[1], 0));

            loop_2d(policy, &ctx, dims[0], dims[1], [&](int64_t x, int64_t y) {
                result[x][y] = values[x][y] + bias;
            });

            EXPECT_EQ(result, expected);
        };

        SmallVector<LoopExecPolicy> policy = {LoopExecPolicy::Sequential, LoopExecPolicy::Parallel};
        for (size_t numThreads = 1; numThreads <= MAX_TEST_THREADS; numThreads *= 2) {
            for (auto it : policy) {
                runMLIRLoop2D(numThreads, it);
            }
        }
    }
}

TEST_F(MLIRThreadPool, ParallelFor_3d) {
    mlir::MLIRContext ctx(registry);
    SmallVector<SmallVector<int64_t>> configurations = {{1, 1, 1024}, {100, 15, 3}, {3, 3, 3}, {126, 148, 30}};

    int64_t bias = 1;
    for (const auto& dims : configurations) {
        // Prepare data
        SmallVector<SmallVector<SmallVector<int64_t>>> values(
                dims[0], SmallVector<SmallVector<int64_t>>(dims[1], SmallVector<int64_t>(dims[2], 0)));
        SmallVector<SmallVector<SmallVector<int64_t>>> expected(
                dims[0], SmallVector<SmallVector<int64_t>>(dims[1], SmallVector<int64_t>(dims[2], bias)));

        const auto runMLIRLoop1D = [&](const int64_t numThreads, LoopExecPolicy policy) {
            llvm::ThreadPool threadPool(llvm::optimal_concurrency(numThreads));
            ctx.disableMultithreading();
            ctx.setThreadPool(threadPool);
            SmallVector<SmallVector<SmallVector<int64_t>>> result(
                    dims[0], SmallVector<SmallVector<int64_t>>(dims[1], SmallVector<int64_t>(dims[2], 0)));

            loop_3d(policy, &ctx, dims[0], dims[1], dims[2], [&](int64_t x, int64_t y, int64_t z) {
                result[x][y][z] = values[x][y][z] + bias;
            });

            EXPECT_EQ(result, expected);
        };

        SmallVector<LoopExecPolicy> policy = {LoopExecPolicy::Sequential, LoopExecPolicy::Parallel};
        for (size_t numThreads = 1; numThreads <= MAX_TEST_THREADS; numThreads *= 2) {
            for (auto it : policy) {
                runMLIRLoop1D(numThreads, it);
            }
        }
    }
}

TEST_F(MLIRThreadPool, ParallelFor_4d) {
    mlir::MLIRContext ctx(registry);
    SmallVector<SmallVector<int64_t>> configurations = {{1, 1, 1, 1},
                                                        {1, 16, 16, 100},
                                                        {100, 1, 1, 32},
                                                        {128, 16, 16, 8},
                                                        {2, 2, 8, 8}};

    int64_t bias = 1;

    for (const auto& dims : configurations) {
        // Prepare data
        SmallVector<SmallVector<SmallVector<SmallVector<int64_t>>>> values(
                dims[0],
                SmallVector<SmallVector<SmallVector<int64_t>>>(
                        dims[1], SmallVector<SmallVector<int64_t>>(dims[2], SmallVector<int64_t>(dims[3], 0))));
        SmallVector<SmallVector<SmallVector<SmallVector<int64_t>>>> expected(
                dims[0],
                SmallVector<SmallVector<SmallVector<int64_t>>>(
                        dims[1], SmallVector<SmallVector<int64_t>>(dims[2], SmallVector<int64_t>(dims[3], bias))));

        const auto runMLIRLoop1D = [&](const int64_t numThreads, LoopExecPolicy policy) {
            llvm::ThreadPool threadPool(llvm::optimal_concurrency(numThreads));
            ctx.disableMultithreading();
            ctx.setThreadPool(threadPool);
            SmallVector<SmallVector<SmallVector<SmallVector<int64_t>>>> result(
                    dims[0],
                    SmallVector<SmallVector<SmallVector<int64_t>>>(
                            dims[1], SmallVector<SmallVector<int64_t>>(dims[2], SmallVector<int64_t>(dims[3], 0))));

            loop_4d(policy, &ctx, dims[0], dims[1], dims[2], dims[3], [&](int64_t x, int64_t y, int64_t z, int64_t w) {
                result[x][y][z][w] = values[x][y][z][w] + bias;
            });

            EXPECT_EQ(result, expected);
        };

        SmallVector<LoopExecPolicy> policy = {LoopExecPolicy::Sequential, LoopExecPolicy::Parallel};
        for (size_t numThreads = 1; numThreads <= MAX_TEST_THREADS; numThreads *= 2) {
            for (auto it : policy) {
                runMLIRLoop1D(numThreads, it);
            }
        }
    }
}
