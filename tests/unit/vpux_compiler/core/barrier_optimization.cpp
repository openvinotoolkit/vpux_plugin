//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/core/barrier_info.hpp"

#include <gtest/gtest.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <llvm/ADT/SetOperations.h>

using namespace vpux;
using BarrierInfoTests = ::testing::Test;
using MLIR_BarrierInfoTests = MLIR_UnitBase;

enum class TestTarget {
    optimizeBarriersWithFIFOdeps,
    createBarrierDependenciesImpliedByFIFO,
    createAndRemoveBarrierDependenciesImpliedByFIFO,
    getControlGraphBlockIndex,
    isSyncPoint,
    getControlGraphBlockTaskRangeFirstBlock,
    getControlGraphBlockTaskRangeMidBlock,
    getControlGraphBlockTaskRangeLastBlock,
};

void checkBarrierMaps(const BarrierInfoTest::BarrierMaps& expectedResult,
                      const BarrierInfoTest::BarrierMaps& testResult, bool checkUpdateAndWaitBarriers = true,
                      bool checkProducersAndConsumers = true) {
    if (checkUpdateAndWaitBarriers) {
        EXPECT_EQ(expectedResult.taskUpdateBarriers, testResult.taskUpdateBarriers);
        EXPECT_EQ(expectedResult.taskWaitBarriers, testResult.taskWaitBarriers);
    }
    if (checkProducersAndConsumers) {
        EXPECT_EQ(expectedResult.barrierProducerMap, testResult.barrierProducerMap);
        EXPECT_EQ(expectedResult.barrierConsumerMap, testResult.barrierConsumerMap);
    }
}

/**
 * Configuration with 1 redundant producer
 *
 *        0                     0
 *       / \                    |
 *       |  \                   |
 *      b0   \                 b0
 *       |\   |                 |\
 *       | 1  |                 | 1
 *       |  \ |                 |  \
 *       |   b1        =>       |   b1
 *       2   /                  2   /
 *       |  /                   |  /
 *      b2 /                   b2 /
 *       |/                     |/
 *       3                      3
 *
 *
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> redundantProducerConfig() {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {0},     // task 2
            {1, 2},  // task 3
    };

    inputBarrierMaps.barrierConsumerMap = {
            {1, 2},  // barrier 0
            {3},     // barrier 1
            {3},     // barrier 2
    };

    inputBarrierMaps.taskUpdateBarriers = {
            {0, 1},  // task 0
            {1},     // task 1
            {2},     // task 2
            {},      // task 3
    };

    inputBarrierMaps.barrierProducerMap = {
            {0},     // barrier 0
            {0, 1},  // barrier 1
            {2},     // barrier 2
    };

    inputBarrierMaps.nTasks = 4;
    inputBarrierMaps.nBarriers = 3;

    // expected results
    // no change expected in waitbarriers
    expectedBarrierMaps.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {0},     // task 2
            {1, 2},  // task 3
    };

    // no change expected in consumer map
    expectedBarrierMaps.barrierConsumerMap = {
            {1, 2},  // barrier 0
            {3},     // barrier 1
            {3},     // barrier 2
    };

    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2
            {},   // task 3
    };

    expectedBarrierMaps.barrierProducerMap = {
            {0},  // barrier 0
            {1},  // barrier 1
            {2},  // barrier 2
    };

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

/**
 *
 * This graph tests optimization of parallel branches depending on limit of allowed number of variants per barrier.
 * If _maxVariantCountPerBarrier = 4 the schedule will not be optimized as tested in
 * optimizeParallelBlocksWithTooFewVariantsPerBarrier.
 *
 * Setting in the test body to
 *
 *      barrierInfoTest.setMaxVariantCountPerBarrier(64);
 *
 * optimizes the graph correctly.
 *
 *        0..7                                  0..7
 *         /\                                     |
 *    fully connected                             |
 *  b0              b1                           b0
 *  |               |                             |
 *  /\              |                            / \
 * 8..16            |                           8..16
 *  \/              |              =>            \ /
 *  b2              |                            b2
 *   |              |                             |
 *  17              |                            17
 *   |              |                             |
 *  b3              |                            b3
 *   |              |                             |
 *   \-------------\|                             |
 *                  |                             |
 *                 / \                           / \
 *                18..28                        18..28
 *                 \ /                           \ /
 *                  b4                            b4
 *                  |                             |
 *                  29                            29
 *
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> parallelBlocksConfig() {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskUpdateBarriers = {
            {0, 1},  // task 0
            {0, 1},  // task 1
            {0, 1},  // task 2
            {0, 1},  // task 3
            {0, 1},  // task 4
            {0, 1},  // task 5
            {0, 1},  // task 6
            {0, 1},  // task 7
            {2},     // task 8
            {2},     // task 9
            {2},     // task 10
            {2},     // task 11
            {2},     // task 12
            {2},     // task 13
            {2},     // task 14
            {2},     // task 15
            {2},     // task 16
            {3},     // task 17 -- end of 1st block
            {4},     // task 18
            {4},     // task 19
            {4},     // task 20
            {4},     // task 21
            {4},     // task 22
            {4},     // task 23
            {4},     // task 24
            {4},     // task 25
            {4},     // task 26
            {4},     // task 27
            {4},     // task 28
            {},      // task 29
    };

    inputBarrierMaps.taskWaitBarriers = {
            {},      // task 0
            {},      // task 1
            {},      // task 2
            {},      // task 3
            {},      // task 4
            {},      // task 5
            {},      // task 6
            {},      // task 7
            {0},     // task 8
            {0},     // task 9
            {0},     // task 10
            {0},     // task 11
            {0},     // task 12
            {0},     // task 13
            {0},     // task 14
            {0},     // task 15
            {0},     // task 16
            {2},     // task 17 -- end of 1st block
            {1, 3},  // task 18
            {1, 3},  // task 19
            {1, 3},  // task 20
            {1, 3},  // task 21
            {1, 3},  // task 22
            {1, 3},  // task 23
            {1, 3},  // task 24
            {1, 3},  // task 25
            {1, 3},  // task 26
            {1, 3},  // task 27
            {1, 3},  // task 28
            {4},     // task 29
    };

    fillProducersAndConsumers(inputBarrierMaps);
    inputBarrierMaps.nTasks = 30;
    inputBarrierMaps.nBarriers = 5;

    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {0},  // task 1
            {0},  // task 2
            {0},  // task 3
            {0},  // task 4
            {0},  // task 5
            {0},  // task 6
            {0},  // task 7
            {2},  // task 8
            {2},  // task 9
            {2},  // task 10
            {2},  // task 11
            {2},  // task 12
            {2},  // task 13
            {2},  // task 14
            {2},  // task 15
            {2},  // task 16
            {3},  // task 17 -- end of 1st block
            {4},  // task 18
            {4},  // task 19
            {4},  // task 20
            {4},  // task 21
            {4},  // task 22
            {4},  // task 23
            {4},  // task 24
            {4},  // task 25
            {4},  // task 26
            {4},  // task 27
            {4},  // task 28
            {},   // task 29
    };

    expectedBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {},   // task 1
            {},   // task 2
            {},   // task 3
            {},   // task 4
            {},   // task 5
            {},   // task 6
            {},   // task 7
            {0},  // task 8
            {0},  // task 9
            {0},  // task 10
            {0},  // task 11
            {0},  // task 12
            {0},  // task 13
            {0},  // task 14
            {0},  // task 15
            {0},  // task 16
            {2},  // task 17 -- end of 1st block
            {3},  // task 18
            {3},  // task 19
            {3},  // task 20
            {3},  // task 21
            {3},  // task 22
            {3},  // task 23
            {3},  // task 24
            {3},  // task 25
            {3},  // task 26
            {3},  // task 27
            {3},  // task 28
            {4},  // task 29
    };

    fillProducersAndConsumers(expectedBarrierMaps);

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

/**
 * Configuration with same producer but different consumers
 *
 *            0                 0
 *           / \                |
 *       bar0   bar1   =>      bar0
 *         |     |              /\
 *         1     2             1  2
 *
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> sameProducerDifferentConsumersConfigA() {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
    };

    inputBarrierMaps.taskUpdateBarriers = {
            {0, 1},  // task 0
            {},      // task 1
            {},      // task 2
    };

    fillProducersAndConsumers(inputBarrierMaps);
    inputBarrierMaps.nTasks = 3;
    inputBarrierMaps.nBarriers = 2;

    // expected results
    expectedBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
    };

    expectedBarrierMaps.barrierConsumerMap = {
            {1, 2},  // barrier 0
            {},      // barrier 1
    };

    // no change expected in update barriers
    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {},   // task 1
            {},   // task 2
    };

    // no change expected in producer map
    expectedBarrierMaps.barrierProducerMap = {
            {0},  // barrier 0
            {},   // barrier 1
    };

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

/**
 * Configuration with same producer but different consumers
 *
 *           0                   0                 0
 *         /   \                 |                 |
 *       bar0 bar2              bar0 -\           bar0
 *         |    |                |    |            |
 *         1    |      =>        1    |    =>      1
 *         |    |     (1)        |    |   (2)      |
 *       bar1   |               bar1  |           bar1
 *         |    |                |    |            |
 *         2----/                2----/            2
 *
 * Optimization done in barrierInfo::optimizeBarriersWithSameProducers (step 1) removes dependence on bar2 from task 2.
 * Subsequent optimization of consumers (step 2) would remove redundant dependence of task 2 on bar0 (scenario tested in
 * redundantConsumerConfig).
 *
 * For twoStageOptimization the full set of optimizations is performed - barrierInfo::barrierOptimization() - in order
 * to assure the correct order of optimization steps.
 *
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> sameProducerDifferentConsumersConfigB(
        bool twoStageOptimization) {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1, 2},  // task 2
    };

    inputBarrierMaps.taskUpdateBarriers = {
            {0, 2},  // task 0
            {1},     // task 1
            {},      // task 2
    };

    fillProducersAndConsumers(inputBarrierMaps);
    inputBarrierMaps.nTasks = 3;
    inputBarrierMaps.nBarriers = 3;

    // expected results
    if (twoStageOptimization) {
        expectedBarrierMaps.taskWaitBarriers = {
                {},   // task 0
                {0},  // task 1
                {1},  // task 2
        };

        expectedBarrierMaps.barrierConsumerMap = {
                {1},  // barrier 0
                {2},  // barrier 1
                {},   // barrier 2
        };

        expectedBarrierMaps.taskUpdateBarriers = {
                {0},  // task 0
                {1},  // task 1
                {},   // task 2
        };

        expectedBarrierMaps.barrierProducerMap = {
                {0},  // barrier 0
                {1},  // barrier 1
                {},   // barrier 2
        };
    } else {
        expectedBarrierMaps.taskWaitBarriers = {
                {},      // task 0
                {0},     // task 1
                {0, 1},  // task 2
        };

        expectedBarrierMaps.barrierConsumerMap = {
                {1, 2},  // barrier 0 -- task 2 now depends on bar0
                {2},     // barrier 1
                {},      // barrier 2
        };

        expectedBarrierMaps.taskUpdateBarriers = {
                {0},  // task 0
                {1},  // task 1
                {},   // task 2
        };

        expectedBarrierMaps.barrierProducerMap = {
                {0},  // barrier 0
                {1},  // barrier 1
                {},   // barrier 2
        };
    }

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

/**
 * Configuration with 1 redundant consumer
 *
 *       0  1                    0  1
 *        \/                      \/
 *       bar0 --\                bar0
 *        /\    |                 /\
 *       2  3   |      =>        2  3
 *        \/    |                 \/
 *       bar1   |                bar1
 *        |     |                 |
 *        4-----/                 4
 *        |                       |
 *       bar2                    bar2
 *
 * Task 4 doesn't need to wait for barrier bar0 as it already waits for barrier bar1
 *
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> redundantConsumerConfig() {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskWaitBarriers = {
            {},      // task 0
            {},      // task 1
            {0},     // task 2
            {0},     // task 3
            {0, 1},  // task 4
    };

    inputBarrierMaps.barrierConsumerMap = {
            {2, 3, 4},  // barrier 0
            {4},        // barrier 1
            {},         // barrier 2
    };

    inputBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
    };

    inputBarrierMaps.barrierProducerMap = {
            {0, 1},  // barrier 0
            {2, 3},  // barrier 1
            {4},     // barrier 2
    };

    inputBarrierMaps.nTasks = 5;
    inputBarrierMaps.nBarriers = 3;

    // expected results
    expectedBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {},   // task 1
            {0},  // task 2
            {0},  // task 3
            {1},  // task 4
    };

    expectedBarrierMaps.barrierConsumerMap = {
            {2, 3},  // barrier 0
            {4},     // barrier 1
            {},      // barrier 2
    };

    // no change expected in update barriers
    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
    };

    // no change expected in producer map
    expectedBarrierMaps.barrierProducerMap = {
            {0, 1},  // barrier 0
            {2, 3},  // barrier 1
            {4},     // barrier 2
    };

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

/**
 * Configuration with 3 redundant consumers
 *
 *       0  1                    0  1
 *        \/                      \/
 *       bar0 --\                bar0
 *        /\    |                 /\
 *       2  3   |                2  3
 *        \/    |                 \/
 *       bar1   |                bar1
 *        |     |                 |
 *        4-----/                 4          <- sync point
 *        |                       |
 *       bar2 -----\             bar2
 *       /|\       |     =>      /|\
 *      / | \      |            / | \
 *     5  6  7     |           5  6  7
 *      \ | /      |            \ | /
 *       bar3 --\  |             bar3
 *        |     |  |              |
 *        8     |  |              8
 *        |     |  |              |
 *       bar4   |  |             bar4
 *        |     |  |              |
 *        9-----/--/              9
 *        |                       |
 *       bar5                    bar5
 *
 * For case when 1st block is optimized task 4 doesn't need to wait for barrier bar0 as it already waits for barrier
 * bar1. For case when 2nd block is optimized task 9 doesn't need to wait for barrier bar0 and bar2 and bar3 as it
 * already waits for barrier bar4. Control graph split is done every 5 tasks. The sync point is
 * located at tasks ID = 4
 *
 * When single block is optimized, the other block should not be altered.
 *
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> redundantConsumersWithTwoBlockTaskSplitConfig(
        std::vector<size_t> blocksToOptimize) {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskWaitBarriers = {
            {},         // task 0
            {},         // task 1
            {0},        // task 2
            {0},        // task 3
            {0, 1},     // task 4
            {2},        // task 5
            {2},        // task 6
            {2},        // task 7
            {3},        // task 8
            {2, 3, 4},  // task 9
    };

    inputBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {3},  // task 6
            {3},  // task 7
            {4},  // task 8
            {5},  // task 9
    };

    fillProducersAndConsumers(inputBarrierMaps);
    inputBarrierMaps.nTasks = 10;
    inputBarrierMaps.nBarriers = 6;
    inputBarrierMaps.syncTasksIds = {4};  // Control graph split done in the middle of the graph.

    // expected results
    if (blocksToOptimize == std::vector<size_t>({0, 1})) {
        expectedBarrierMaps.taskWaitBarriers = {
                {},   // task 0
                {},   // task 1
                {0},  // task 2
                {0},  // task 3
                {1},  // task 4
                {2},  // task 5
                {2},  // task 6
                {2},  // task 7
                {3},  // task 8
                {4},  // task 9
        };

        expectedBarrierMaps.barrierConsumerMap = {
                {2, 3},     // barrier 0
                {4},        // barrier 1
                {5, 6, 7},  // barrier 2
                {8},        // barrier 3
                {9},        // barrier 4
                {},         // barrier 5
        };
    } else if (blocksToOptimize == std::vector<size_t>({1})) {
        expectedBarrierMaps.taskWaitBarriers = {
                {},      // task 0
                {},      // task 1
                {0},     // task 2
                {0},     // task 3
                {0, 1},  // task 4
                {2},     // task 5
                {2},     // task 6
                {2},     // task 7
                {3},     // task 8
                {4},     // task 9
        };

        expectedBarrierMaps.barrierConsumerMap = {
                {2, 3, 4},  // barrier 0
                {4},        // barrier 1
                {5, 6, 7},  // barrier 2
                {8},        // barrier 3
                {9},        // barrier 4
                {},         // barrier 5
        };
    }

    // no change expected in update barriers
    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {0},  // task 1
            {1},  // task 2
            {1},  // task 3
            {2},  // task 4
            {3},  // task 5
            {3},  // task 6
            {3},  // task 7
            {4},  // task 8
            {5},  // task 9
    };

    // no change expected in producer map
    expectedBarrierMaps.barrierProducerMap = {
            {0, 1},     // barrier 0
            {2, 3},     // barrier 1
            {4},        // barrier 2
            {5, 6, 7},  // barrier 3
            {8},        // barrier 4
            {9},        // barrier 5
    };

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

/**
 *
 *           Shave0_1                      Shave0_1
 *             |                             |
 *           Bar0                          Bar0
 *         /      \                      /      \
 *      DMA0_1    Shave0_2            DMA0_1    Shave0_2
 *       |          |                            |
 *       |         Bar2                         Bar2
 *       |       /   |                        /   |
 *       |    DMA0_2 Shave0_3  =>          DMA0_2 Shave0_3
 *       |       \   |                        \   |
 *       |         Bar3                         Bar3
 *       \          |                            |
 *         \       Shave0_4                     Shave0_4
 *           \     /                            /
 *             Bar1                          Bar1
 *              |                             |
 *             Shave0_5                      Shave0_5
 *
 *  Dependency between DMA1->Bar1 is not needed!
 *  because DMA2 executes on same engine after DMA1 So there is implicit dep from DMA1- > DMA2
 *
 */
BarrierInfoTest::BarrierMaps barriersWithFIFOdependenciesNPU40XXconfig(mlir::MLIRContext* ctx,
                                                                       mlir::OwningOpRef<mlir::ModuleOp>& module) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>} {
            IE.PipelineOptions @Options {
            IE.Option @VPU.ReduceSupported : false
            IE.Option @VPU.AutoPaddingODU : false
            IE.Option @VPU.BarrierMaxVariantSum : 64
            IE.Option @VPU.BarrierMaxVariantCount : 128
            }
            IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
            IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
            IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
            IE.ExecutorResource 2 of @SHAVE_ACT
            IE.ExecutorResource 1 of @DPU
            }
            IE.ExecutorResource 1 of @M2I
            IE.ExecutorResource 2 of @DMA_NN
            IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}

            module @VPU.SW {
                func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE }
                func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
            }

            func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
                %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
                %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

                VPURT.Task updates(%bar0: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }

                VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }

                VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }

                VPURT.Task waits(%bar1: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }

                return %arg1: memref<1x3x64x64xf16, @DDR>
            }
        }
    )";

    module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    VPUX_THROW_UNLESS(module.get() != nullptr, "Cannot extract module from input IR");

    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {},   // task 1
            {2},  // task 2
            {3},  // task 3
            {3},  // task 4
            {1},  // task 5
            {},   // task 6
    };

    expectedBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {0},  // task 2
            {2},  // task 3
            {2},  // task 4
            {3},  // task 5
            {1},  // task 6
    };

    fillProducersAndConsumers(expectedBarrierMaps);
    return expectedBarrierMaps;
}

/**
 * All tasks are on port 0.
 *
 *        0                       0
 *        |                       |
 *       b0                      b0
 *     /  |  \                    |
 *    1   |  |                    1
 *    .   |  |                    .
 *    .  /   |                    .
 *    . /   /        =>           .
 *    2    /                      2
 *    . \ /                       .
 *    .  /                        .
 *    . /  \                      .
 *    3     \                     3
 *      \   /                     |
 *       b1                      b1
 *        |                       |
 *        4                       4
 *
 * 2 - doesn't need to update b1 and
 * 2 and 3 - do not need to wait for b0 because they are on the same FIFO
 * Barriers b0 and b1 are not optimized out within the scope of this test.
 *
 */
BarrierInfoTest::BarrierMaps barriersWithFIFOdependenciesNPU40XXconfig2(mlir::MLIRContext* ctx,
                                                                        mlir::OwningOpRef<mlir::ModuleOp>& module) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>} {
            IE.PipelineOptions @Options {
            IE.Option @VPU.ReduceSupported : false
            IE.Option @VPU.AutoPaddingODU : false
            IE.Option @VPU.BarrierMaxVariantSum : 64
            IE.Option @VPU.BarrierMaxVariantCount : 128
            }
            IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
            IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
            IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
            IE.ExecutorResource 2 of @SHAVE_ACT
            IE.ExecutorResource 1 of @DPU
            }
            IE.ExecutorResource 1 of @M2I
            IE.ExecutorResource 2 of @DMA_NN
            IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}

            func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
                %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
                %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

                VPURT.Task updates(%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar1 : !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                return %arg1: memref<1x3x64x64xf16, @DDR>
            }
        }
    )";

    module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    VPUX_THROW_UNLESS(module.get() != nullptr, "Cannot extract module from input IR");

    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {},   // task 1
            {},   // task 2
            {1},  // task 3
            {},   // task 4
    };

    expectedBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {},   // task 2
            {},   // task 3
            {1},  // task 4
    };

    fillProducersAndConsumers(expectedBarrierMaps);
    return expectedBarrierMaps;
}

/**
 *
 *          0p1
 *           |
 *           b0
 *       /  /|\  \
 *    1p1  / | \  4p0
 *    .   /  |  \  .
 *    .  /   |   \ .
 *    2p1   /     5p0
 *    . \  /       |
 *    .   /       /
 *     3p1  \    /
 *        \  |  /
 *           b1
 *           |
 *          6p1
 *
 */
BarrierInfoTest::BarrierMaps barriersWithFIFOdependenciesNPU40XXconfig3(mlir::MLIRContext* ctx,
                                                                        mlir::OwningOpRef<mlir::ModuleOp>& module,
                                                                        TestTarget target) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>} {
            IE.PipelineOptions @Options {
            IE.Option @VPU.ReduceSupported : false
            IE.Option @VPU.AutoPaddingODU : false
            IE.Option @VPU.BarrierMaxVariantSum : 64
            IE.Option @VPU.BarrierMaxVariantCount : 128
            }
            IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
            IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
            IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
            IE.ExecutorResource 2 of @SHAVE_ACT
            IE.ExecutorResource 1 of @DPU
            }
            IE.ExecutorResource 1 of @M2I
            IE.ExecutorResource 2 of @DMA_NN
            IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}

            func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
                %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
                %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

                VPURT.Task updates(%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar1 : !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                return %arg1: memref<1x3x64x64xf16, @DDR>
            }
        }
    )";

    module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    VPUX_THROW_UNLESS(module.get() != nullptr, "Cannot extract module from input IR");

    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    switch (target) {
    case TestTarget::optimizeBarriersWithFIFOdeps:
        /**
         * Expected results
         * for BarrierInfo::optimizeBarriers(false, true)
         * 2 - doesn't need to update b1 and
         * 2 and 3 - do not need to wait for b0 because they are on the same FIFO.
         * 5 doesn't need to wait for b0 because it's in the same FIFO as 4 which already
         * waits for b0.
         *
         *        0p1
         *        |
         *        b0
         *       /  \
         *     1p1   4p0
         *     |     |
         *     2p1   5p0
         *     |     |
         *     3p1   |
         *       \  /
         *        b1
         *        |
         *       6p1
         */
        expectedBarrierMaps.taskUpdateBarriers = {
                {0},  // task 0
                {},   // task 1
                {},   // task 2
                {1},  // task 3
                {},   // task 4
                {1},  // task 5
                {},   // task 6
        };

        expectedBarrierMaps.taskWaitBarriers = {
                {},   // task 0
                {0},  // task 1
                {},   // task 2
                {},   // task 3
                {0},  // task 4
                {},   // task 5
                {1},  // task 6
        };
        break;
    case TestTarget::createAndRemoveBarrierDependenciesImpliedByFIFO:
        /**
         * Expected results
         * for BarrierInfo::createBarrierDependenciesImpliedByFIFO
         * followed by BarrierInfo::removeBarrierDependenciesImpliedByFIFO
         * No change is expected from the input graph.
         *
         *          0p1
         *           |
         *           b0
         *       /  /|\  \
         *    1p1  / | \  4p0
         *    .   /  |  \  .
         *    .  /   |   \ .
         *    2p1   /     5p0
         *    . \  /       |
         *    .   /       /
         *     3p1  \    /
         *        \  |  /
         *           b1
         *           |
         *          6p1
         */
        expectedBarrierMaps.taskUpdateBarriers = {
                {0},  // task 0
                {},   // task 1
                {1},  // task 2
                {1},  // task 3
                {},   // task 4
                {1},  // task 5
                {},   // task 6
        };

        expectedBarrierMaps.taskWaitBarriers = {
                {},   // task 0
                {0},  // task 1
                {0},  // task 2
                {0},  // task 3
                {0},  // task 4
                {0},  // task 5
                {1},  // task 6
        };
        break;
    case TestTarget::createBarrierDependenciesImpliedByFIFO:
        /**
         * Expected result
         * for BarrierInfo::createBarrierDependenciesImpliedByFIFO(0)
         *
         *            0p1
         *             |
         *            b0
         *         /  /|\  \
         *     1p1   / | \   4p0
         *     |    /  |  \  |
         *     b3  /   |   \ b2
         *     |  /   /     \|
         *     2p1   /      5p0
         *     |  \ /        |
         *     b4  /        /
         *     |  /  \     /
         *     3p1    |   /
         *         \  |  /
         *            b1
         *            |
         *           6p1
         */
        expectedBarrierMaps.taskUpdateBarriers = {
                {0},     // task 0
                {3},     // task 1
                {1, 4},  // task 2
                {1},     // task 3
                {2},     // task 4
                {1},     // task 5
                {},      // task 6
        };

        expectedBarrierMaps.taskWaitBarriers = {
                {},      // task 0
                {0},     // task 1
                {0, 3},  // task 2
                {0, 4},  // task 3
                {0},     // task 4
                {0, 2},  // task 5
                {1},     // task 6
        };
        break;
    default:
        VPUX_THROW("Unsupported target for this test");
    }

    fillProducersAndConsumers(expectedBarrierMaps);
    return expectedBarrierMaps;
}

/**
 *
 *    0p1
 *   / . \
 *  b0 .  b1
 *   \ . /
 *    1p1
 *
 */
BarrierInfoTest::BarrierMaps barriersWithFIFOdependenciesNPU40XXconfig6(
        mlir::MLIRContext* ctx, mlir::OwningOpRef<mlir::ModuleOp>& module,
        TestTarget target = TestTarget::optimizeBarriersWithFIFOdeps) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>} {
            IE.PipelineOptions @Options {
            IE.Option @VPU.ReduceSupported : false
            IE.Option @VPU.AutoPaddingODU : false
            IE.Option @VPU.BarrierMaxVariantSum : 64
            IE.Option @VPU.BarrierMaxVariantCount : 128
            }
            IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
            IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
            IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
            IE.ExecutorResource 2 of @SHAVE_ACT
            IE.ExecutorResource 1 of @DPU
            }
            IE.ExecutorResource 1 of @M2I
            IE.ExecutorResource 2 of @DMA_NN
            IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}

            func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
                %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
                %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

                VPURT.Task updates (%bar0, %bar1:  !VPURT.Barrier, !VPURT.Barrier)  {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0, %bar1: !VPURT.Barrier, !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                return %arg1: memref<1x3x64x64xf16, @DDR>
            }
        }
    )";

    module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    VPUX_THROW_UNLESS(module.get() != nullptr, "Cannot extract module from input IR");

    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    switch (target) {
    case TestTarget::optimizeBarriersWithFIFOdeps:
        /**
         * Expected result
         * for BarrierInfo::optimizeBarriers(false, true)
         *
         *        0p1
         *       / .
         *      b0 .
         *       \ .
         *        1p1
         *
         * Redundant barrier b1 is removed in regular optimization but b0 is not removed in this optimization.
         * This is done separately by logic in DMABarrierOptimizationPass.
         *
         */
        expectedBarrierMaps.taskUpdateBarriers = {
                {0},  // task 0
                {},   // task 1
        };

        expectedBarrierMaps.taskWaitBarriers = {
                {},   // task 0
                {0},  // task 1
        };
        break;
    default:
        VPUX_THROW("Unsupported target for this test");
    }

    return expectedBarrierMaps;
}

/**
 *
 *        0p1               \
 *        |                 |
 *        b0                |
 *       /  \               |
 *     1p1   2p0            |  block: 0, size: 4 tasks
 *       \  /               |
 *        b1                |
 *        |                 |
 *        3p0 -- sync task  /
 *        |
 *        b2                \
 *       /  \               |
 * 4(shave)  5p0            |
 *       \  /               |
 *        b3                |
 *        |                 |
 *        6 (shave)         |
 *        |                 |  block: 1, size: 5 tasks
 *        b4                |
 *        |                 |
 *       7p1                |
 *        |                 |
 *        b5                |
 *        |                 |
 *       8p0  -- sync task  /
 *        |
 *        b6                \
 *       /  \               |  block: 2, size: 2 tasks
 * 9(shave)  10p0           /
 *
 *
 */
SmallVector<size_t> variableGraphSplitBlockSizeNPU40XXconfig(
        mlir::MLIRContext* ctx, mlir::OwningOpRef<mlir::ModuleOp>& module,
        TestTarget target = TestTarget::optimizeBarriersWithFIFOdeps) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>} {
            IE.PipelineOptions @Options {
            IE.Option @VPU.ReduceSupported : false
            IE.Option @VPU.AutoPaddingODU : false
            IE.Option @VPU.BarrierMaxVariantSum : 64
            IE.Option @VPU.BarrierMaxVariantCount : 128
            }
            IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
            IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
            IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
            IE.ExecutorResource 2 of @SHAVE_ACT
            IE.ExecutorResource 1 of @DPU
            }
            IE.ExecutorResource 1 of @M2I
            IE.ExecutorResource 2 of @DMA_NN
            IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}

            module @VPU.SW {
                func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE }
                func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
            }

            func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
                %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
                %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

                VPURT.Task updates(%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {"sync-task"} {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }

                VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }

                VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier)  attributes {"sync-task"} {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar6: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }

                VPURT.Task waits(%bar6: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                return %arg1: memref<1x3x64x64xf16, @DDR>
            }
        }
    )";

    module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    VPUX_THROW_UNLESS(module.get() != nullptr, "Cannot extract module from input IR");

    SmallVector<size_t> expectedResult;
    switch (target) {
    case TestTarget::getControlGraphBlockIndex: {
        SmallVector<size_t> expectedBlockIds = {
                0,  // task 0
                0,  // task 1
                0,  // task 2
                0,  // task 3
                1,  // task 4
                1,  // task 5
                1,  // task 6
                1,  // task 7
                1,  // task 8
                2,  // task 9
                2,  // task 10
        };
        expectedResult = expectedBlockIds;
        break;
    }
    case TestTarget::isSyncPoint: {
        SmallVector<size_t> expectedSyncPoints = {
                0,  // task 0
                0,  // task 1
                0,  // task 2
                1,  // task 3 - sync point
                0,  // task 4
                0,  // task 5
                0,  // task 6
                0,  // task 7
                1,  // task 8 - sync point
                0,  // task 9
                0,  // task 10
        };
        expectedResult = expectedSyncPoints;
        break;
    }
    case TestTarget::getControlGraphBlockTaskRangeFirstBlock: {
        SmallVector<size_t> tasksFromFirstBlock = {
                0,  // task 0
                1,  // task 1
                2,  // task 2
                3,  // task 3 - sync point
        };
        expectedResult = tasksFromFirstBlock;
        break;
    }
    case TestTarget::getControlGraphBlockTaskRangeMidBlock: {
        SmallVector<size_t> tasksFromMidBlock = {
                4,  // task 4
                5,  // task 5
                6,  // task 6
                7,  // task 7
                8,  // task 8 - sync point
        };
        expectedResult = tasksFromMidBlock;
        break;
    }
    case TestTarget::getControlGraphBlockTaskRangeLastBlock: {
        SmallVector<size_t> tasksFromLastBlock = {
                9,   // task 9
                10,  // task 10
        };
        expectedResult = tasksFromLastBlock;
        break;
    }

    default:
        VPUX_THROW("Unsupported target for this test");
    }

    return expectedResult;
}

/**
 * Test conversions of update barrier (wait barrier) map to barrier producers (consumers) map.
 */
TEST_F(BarrierInfoTests, fillProducersAndConsumers) {
    auto [barrierConfig, expectedResult] = redundantProducerConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    BarrierInfoTest::BarrierMaps testMaps = barrierConfig;
    fillProducersAndConsumers(testMaps);
    checkBarrierMaps(barrierConfig, testMaps);

    std::tie(barrierConfig, expectedResult) = redundantConsumerConfig();
    testMaps = barrierConfig;
    fillProducersAndConsumers(testMaps);
    checkBarrierMaps(barrierConfig, testMaps);
}

/*
 * Test BarrierInfo::optimizeBarrierProducers
 */
TEST_F(BarrierInfoTests, optimizeBarrierProducers) {
    auto [barrierConfig, expectedResult] = redundantProducerConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    auto testResult = barrierInfoTest.optimizeBarrierProducers(/* blockIdx */ 0);
    checkBarrierMaps(expectedResult, testResult);
}

/*
 * Test BarrierInfo::optimizeBarriersWithSameProducers
 */
TEST_F(BarrierInfoTests, optimizeBarriersWithSameProducers) {
    auto [barrierConfig, expectedResult] = sameProducerDifferentConsumersConfigA();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    barrierInfoTest.setMaxVariantCountPerBarrier(64);
    auto testResult = barrierInfoTest.optimizeBarriersWithSameProducers(/* blockIdx */ 0);
    checkBarrierMaps(expectedResult, testResult);

    std::tie(barrierConfig, expectedResult) = sameProducerDifferentConsumersConfigB(false);
    barrierInfoTest.initializeBarrierMaps(barrierConfig);
    testResult = barrierInfoTest.optimizeBarriersWithSameProducers(/* blockIdx */ 0);
    checkBarrierMaps(expectedResult, testResult);

    // perform two-stage optimization
    std::tie(barrierConfig, expectedResult) = sameProducerDifferentConsumersConfigB(true);
    barrierInfoTest.initializeBarrierMaps(barrierConfig);
    testResult = barrierInfoTest.optimizeBarriers();
    checkBarrierMaps(expectedResult, testResult);
}

/*
 * Test BarrierInfo::optimizeBarrierConsumers
 */
TEST_F(BarrierInfoTests, optimizeBarrierConsumers) {
    auto [barrierConfig, expectedResult] = redundantConsumerConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    auto testResult = barrierInfoTest.optimizeBarrierConsumers(/* blockIdx */ 0);
    checkBarrierMaps(expectedResult, testResult);
}

TEST_F(BarrierInfoTests, optimizeBarrierConsumersWithTwoTaskBlocks) {
    std::vector<size_t> blocksToOptimize = {0, 1};
    auto [barrierConfig, expectedResult] = redundantConsumersWithTwoBlockTaskSplitConfig(blocksToOptimize);
    BarrierInfoTest barrierInfoTest(barrierConfig);
    auto testResult = barrierInfoTest.optimizeBarrierConsumers(/* blockIdx */ 0);
    testResult = barrierInfoTest.optimizeBarrierConsumers(/* blockIdx */ 1);
    checkBarrierMaps(expectedResult, testResult);

    // optimize only block 1 in the same graph
    blocksToOptimize = std::vector<size_t>({1});
    std::tie(barrierConfig, expectedResult) = redundantConsumersWithTwoBlockTaskSplitConfig(blocksToOptimize);
    barrierInfoTest.initializeBarrierMaps(barrierConfig);
    testResult = barrierInfoTest.optimizeBarrierConsumers(/* blockIdx */ 1);
    checkBarrierMaps(expectedResult, testResult);
}

/**
 * Test BarrierInfo::optimizeBarriers
 *
 */
TEST_F(BarrierInfoTests, optimizeParallelBlocks) {
    auto [barrierConfig, expectedResult] = parallelBlocksConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    barrierInfoTest.setMaxVariantCountPerBarrier(64);
    auto testResult = barrierInfoTest.optimizeBarriers();
    checkBarrierMaps(expectedResult, testResult);
}

TEST_F(BarrierInfoTests, optimizeParallelBlocksWithTooFewVariantsPerBarrier) {
    auto [barrierConfig, expectedResult] = parallelBlocksConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    barrierInfoTest.setMaxVariantCountPerBarrier(4);
    auto testResult = barrierInfoTest.optimizeBarriers();
    // we do not expect optimized result when there are insufficient slots
    checkBarrierMaps(barrierConfig, testResult);

    // force optimization without slot count checks
    testResult = barrierInfoTest.optimizeBarriers(/* checkValidSlotCount */ false);
    checkBarrierMaps(expectedResult, testResult);
}

TEST_F(MLIR_BarrierInfoTests, optimizeBarriersWithFIFOdependenciesNPU40XX) {
    mlir::MLIRContext ctx(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto expectedResult = barriersWithFIFOdependenciesNPU40XXconfig(&ctx, module);
    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    VPUX_THROW_UNLESS(funcOp != nullptr, "Cannot extract funcOp from module");

    BarrierInfoTest barrierInfoTest(funcOp);
    auto testResult =
            barrierInfoTest.optimizeBarriers(/* checkValidSlotCount  */ false, /* considerTaskFifoDependency */ true);

    checkBarrierMaps(expectedResult, testResult, /* checkUpdateAndWaitBarriers */ true,
                     /* checkProducersAndConsumers */ true);
}

TEST_F(MLIR_BarrierInfoTests, optimizeBarriersWithFIFOdependencies2NPU40XX) {
    mlir::MLIRContext ctx(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto expectedResult = barriersWithFIFOdependenciesNPU40XXconfig2(&ctx, module);
    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    VPUX_THROW_UNLESS(funcOp != nullptr, "Cannot extract funcOp from module");

    BarrierInfoTest barrierInfoTest(funcOp);
    auto testResult =
            barrierInfoTest.optimizeBarriers(/* checkValidSlotCount  */ false, /* considerTaskFifoDependency */ true);

    checkBarrierMaps(expectedResult, testResult, /* checkUpdateAndWaitBarriers */ true,
                     /* checkProducersAndConsumers */ false);
}

TEST_F(MLIR_BarrierInfoTests, optimizeBarriersWithFIFOdependencies3NPU40XX) {
    mlir::MLIRContext ctx(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto expectedResult =
            barriersWithFIFOdependenciesNPU40XXconfig3(&ctx, module, TestTarget::optimizeBarriersWithFIFOdeps);
    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    VPUX_THROW_UNLESS(funcOp != nullptr, "Cannot extract funcOp from module");

    BarrierInfoTest barrierInfoTest(funcOp);
    auto testResult =
            barrierInfoTest.optimizeBarriers(/* checkValidSlotCount  */ false, /* considerTaskFifoDependency */ true);

    checkBarrierMaps(expectedResult, testResult, /* checkUpdateAndWaitBarriers */ true,
                     /* checkProducersAndConsumers */ true);
}

TEST_F(MLIR_BarrierInfoTests, createBarrierDependenciesImpliedByFifoNPU40XX) {
    mlir::MLIRContext ctx(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto expectedResult = barriersWithFIFOdependenciesNPU40XXconfig3(
            &ctx, module, TestTarget::createBarrierDependenciesImpliedByFIFO);
    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    VPUX_THROW_UNLESS(funcOp != nullptr, "Cannot extract funcOp from module");

    BarrierInfoTest barrierInfoTest(funcOp);
    barrierInfoTest.buildTaskQueueTypeMap();
    auto createdFifoDepsCount = barrierInfoTest.createBarrierDependenciesImpliedByFIFO(0);
    auto testResult = barrierInfoTest.getBarrierMaps();

    checkBarrierMaps(expectedResult, testResult, /* checkUpdateAndWaitBarriers */ true,
                     /* checkProducersAndConsumers */ true);

    EXPECT_EQ(createdFifoDepsCount, 6);
}

TEST_F(MLIR_BarrierInfoTests, createAndRemoveBarrierDependenciesImpliedByFifoNPU40XX) {
    mlir::MLIRContext ctx(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto expectedResult = barriersWithFIFOdependenciesNPU40XXconfig3(
            &ctx, module, TestTarget::createAndRemoveBarrierDependenciesImpliedByFIFO);
    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    VPUX_THROW_UNLESS(funcOp != nullptr, "Cannot extract funcOp from module");

    BarrierInfoTest barrierInfoTest(funcOp);
    barrierInfoTest.buildTaskQueueTypeMap();
    auto createdFifoDepsCount = barrierInfoTest.createBarrierDependenciesImpliedByFIFO(0);
    auto removedFifoDepsCount = barrierInfoTest.removeBarrierDependenciesImpliedByFIFO();
    auto testResult = barrierInfoTest.getBarrierMaps();

    checkBarrierMaps(expectedResult, testResult, /* checkUpdateAndWaitBarriers */ true,
                     /* checkProducersAndConsumers */ false);

    EXPECT_EQ(createdFifoDepsCount, removedFifoDepsCount);
}

TEST_F(MLIR_BarrierInfoTests, optimizeBarriersWithFIFOdependencies4NPU40XX) {
    mlir::MLIRContext ctx(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto expectedResult =
            barriersWithFIFOdependenciesNPU40XXconfig6(&ctx, module, TestTarget::optimizeBarriersWithFIFOdeps);
    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    VPUX_THROW_UNLESS(funcOp != nullptr, "Cannot extract funcOp from module");

    BarrierInfoTest barrierInfoTest(funcOp);
    auto testResult =
            barrierInfoTest.optimizeBarriers(/* checkValidSlotCount  */ false, /* considerTaskFifoDependency */ true);

    checkBarrierMaps(expectedResult, testResult, /* checkUpdateAndWaitBarriers */ true,
                     /* checkProducersAndConsumers */ false);
}

TEST_F(MLIR_BarrierInfoTests, variableGraphSplitBlockSizeNPU40XX) {
    mlir::MLIRContext ctx(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;

    // test task block indexes
    auto expectedResult = variableGraphSplitBlockSizeNPU40XXconfig(&ctx, module, TestTarget::getControlGraphBlockIndex);
    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    VPUX_THROW_UNLESS(funcOp != nullptr, "Cannot extract funcOp from module");

    BarrierInfoTest barrierInfoTest(funcOp);
    for (auto taskInd : irange(barrierInfoTest.getNumOfTasks())) {
        auto blockInd = barrierInfoTest.getControlGraphBlockIndex(taskInd);

        EXPECT_EQ(expectedResult[taskInd], blockInd);
    }

    EXPECT_TRUE(barrierInfoTest.verifyControlGraphSplit());

    // test sync point locations
    expectedResult = variableGraphSplitBlockSizeNPU40XXconfig(&ctx, module, TestTarget::isSyncPoint);
    for (auto taskInd : irange(barrierInfoTest.getNumOfTasks())) {
        bool isSyncTask = barrierInfoTest.isSyncPoint(taskInd);

        EXPECT_EQ(static_cast<bool>(expectedResult[taskInd]), isSyncTask);
    }

    // test first block task ranges
    expectedResult =
            variableGraphSplitBlockSizeNPU40XXconfig(&ctx, module, TestTarget::getControlGraphBlockTaskRangeFirstBlock);
    auto [blockFirstTaskInd, blockLastTaskInd] = barrierInfoTest.getControlGraphBlockTaskRange(
            0, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);

    EXPECT_EQ(expectedResult.front(), blockFirstTaskInd);
    EXPECT_EQ(expectedResult.back(), blockLastTaskInd);
    EXPECT_EQ(expectedResult.size(), blockLastTaskInd - blockFirstTaskInd + 1);

    std::tie(blockFirstTaskInd, blockLastTaskInd) = barrierInfoTest.getControlGraphBlockTaskRange(
            0, /* blockStartSyncPoint */ false, /* blockEndSyncPoint */ true);
    EXPECT_EQ(expectedResult.front(), blockFirstTaskInd);
    EXPECT_EQ(expectedResult.back(), blockLastTaskInd);
    EXPECT_EQ(expectedResult.size(), blockLastTaskInd - blockFirstTaskInd + 1);

    // test middle block task ranges
    expectedResult =
            variableGraphSplitBlockSizeNPU40XXconfig(&ctx, module, TestTarget::getControlGraphBlockTaskRangeMidBlock);
    std::tie(blockFirstTaskInd, blockLastTaskInd) = barrierInfoTest.getControlGraphBlockTaskRange(
            1, /* blockStartSyncPoint */ false, /* blockEndSyncPoint */ true);
    EXPECT_EQ(expectedResult.front(), blockFirstTaskInd);
    EXPECT_EQ(expectedResult.back(), blockLastTaskInd);
    EXPECT_EQ(expectedResult.size(), blockLastTaskInd - blockFirstTaskInd + 1);

    std::tie(blockFirstTaskInd, blockLastTaskInd) = barrierInfoTest.getControlGraphBlockTaskRange(
            1, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ false);
    EXPECT_EQ(expectedResult.front() - 1, blockFirstTaskInd);
    EXPECT_EQ(expectedResult.back() - 1, blockLastTaskInd);
    EXPECT_EQ(expectedResult.size(), blockLastTaskInd - blockFirstTaskInd + 1);

    // test last block task ranges
    expectedResult =
            variableGraphSplitBlockSizeNPU40XXconfig(&ctx, module, TestTarget::getControlGraphBlockTaskRangeLastBlock);
    std::tie(blockFirstTaskInd, blockLastTaskInd) = barrierInfoTest.getControlGraphBlockTaskRange(
            2, /* blockStartSyncPoint */ false, /* blockEndSyncPoint */ true);
    EXPECT_EQ(expectedResult.front(), blockFirstTaskInd);
    EXPECT_EQ(expectedResult.back(), blockLastTaskInd);
    EXPECT_EQ(expectedResult.size(), blockLastTaskInd - blockFirstTaskInd + 1);

    std::tie(blockFirstTaskInd, blockLastTaskInd) = barrierInfoTest.getControlGraphBlockTaskRange(
            2, /* blockStartSyncPoint */ false, /* blockEndSyncPoint */ false);
    EXPECT_EQ(expectedResult.front(), blockFirstTaskInd);
    EXPECT_EQ(expectedResult.back(), blockLastTaskInd);
    EXPECT_EQ(expectedResult.size(), blockLastTaskInd - blockFirstTaskInd + 1);
}

/**
 * HW FIFO A(DMA): t0 t1 t3
 * HW FIFO B(DPU): t2 t4 t5 t6
 *
 *      t0(A)
 *     /   \
 *    b0   b1
 *    |    |
 *  t1(A) t2(B)
 *    |    |
 *    |    |
 *    |    |
 *  t3(A) t4(B)
 *     \   /
 *       b2
 *       |
 *      t5(B)
 *       |
 *       b3
 *       |
 *      t6(B)
 */

BarrierInfoTest::BarrierMaps graphToCheckControlPathsWithFifo() {
    BarrierInfoTest::BarrierMaps barrierMapsConfig;

    barrierMapsConfig.taskUpdateBarriers = {
            {0, 1},  // task 0
            {},      // task 1
            {},      // task 2
            {3},     // task 3
            {3},     // task 4
            {4},     // task 5
            {}       // task 6
    };

    barrierMapsConfig.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2
            {},   // task 3
            {},   // task 4
            {3},  // task 5
            {4}   // task 6
    };

    fillProducersAndConsumers(barrierMapsConfig);
    barrierMapsConfig.nTasks = barrierMapsConfig.taskUpdateBarriers.size();
    barrierMapsConfig.nBarriers = barrierMapsConfig.barrierProducerMap.size();

    const VPURT::TaskQueueType dmaType{VPU::ExecutorKind::DMA_NN, 0};
    const VPURT::TaskQueueType dpuType{VPU::ExecutorKind::DPU, 0};
    barrierMapsConfig.taskQueueTypeMap[dmaType] = {0, 1, 3};
    barrierMapsConfig.taskQueueTypeMap[dpuType] = {2, 4, 5, 6};

    return barrierMapsConfig;
}

TEST_F(BarrierInfoTests, CheckBarrierReuseWithDepsThroughFifo) {
    auto barrierAndFifoConfig = graphToCheckControlPathsWithFifo();

    BarrierInfoTest barrierInfoTest(barrierAndFifoConfig);

    const auto& [taskControlMap, _] = barrierInfoTest.buildTaskControlMap(0, /* considerTaskFifoDependency */ true);

    EXPECT_TRUE(barrierInfoTest.controlPathExistsBetweenTasksInSameBlock(taskControlMap, 1, 3));
    EXPECT_FALSE(barrierInfoTest.controlPathExistsBetweenTasksInSameBlock(taskControlMap, 2, 3));
    EXPECT_TRUE(barrierInfoTest.controlPathExistsBetweenTasksInSameBlock(taskControlMap, 2, 4));
    EXPECT_TRUE(barrierInfoTest.controlPathExistsBetweenTasksInSameBlock(taskControlMap, 0, 6));
    EXPECT_TRUE(barrierInfoTest.controlPathExistsBetweenTasksInSameBlock(taskControlMap, 1, 6));
    EXPECT_TRUE(barrierInfoTest.controlPathExistsBetweenTasksInSameBlock(taskControlMap, 2, 6));
}

/**
 * HW FIFO DMA port-0: 6 7 8 9 10 11
 * HW FIFO DMA port-1: 12 13 14 15 16 17
 * HW FIFO SW: 0 1 2 3 4 5
 *
 *    0-5  12-15  6-11
 *   (SW)  (DMA1) (DMA0)
 *       \   |   / |
 *        \  |  /  |
 *           b0   b1
 *              x (fully connected)
 *           16   17
 *
 */

void parallelWaitBarriersIRconfig(mlir::MLIRContext* ctx, mlir::OwningOpRef<mlir::ModuleOp>& module) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>} {
            IE.PipelineOptions @Options {
            IE.Option @VPU.ReduceSupported : false
            IE.Option @VPU.AutoPaddingODU : false
            IE.Option @VPU.BarrierMaxVariantSum : 64
            IE.Option @VPU.BarrierMaxVariantCount : 128
            }
            IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
            IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
            IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
            IE.ExecutorResource 2 of @SHAVE_ACT
            IE.ExecutorResource 1 of @DPU
            }
            IE.ExecutorResource 1 of @M2I
            IE.ExecutorResource 2 of @DMA_NN
            IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}

            module @VPU.SW {
                func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE }
                func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
            }

            func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
                %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
                %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
                %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }
                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }
                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }
                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }
                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }
                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
                    }
                }

                VPURT.Task updates (%bar0, %bar1:  !VPURT.Barrier, !VPURT.Barrier)  {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }
                VPURT.Task updates (%bar0, %bar1:  !VPURT.Barrier, !VPURT.Barrier)  {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }
                VPURT.Task updates (%bar0, %bar1:  !VPURT.Barrier, !VPURT.Barrier)  {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }
                VPURT.Task updates (%bar0, %bar1:  !VPURT.Barrier, !VPURT.Barrier)  {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }
                VPURT.Task updates (%bar0, %bar1:  !VPURT.Barrier, !VPURT.Barrier)  {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }
                VPURT.Task updates (%bar0, %bar1:  !VPURT.Barrier, !VPURT.Barrier)  {
                    VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }
                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }
                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }
                VPURT.Task updates (%bar0: !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                VPURT.Task waits(%bar0, %bar1: !VPURT.Barrier, !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }
                VPURT.Task waits(%bar0, %bar1: !VPURT.Barrier, !VPURT.Barrier) {
                    VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
                }

                return %arg1: memref<1x3x64x64xf16, @DDR>
            }
        }
    )";

    module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    VPUX_THROW_UNLESS(module.get() != nullptr, "Cannot extract module from input IR");
}

TEST_F(MLIR_BarrierInfoTests, createLegalVariantBatchesByExecutorType) {
    mlir::MLIRContext ctx(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    parallelWaitBarriersIRconfig(&ctx, module);
    auto funcOp = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    VPUX_THROW_UNLESS(funcOp != nullptr, "Cannot extract funcOp from module");

    BarrierInfoTest barrierInfoTest(funcOp);

    BarrierInfo::TaskSet& parallelProducers = barrierInfoTest.getBarrierProducers(0);
    llvm::set_union(parallelProducers, barrierInfoTest.getBarrierProducers(1));

    auto legalVariantBatches =
            barrierInfoTest.createLegalVariantBatches(parallelProducers, 8, /* considerTaskExecutorType */ true);
    SmallVector<SmallVector<size_t>> expectedVariantBatches = {{0, 1, 2, 3, 4, 5},            // SW
                                                               {6, 7, 8, 9, 10, 11, 12, 13},  // DMA
                                                               {14, 15}};                     // DMA
    EXPECT_EQ(barrierInfoTest.toTaskVec(legalVariantBatches), expectedVariantBatches);

    legalVariantBatches =
            barrierInfoTest.createLegalVariantBatches(parallelProducers, 8, /* considerTaskExecutorType */ false);
    expectedVariantBatches = {
            {0, 1, 2, 3, 4, 5, 6, 7},
            {8, 9, 10, 11, 12, 13, 14, 15},
    };
    EXPECT_EQ(barrierInfoTest.toTaskVec(legalVariantBatches), expectedVariantBatches);
}

/**
 * Correct out-of-block dependencies according to graph-split constraints
 *
 * Test should remove out-of-block connection to consumer
 *         /------------------------------------------------------------------------------\
 *   0 - b0 - 1 - b1 - 2 (sync) - b2 - 3 - b3 - 4 - b4 - 5 (sync) - b5 - 6 - b6 - 7 - b7 - 8
 *
 * since task 8 already waits for b7
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> outOfBlockWaitDependencyConfig() {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2 - sync task
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5 - sync task
            {6},  // task 6
            {7},  // task 7
            {},   // task 8
    };

    inputBarrierMaps.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2 - sync task
            {2},     // task 3
            {3},     // task 4
            {4},     // task 5 - sync task
            {5},     // task 6
            {6},     // task 7
            {0, 7},  // task 8
    };

    fillProducersAndConsumers(inputBarrierMaps);
    inputBarrierMaps.nTasks = inputBarrierMaps.taskUpdateBarriers.size();
    inputBarrierMaps.nBarriers = inputBarrierMaps.barrierProducerMap.size();
    inputBarrierMaps.syncTasksIds = {2, 5};

    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2 - sync task
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5 - sync task
            {6},  // task 6
            {7},  // task 7
            {},   // task 8
    };

    expectedBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2 - sync task
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5 - sync task
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
    };

    fillProducersAndConsumers(expectedBarrierMaps);

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

TEST_F(BarrierInfoTests, fixOutOfBlockWaitDependency) {
    auto [barrierConfig, expectedResult] = outOfBlockWaitDependencyConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);

    EXPECT_FALSE(barrierInfoTest.verifyControlGraphSplit());

    BarrierInfo::TaskSet tasks;
    tasks.insert(8);
    barrierInfoTest.adjustTasksDependenciesToGraphSplitConstraints(tasks);
    auto testResult = barrierInfoTest.getBarrierMaps();
    checkBarrierMaps(expectedResult, testResult);

    EXPECT_TRUE(barrierInfoTest.verifyControlGraphSplit());
}

/**
 * Correct out-of-block dependencies according to graph-split constraints
 *
 * Test should remove out-of-block connection to consumer sync-point
 *         /--------------------------------------------\
 *   0 - b0 - 1 - b1 - 2 (sync) - b2 - 3 - b3 - 4 - b4 - 5 (sync) - b5 - 6 - b6 - 7 - b7 - 8
 *
 * because sync points were connected when graph was split
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> outOfBlockWaitDependencyToSyncTaskConfig() {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2 - sync task
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5 - sync task
            {6},  // task 6
            {7},  // task 7
            {},   // task 8
    };

    inputBarrierMaps.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2 - sync task
            {2},     // task 3
            {3},     // task 4
            {0, 4},  // task 5 - sync task
            {5},     // task 6
            {6},     // task 7
            {7},     // task 8
    };

    fillProducersAndConsumers(inputBarrierMaps);
    inputBarrierMaps.nTasks = inputBarrierMaps.taskUpdateBarriers.size();
    inputBarrierMaps.nBarriers = inputBarrierMaps.barrierProducerMap.size();
    inputBarrierMaps.syncTasksIds = {2, 5};

    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2 - sync task
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5 - sync task
            {6},  // task 6
            {7},  // task 7
            {},   // task 8
    };

    expectedBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2 - sync task
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5 - sync task
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
    };

    fillProducersAndConsumers(expectedBarrierMaps);

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

TEST_F(BarrierInfoTests, fixOutOfBlockWaitDependencyToSyncTask) {
    auto [barrierConfig, expectedResult] = outOfBlockWaitDependencyToSyncTaskConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);

    EXPECT_FALSE(barrierInfoTest.verifyControlGraphSplit());

    BarrierInfo::TaskSet tasks;
    tasks.insert(5);
    barrierInfoTest.adjustTasksDependenciesToGraphSplitConstraints(tasks);
    auto testResult = barrierInfoTest.getBarrierMaps();
    checkBarrierMaps(expectedResult, testResult);

    EXPECT_TRUE(barrierInfoTest.verifyControlGraphSplit());
}

/**
 * Correct out-of-block dependencies according to graph-split constraints
 *
 * Test should remove out-of-block connection to consumers
 *          /--------------------------------------------------------------------\
 *         /---------------------------------------------------------------------------\
 *   0 - b0 - 1 - b1 - 2 (sync) - b2 - 3 - b3 - 4 - b4 - 5 (sync) - b5 - 6 - b6 - 7     8
 *
 * and insert a connection to previous sync-point update barrier if the connection doesn't exist
 *
 *   0 - b0 - 1 - b1 - 2 (sync) - b2 - 3 - b3 - 4 - b4 - 5 (sync) - b5 - 6 - b6 - 7     8
 *                                                                   \-----------------/
 *
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> multipleOutOfBlockWaitDependenciesConfig() {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2 - sync task
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5 - sync task
            {6},  // task 6
            {7},  // task 7
            {},   // task 8
    };

    inputBarrierMaps.taskWaitBarriers = {
            {},      // task 0
            {0},     // task 1
            {1},     // task 2 - sync task
            {2},     // task 3
            {3},     // task 4
            {4},     // task 5 - sync task
            {5},     // task 6
            {0, 6},  // task 7
            {0},     // task 8
    };

    fillProducersAndConsumers(inputBarrierMaps);
    inputBarrierMaps.nTasks = inputBarrierMaps.taskUpdateBarriers.size();
    inputBarrierMaps.nBarriers = inputBarrierMaps.barrierProducerMap.size();
    inputBarrierMaps.syncTasksIds = {2, 5};

    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2 - sync task
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5 - sync task
            {6},  // task 6
            {7},  // task 7
            {},   // task 8
    };

    expectedBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2 - sync task
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5 - sync task
            {5},  // task 6
            {6},  // task 7
            {5},  // task 8
    };

    fillProducersAndConsumers(expectedBarrierMaps);

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

TEST_F(BarrierInfoTests, fixMultipleOutOfBlockWaitDependencies) {
    auto [barrierConfig, expectedResult] = multipleOutOfBlockWaitDependenciesConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);

    EXPECT_FALSE(barrierInfoTest.verifyControlGraphSplit());

    BarrierInfo::TaskSet tasks;
    tasks.insert(7);
    tasks.insert(8);
    barrierInfoTest.adjustTasksDependenciesToGraphSplitConstraints(tasks);
    auto testResult = barrierInfoTest.getBarrierMaps();
    checkBarrierMaps(expectedResult, testResult);

    EXPECT_TRUE(barrierInfoTest.verifyControlGraphSplit());
}

/**
 * Correct out-of-block dependencies according to graph-split constraints
 *
 * Test should remove redundant out-of-block connections from producers
 *                                      /-----------------------------------\
 *      /---------------------------------\
 *     /--------------------------------------------------------------------\
 *    /------------------------------------------------------------------------------\
 *   0 - b0 - 1 - b1 - 2 (sync) - b2 - 3 - b3 - 4 - b4 - 5 (sync) - b5 - 6 - b6 - 7 - b7 - 8
 *
 */
std::pair<BarrierInfoTest::BarrierMaps, BarrierInfoTest::BarrierMaps> outOfBlockUpdateDependencyConfig() {
    BarrierInfoTest::BarrierMaps inputBarrierMaps;
    BarrierInfoTest::BarrierMaps expectedBarrierMaps;
    inputBarrierMaps.taskUpdateBarriers = {
            {0, 3, 6, 7},  // task 0
            {1},           // task 1
            {2},           // task 2 - sync task
            {3, 6},        // task 3
            {4},           // task 4
            {5},           // task 5 - sync task
            {6},           // task 6
            {7},           // task 7
            {},            // task 8
    };

    inputBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2 - sync task
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5 - sync task
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
    };

    fillProducersAndConsumers(inputBarrierMaps);
    inputBarrierMaps.nTasks = inputBarrierMaps.taskUpdateBarriers.size();
    inputBarrierMaps.nBarriers = inputBarrierMaps.barrierProducerMap.size();
    inputBarrierMaps.syncTasksIds = {2, 5};

    expectedBarrierMaps.taskUpdateBarriers = {
            {0},  // task 0
            {1},  // task 1
            {2},  // task 2 - sync task
            {3},  // task 3
            {4},  // task 4
            {5},  // task 5 - sync task
            {6},  // task 6
            {7},  // task 7
            {},   // task 8
    };

    expectedBarrierMaps.taskWaitBarriers = {
            {},   // task 0
            {0},  // task 1
            {1},  // task 2 - sync task
            {2},  // task 3
            {3},  // task 4
            {4},  // task 5 - sync task
            {5},  // task 6
            {6},  // task 7
            {7},  // task 8
    };

    fillProducersAndConsumers(expectedBarrierMaps);

    return std::make_pair(inputBarrierMaps, expectedBarrierMaps);
}

TEST_F(BarrierInfoTests, fixOutOfBlockUpdateDependencies) {
    auto [barrierConfig, expectedResult] = outOfBlockUpdateDependencyConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);

    EXPECT_FALSE(barrierInfoTest.verifyControlGraphSplit());

    BarrierInfo::TaskSet tasks;
    tasks.insert(0);
    tasks.insert(3);
    barrierInfoTest.adjustTasksDependenciesToGraphSplitConstraints(tasks);
    auto testResult = barrierInfoTest.getBarrierMaps();
    checkBarrierMaps(expectedResult, testResult);

    EXPECT_TRUE(barrierInfoTest.verifyControlGraphSplit());
}
