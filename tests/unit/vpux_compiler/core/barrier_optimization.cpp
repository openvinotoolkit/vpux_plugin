//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"

#include <gtest/gtest.h>

using namespace vpux;
using BarrierInfoTests = ::testing::Test;

/**
 * Calculate barrier producers/consumers map from task update/wait barriers map
 */
void fillProducersAndConsumers(BarrierInfoTest::BarrierMaps& barrierMaps) {
    barrierMaps.barrierProducerMap.clear();
    barrierMaps.barrierConsumerMap.clear();

    // calculate barrier maximal ID
    llvm::DenseSet<size_t> barriers;
    for (auto taskBarriers : barrierMaps.taskUpdateBarriers) {
        barriers.insert(taskBarriers.begin(), taskBarriers.end());
    }
    for (auto taskBarriers : barrierMaps.taskWaitBarriers) {
        barriers.insert(taskBarriers.begin(), taskBarriers.end());
    }
    if (barriers.empty()) {
        return;
    }
    size_t maxBarrierID = *std::max_element(barriers.begin(), barriers.end());

    auto fillBarrierMap = [&](const BarrierMap& barriersPerTaskMap, BarrierMap& tasksPerBarrierMap) {
        // clear producers / consumers
        tasksPerBarrierMap.resize(maxBarrierID + 1);
        for (auto& tasks : tasksPerBarrierMap) {
            tasks.clear();
        }

        // fill-in tasks for each barrier
        for (auto taskBarriers : barriersPerTaskMap | indexed) {
            for (auto barrier : taskBarriers.value()) {
                tasksPerBarrierMap[barrier].push_back(taskBarriers.index());
            }
        }

        // sort maps
        for (auto& tasks : tasksPerBarrierMap) {
            llvm::sort(tasks);
        }
    };

    fillBarrierMap(barrierMaps.taskUpdateBarriers, barrierMaps.barrierProducerMap);
    fillBarrierMap(barrierMaps.taskWaitBarriers, barrierMaps.barrierConsumerMap);
}

void checkBarrierMaps(const BarrierInfoTest::BarrierMaps& expectedResult,
                      const BarrierInfoTest::BarrierMaps& testResult) {
    ASSERT_EQ(expectedResult.barrierConsumerMap, testResult.barrierConsumerMap);
    ASSERT_EQ(expectedResult.taskWaitBarriers, testResult.taskWaitBarriers);
    ASSERT_EQ(expectedResult.barrierProducerMap, testResult.barrierProducerMap);
    ASSERT_EQ(expectedResult.taskUpdateBarriers, testResult.taskUpdateBarriers);
}

/**
 * Configuration with 1 redundant producer
 *
 *        0                     0
 *       /  \                   |
 *       |   \                  |
 *      b0    \                b0
 *       | \   \                | \
 *       |   1  |               |   1
 *       |    \ |               |    \
 *       |     b1      =>       |     b1
 *       |      |               |      |
 *       2      |               2      |
 *       |     /                |     /
 *      bar2  /                bar2  /
 *       |   /                  |   /
 *       3--/                   3--/
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

    inputBarrierMaps.controlGraphBlockSize = 0;  // Control graph split is not done
    inputBarrierMaps.Ntasks = 4;
    inputBarrierMaps.Nbarriers = 3;

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
    inputBarrierMaps.controlGraphBlockSize = 0;  // Control graph split is not done
    inputBarrierMaps.Ntasks = 30;
    inputBarrierMaps.Nbarriers = 5;

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
    inputBarrierMaps.controlGraphBlockSize = 0;  // Control graph split is not done
    inputBarrierMaps.Ntasks = 3;
    inputBarrierMaps.Nbarriers = 2;

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
    inputBarrierMaps.controlGraphBlockSize = 0;  // Control graph split is not done
    inputBarrierMaps.Ntasks = 3;
    inputBarrierMaps.Nbarriers = 3;

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

    inputBarrierMaps.controlGraphBlockSize = 0;  // Control graph split is not done
    inputBarrierMaps.Ntasks = 5;
    inputBarrierMaps.Nbarriers = 3;

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
    inputBarrierMaps.controlGraphBlockSize = 5;  // Control graph split done in the middle of the graph.
                                                 // controlGraphBlockSize is an integer multiple of Ntasks
    inputBarrierMaps.Ntasks = 10;
    inputBarrierMaps.Nbarriers = 6;
    inputBarrierMaps.syncTasksIds = {4};

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
    auto optimizedResult = barrierInfoTest.optimizeBarrierProducers(/* blockIdx */ 0);
    checkBarrierMaps(expectedResult, optimizedResult);
}

/*
 * Test BarrierInfo::optimizeBarriersWithSameProducers
 */
TEST_F(BarrierInfoTests, optimizeBarriersWithSameProducers) {
    auto [barrierConfig, expectedResult] = sameProducerDifferentConsumersConfigA();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    barrierInfoTest.setMaxVariantCountPerBarrier(64);
    auto optimizedResult = barrierInfoTest.optimizeBarriersWithSameProducers(/* blockIdx */ 0);
    checkBarrierMaps(expectedResult, optimizedResult);

    std::tie(barrierConfig, expectedResult) = sameProducerDifferentConsumersConfigB(false);
    barrierInfoTest.initializeBarrierMaps(barrierConfig);
    optimizedResult = barrierInfoTest.optimizeBarriersWithSameProducers(/* blockIdx */ 0);
    checkBarrierMaps(expectedResult, optimizedResult);

    // perform two-stage optimization
    std::tie(barrierConfig, expectedResult) = sameProducerDifferentConsumersConfigB(true);
    barrierInfoTest.initializeBarrierMaps(barrierConfig);
    optimizedResult = barrierInfoTest.optimizeBarriers();
    checkBarrierMaps(expectedResult, optimizedResult);
}

/*
 * Test BarrierInfo::optimizeBarrierConsumers
 */
TEST_F(BarrierInfoTests, optimizeBarrierConsumers) {
    auto [barrierConfig, expectedResult] = redundantConsumerConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    auto optimizedResult = barrierInfoTest.optimizeBarrierConsumers(/* blockIdx */ 0);
    checkBarrierMaps(expectedResult, optimizedResult);
}

TEST_F(BarrierInfoTests, optimizeBarrierConsumersWithTwoTaskBlocks) {
    std::vector<size_t> blocksToOptimize = {0, 1};
    auto [barrierConfig, expectedResult] = redundantConsumersWithTwoBlockTaskSplitConfig(blocksToOptimize);
    BarrierInfoTest barrierInfoTest(barrierConfig);
    auto optimizedResult = barrierInfoTest.optimizeBarrierConsumers(/* blockIdx */ 0);
    optimizedResult = barrierInfoTest.optimizeBarrierConsumers(/* blockIdx */ 1);
    checkBarrierMaps(expectedResult, optimizedResult);

    // optimize only block 1 in the same graph
    blocksToOptimize = std::vector<size_t>({1});
    std::tie(barrierConfig, expectedResult) = redundantConsumersWithTwoBlockTaskSplitConfig(blocksToOptimize);
    barrierInfoTest.initializeBarrierMaps(barrierConfig);
    optimizedResult = barrierInfoTest.optimizeBarrierConsumers(/* blockIdx */ 1);
    checkBarrierMaps(expectedResult, optimizedResult);
}

/**
 * Test BarrierInfo::optimizeBarriers
 *
 */
TEST_F(BarrierInfoTests, optimizeParallelBlocks) {
    auto [barrierConfig, expectedResult] = parallelBlocksConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    barrierInfoTest.setMaxVariantCountPerBarrier(64);
    auto optimizedResult = barrierInfoTest.optimizeBarriers();
    checkBarrierMaps(expectedResult, optimizedResult);
}

TEST_F(BarrierInfoTests, optimizeParallelBlocksWithTooFewVariantsPerBarrier) {
    auto [barrierConfig, expectedResult] = parallelBlocksConfig();
    BarrierInfoTest barrierInfoTest(barrierConfig);
    barrierInfoTest.setMaxVariantCountPerBarrier(4);
    auto optimizedResult = barrierInfoTest.optimizeBarriers();
    // we do not expect optimized result when there are insufficient slots
    checkBarrierMaps(barrierConfig, optimizedResult);

    // force optimization without slot count checks
    optimizedResult = barrierInfoTest.optimizeBarriers(/* checkValidSlotCount */ false);
    checkBarrierMaps(expectedResult, optimizedResult);
}
