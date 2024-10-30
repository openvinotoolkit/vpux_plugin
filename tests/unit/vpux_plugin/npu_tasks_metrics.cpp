//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/reports/stats.hpp"
#include "vpux/utils/profiling/reports/tasklist.hpp"

#include <gtest/gtest.h>

#include <iostream>

using MetricsUnitTests = ::testing::Test;
using namespace vpux::profiling;

TaskInfo makeTask(uint64_t tm_start, uint64_t tm_stop, TaskInfo::ExecType execType = TaskInfo::ExecType::DMA) {
    TaskInfo taskInfo{"", "", execType, tm_start, tm_stop - tm_start};
    return taskInfo;
}

struct TasksDurations {
    uint64_t totDuration;
    uint64_t overlapTime;
    uint64_t idleTime;
    uint64_t sumOfDurations;
};

TasksDurations testTaskDurations(TaskList testTasks, TaskList refTasks) {
    TaskList reportedTasks(testTasks);
    reportedTasks.append(refTasks);

    details::TaskTrack track1;
    track1.insert(testTasks).coalesce();

    details::TaskTrack track2;
    track2.insert(refTasks).coalesce();

    TasksDurations stats;
    stats.totDuration = reportedTasks.getTotalDuration();
    stats.sumOfDurations = reportedTasks.getSumOfDurations();

    auto overlapAndIdleDur = track1.calculateOverlap(track2);
    stats.overlapTime = overlapAndIdleDur.first;
    stats.idleTime = overlapAndIdleDur.second;
    return stats;
}

/*
 * Test cross tracks overlap
 *
 *       time: 1 3    8      15
 * test tasks: ..*****..*****
 *  ref tasks: *....******...
 *
 * . - idle time
 * * - task time
 *
 * overlap duration - occurs in [6,8) and [12,12) and amounts to 4 time units
 * idle duration    - occurs in [2,3) and amounts to 1 time unit
 * total duration   - occurs in [1,15) and amounts to 14 time units
 * work duration    - sums up all work units done in both tracks and amounts to 17
 *
 */
TEST_F(MetricsUnitTests, TasksDistributionStats_testOverlapTest) {
    TaskList testTasks({makeTask(3, 8), makeTask(10, 15)});
    TaskList refTasks({makeTask(1, 2), makeTask(6, 12)});

    const auto stats = testTaskDurations(testTasks, refTasks);

    EXPECT_EQ(stats.overlapTime, 4);
    EXPECT_EQ(stats.idleTime, 1);
    EXPECT_EQ(stats.totDuration, 14);
    EXPECT_EQ(stats.sumOfDurations, 17);
}

TEST_F(MetricsUnitTests, TasksDistributionStats_testIntegerOverflow) {
    constexpr auto maxInt = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    constexpr auto maxUInt = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

    TaskList testTasks({makeTask(0, 8), makeTask(maxInt - 10, maxInt + 1)});
    TaskList testTasksUint({makeTask(0, 8), makeTask(maxUInt - 10, maxUInt + 1)});

    EXPECT_EQ(testTasks.getTotalDuration(), maxInt + 1);
    EXPECT_EQ(testTasksUint.getTotalDuration(), maxUInt + 1);

    TaskList testTasksS({makeTask(0, maxInt - 10), makeTask(1, 12)});
    TaskList testTasksSUint({makeTask(0, maxUInt - 10), makeTask(1, 12)});

    EXPECT_EQ(testTasksS.getSumOfDurations(), maxInt + 1);
    EXPECT_EQ(testTasksSUint.getSumOfDurations(), maxUInt + 1);

    constexpr auto maxU64 = std::numeric_limits<uint64_t>::max();
    constexpr auto execType = TaskInfo::ExecType::DMA;

    // this task list is valid, while not summ of all duration
    TaskList testTasksU64{{TaskInfo{"", "", execType, 0, 1}, TaskInfo{"", "", execType, maxU64 - 10, 10}}};

    EXPECT_EQ(testTasksU64.getSumOfDurations(), 11);
    EXPECT_EQ(testTasksU64.getTotalDuration(), maxU64);

    TaskList testTaskU64_2{{TaskInfo{"", "", execType, 0, 1}, TaskInfo{"", "", execType, 1, maxU64 - 1}}};
    EXPECT_EQ(testTaskU64_2.getSumOfDurations(), maxU64);

    // this task list have a task that end time is u64 overflow - so total duration cannot be computed
    TaskList testTasksU64_3{{TaskInfo{"", "", execType, 0, 1}, TaskInfo{"", "", execType, maxU64 - 10, 11}}};

    EXPECT_ANY_THROW(testTasksU64_3.getTotalDuration());
    EXPECT_EQ(testTasksU64_3.getSumOfDurations(), 12);

    // sum of durations cannot be computed in u64 precision
    TaskList testTasksU64_4{{TaskInfo{"", "", execType, 0, 11}, TaskInfo{"", "", execType, 10, maxU64 - 10}}};

    EXPECT_EQ(testTasksU64_4.getTotalDuration(), maxU64);
    EXPECT_ANY_THROW(testTasksU64_4.getSumOfDurations());
}
/*
 * Test overlap of tracks without parallel tasks
 *
 */
TEST_F(MetricsUnitTests, TasksDistributionStats_testDisjoint) {
    TaskList testTasks({makeTask(3, 8), makeTask(10, 15)});
    TaskList refTasks({makeTask(100, 200), makeTask(0, 1)});

    auto stats = testTaskDurations(testTasks, refTasks);

    EXPECT_EQ(stats.overlapTime, 0);
    EXPECT_EQ(stats.idleTime, 89);
    EXPECT_EQ(stats.totDuration, 200);
    EXPECT_EQ(stats.sumOfDurations, 111);
}

/**
 * Test behaviour for tracks containing zero-duration tasks.
 * Note that the total duration is obtained before tasks coalescence
 * which accounts for the zero duration tasks
 *
 *       time: 0123456
 * test tasks: ...|*.|
 *  ref tasks: *......
 *
 * . - idle time
 * | - zero duration tasks
 * * - non-zero duration tasks
 *
 * overlap duration - 0 as the overlap does not occur
 * idle duration    - occurs at [1,4) and amounts to 3 time units
 * total duration   - occurs in range [0,6) as the last task in test track has zero duration
 *      its end time is 6 and the total (eg. inference) duration amounts to 6
 * work duration    - sums up all work units done in both tracks and amounts to 2
 *
 */
TEST_F(MetricsUnitTests, TasksDistributionStats_testZeroDuration) {
    TaskList testTasks({makeTask(3, 3), makeTask(4, 5), makeTask(6, 6)});
    TaskList refTasks({makeTask(6, 6), makeTask(0, 1)});

    auto stats = testTaskDurations(testTasks, refTasks);

    EXPECT_EQ(stats.overlapTime, 0);
    EXPECT_EQ(stats.idleTime, 3);
    EXPECT_EQ(stats.totDuration, 6);
    EXPECT_EQ(stats.sumOfDurations, 2);
}

/**
 * Test tracks that self overlap.
 * Self-overlapping tasks are coalesced before testing the time overlap with reference track
 *
 *       time: 0123456789
 * test tasks: ....**..
 *           : .....**.
 *  ref tasks: ........*.
 *
 * . - idle time
 * * - task time
 *
 * overlap duration - 0 as the overlap does not occur between test and reference tasks
 * idle duration    - occurs in range [7,8) and amounts to 1 time unit
 * total duration   - occurs in range [4,9) and amounts to 5 time units
 * work duration    - sums up all work units done in both tracks and amounts to 5. Note that this can
 *      sum up to a larger value than the total duration due to concurrency. This can measure eg.
 *      NCE tiling efficiency.
 *
 */
TEST_F(MetricsUnitTests, TasksDistributionStats_testSelfOverlap1) {
    TaskList testTasks({makeTask(4, 6), makeTask(5, 7)});
    TaskList refTasks({makeTask(8, 9)});

    auto stats = testTaskDurations(testTasks, refTasks);

    EXPECT_EQ(stats.overlapTime, 0);
    EXPECT_EQ(stats.idleTime, 1);
    EXPECT_EQ(stats.totDuration, 5);
    EXPECT_EQ(stats.sumOfDurations, 5);
}

/**
 * Test tracks that self overlap.
 * Self-overlapping tasks are coalesced before testing the time overlap with reference track
 *
 *       time: 0123456789
 * test tasks: ....**..
 *           : ....**..
 *           : .....**.
 *  ref tasks: ........*.
 *
 * . - idle time
 * * - task time
 *
 * overlap duration - 0 as the overlap does not occur between test and reference tasks
 * idle duration    - occurs in range [7,8) and amounts to 1 time unit
 * total duration   - occurs in range [4,9) and amounts to 5 time units
 * work duration    - sums up all work units done in both tracks and amounts to 7.
 *
 */
TEST_F(MetricsUnitTests, TasksDistributionStats_testSelfOverlap2) {
    TaskList testTasks({makeTask(4, 6), makeTask(4, 6), makeTask(5, 7)});
    TaskList refTasks({makeTask(8, 9)});

    auto stats = testTaskDurations(testTasks, refTasks);

    EXPECT_EQ(stats.overlapTime, 0);
    EXPECT_EQ(stats.idleTime, 1);
    EXPECT_EQ(stats.totDuration, 5);
    EXPECT_EQ(stats.sumOfDurations, 7);
}

/**
 * Tests tasks statistics when one track is empty
 *
 *       time: 01234567
 * test tasks: ....**..
 *           : ....**..
 *           : .....**.
 *  ref tasks: ........
 *
 * . - idle time
 * * - task time
 *
 * overlap duration - 0 as the overlap does not occur
 * idle duration    - idle does not occur hence 0
 * total duration   - occurs in range [4,7) and amounts to 3
 * work duration    - sums up all work units done in both tracks and amounts to 6
 *
 */
TEST_F(MetricsUnitTests, TasksDistributionStats_testEmpty) {
    TaskList testTasks({makeTask(4, 6), makeTask(4, 6), makeTask(5, 7)});
    TaskList refTasks;

    auto stats = testTaskDurations(testTasks, refTasks);

    EXPECT_EQ(stats.overlapTime, 0);
    EXPECT_EQ(stats.idleTime, 0);
    EXPECT_EQ(stats.totDuration, 3);
    EXPECT_EQ(stats.sumOfDurations, 6);
}

/*
 * Test TaskTrack methods
 *
 */
TEST_F(MetricsUnitTests, TasksDistributionStats_testTaskTrackWorkloads) {
    TaskList testTasks({makeTask(3, 8), makeTask(10, 15)});
    details::TaskTrack track;
    auto totalDuration = track.insert(testTasks).getSumOfDurations();
    EXPECT_EQ(totalDuration, 20);
    totalDuration = track.coalesce().getSumOfDurations();
    EXPECT_EQ(totalDuration, 10);
}
