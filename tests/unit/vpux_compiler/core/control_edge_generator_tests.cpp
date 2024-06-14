//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/control_edge_generator.hpp"
#include "vpux/compiler/core/feasible_scheduler_utils.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_ControlEdgeGenerator = MLIR_UnitBase;

TEST(MLIR_ControlEdgeGenerator, TestMemOverlapEdges) {
    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(1, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(2, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(3, 0, 50, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(4, 51, 100, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(5, 0, 50, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(6, 51, 100, ScheduledOpOneResource::EResRelation::CONSUMER),
    };

    // For above configuration expected inserted memory control edges are:
    // 0 -> 1,2
    // 1,2 -> 3
    // 1,2 -> 4
    // 3 -> 5
    // 4 -> 6
    SmallVector<ControlEdge> expectedControlEdges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {1, 4}, {2, 4}, {3, 5}, {4, 6}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}

TEST(MLIR_ControlEdgeGenerator, TestMemOverlapEdgesWithSubViewTest1) {
    ScheduledOpOneResource::ResourceView resView0({{0}, {1}, {1}});
    ScheduledOpOneResource::ResourceView resView1({{1}, {1}, {1}});

    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(1, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView1),
            ScheduledOpOneResource(2, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView0),
            ScheduledOpOneResource(3, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView1)};

    // For above configuration expected inserted memory control edges are:
    // 0 -> 2
    // 1 -> 3
    SmallVector<ControlEdge> expectedControlEdges = {{0, 2}, {1, 3}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}

TEST(MLIR_ControlEdgeGenerator, TestMemOverlapEdgesWithSubViewTest2) {
    ScheduledOpOneResource::ResourceView resView0({{0}, {1}, {1}});
    ScheduledOpOneResource::ResourceView resView1({{1}, {1}, {1}});

    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(1, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView1),
            ScheduledOpOneResource(2, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView0),
            ScheduledOpOneResource(3, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView1),
            ScheduledOpOneResource(4, 50, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),

    };

    // For above configuration expected inserted memory control edges are:
    // 0 -> 2
    // 1 -> 3
    // 2,3 -> 4
    SmallVector<ControlEdge> expectedControlEdges = {{0, 2}, {1, 3}, {2, 4}, {3, 4}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}

TEST(MLIR_ControlEdgeGenerator, TestMemOverlapEdgesWithSubViewTest3) {
    ScheduledOpOneResource::ResourceView resView0({{0}, {1}, {1}});
    ScheduledOpOneResource::ResourceView resView1({{1}, {1}, {1}});

    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(1, 0, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(2, 0, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView1),
    };

    // For above configuration expected inserted memory control edges are:
    // 0 -> 1
    // 0 -> 2
    SmallVector<ControlEdge> expectedControlEdges = {{0, 1}, {0, 2}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}

TEST(MLIR_ControlEdgeGenerator, TestMemOverlapEdgesWithSubViewTest4) {
    ScheduledOpOneResource::ResourceView resView0({{0}, {1}, {1}});
    ScheduledOpOneResource::ResourceView resView1({{1}, {1}, {1}});

    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(1, 101, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(2, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView0),
            ScheduledOpOneResource(3, 0, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(4, 0, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView1),
    };

    // For above configuration expected inserted memory control edges are:
    // 0 -> 2
    // 1,2 -> 3
    // 1,2 -> 4
    SmallVector<ControlEdge> expectedControlEdges = {{0, 2}, {2, 3}, {1, 3}, {1, 4}, {2, 4}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}
