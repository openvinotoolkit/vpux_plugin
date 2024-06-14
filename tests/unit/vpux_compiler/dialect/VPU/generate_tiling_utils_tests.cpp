//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>
#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"

using namespace vpux;

using vpux::VPU::ArchKind;
using MLIR_VPU_Generate_Tiling = MLIR_UnitBase;

TEST_F(MLIR_VPU_Generate_Tiling, Calculate_Workload_Number) {
    SmallVector<Shape> shapesOf2Clusters{{1, 2048, 16, 16}, {1, 2032, 16, 16}};
    SmallVector<int64_t> supportedChannels{64, 32, 16};
    auto maxNumPerClusterFor2Clusters = VPU::getMaxWorkLoadNumberPerClusterForNCEWithSparseOutput(
            ArchKind::NPU37XX, shapesOf2Clusters, supportedChannels);
    // cluster-0: 32, 32,..., 32, 32
    // cluster-1: 32, 32,..., 32, 16
    EXPECT_EQ(maxNumPerClusterFor2Clusters, 64);

    SmallVector<Shape> shapesOf2ClustersWorseCase1{{1, 112, 16, 16}, {1, 112, 16, 16}};
    auto maxNumPerClusterFor2ClustersFor2ClustersWorseCase1 = VPU::getMaxWorkLoadNumberPerClusterForNCEWithSparseOutput(
            ArchKind::NPU37XX, shapesOf2ClustersWorseCase1, supportedChannels);
    // cluster-0: 16, 16, 16, 16, 16, 16, 16
    // cluster-1: 16, 16, 16, 16, 16, 16, 16
    EXPECT_EQ(maxNumPerClusterFor2ClustersFor2ClustersWorseCase1, 7);

    SmallVector<Shape> shapesOf2ClustersWorseCase2{{1, 80, 16, 16}, {1, 80, 16, 16}};
    auto maxNumPerClusterFor2ClustersFor2ClustersWorseCase2 = VPU::getMaxWorkLoadNumberPerClusterForNCEWithSparseOutput(
            ArchKind::NPU37XX, shapesOf2ClustersWorseCase2, supportedChannels);
    // cluster-0: 16, 16, 16, 16, 16
    // cluster-1: 16, 16, 16, 16, 16
    EXPECT_EQ(maxNumPerClusterFor2ClustersFor2ClustersWorseCase2, 5);

    SmallVector<Shape> shapesOf6Clusters{{1, 128, 16, 16}, {1, 128, 16, 16}, {1, 128, 16, 16},
                                         {1, 128, 16, 16}, {1, 128, 16, 16}, {1, 96, 16, 16}};
    auto maxNumPerClusterFor2ClustersFor6Clusters = VPU::getMaxWorkLoadNumberPerClusterForNCEWithSparseOutput(
            ArchKind::NPU40XX, shapesOf6Clusters, supportedChannels);
    // cluster-0: 64, 64   cluster-1: 64, 64 cluster-2: 64, 64
    // cluster-3: 64, 64   cluster-4: 64, 64 cluster-5: 64, 32
    EXPECT_EQ(maxNumPerClusterFor2ClustersFor6Clusters, 2);

    SmallVector<Shape> shapesOf6ClustersCase1{{1, 128, 16, 16}, {1, 128, 16, 16}, {1, 128, 16, 16},
                                              {1, 128, 16, 16}, {1, 128, 16, 16}, {1, 32, 16, 16}};
    auto maxNumPerClusterFor2ClustersFor6ClustersCase1 = VPU::getMaxWorkLoadNumberPerClusterForNCEWithSparseOutput(
            ArchKind::NPU40XX, shapesOf6Clusters, supportedChannels);
    // cluster-0: 64, 64   cluster-1: 64, 64 cluster-2: 64, 64
    // cluster-3: 64, 64   cluster-4: 64, 64 cluster-5: 32
    EXPECT_EQ(maxNumPerClusterFor2ClustersFor6ClustersCase1, 2);
}
