//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/subspaces.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(MLIR_SubSpacesTests, GetCoord) {
    const MemShape dims = {2, 4};
    const int64_t numSections = 8;

    const auto coord = subspace::getCoord(dims, numSections);
    ASSERT_EQ(coord.size(), dims.size());
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 0);
}

TEST(MLIR_SubSpacesTests, getOffset) {
    const auto elemSize = 8_Bit;
    const MemShape dims = {2, 4};
    const MemShape coord = {1, 0};
    const MemStrides strides = {dims[MemDim(1)] * elemSize, elemSize};

    const auto offset = subspace::getOffset(coord, strides);
    EXPECT_EQ(offset, 4 * elemSize);

    const MemShape coord2 = {1, 2};
    const auto offset2 = subspace::getOffset(coord2, strides);
    EXPECT_EQ(offset2, 4 * elemSize + 2 * elemSize);
}

TEST(MLIR_SubSpacesTests, Increment1Coord) {
    const MemShape dims = {2, 4};
    MemShape coord = {0, 0};

    subspace::increment1Coord(coord, dims);
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 1);

    subspace::increment1Coord(coord, dims);
    subspace::increment1Coord(coord, dims);
    subspace::increment1Coord(coord, dims);

    EXPECT_EQ(coord[MemDim(0)], 1);
    EXPECT_EQ(coord[MemDim(1)], 0);
}

TEST(MLIR_SubSpacesTests, IncrementNCoord) {
    const MemShape dims = {2, 4};
    MemShape coord = {0, 0};

    subspace::incrementNCoord(coord, dims, 2);
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 2);

    subspace::incrementNCoord(coord, dims, 2);
    EXPECT_EQ(coord[MemDim(0)], 1);
    EXPECT_EQ(coord[MemDim(1)], 0);
}

TEST(MLIR_SubSpacesTests, IncrementLine) {
    const MemShape dims = {4, 4, 2};
    MemShape coord = {0, 0, 0};

    subspace::incrementLine(coord, dims, MemDim(1));
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 0);
    EXPECT_EQ(coord[MemDim(2)], 1);

    subspace::incrementLine(coord, dims, MemDim(1));
    EXPECT_EQ(coord[MemDim(0)], 1);
    EXPECT_EQ(coord[MemDim(1)], 0);
    EXPECT_EQ(coord[MemDim(2)], 0);

    subspace::incrementLine(coord, dims, MemDim(1));
    subspace::incrementLine(coord, dims, MemDim(1));
    EXPECT_EQ(coord[MemDim(0)], 2);
    EXPECT_EQ(coord[MemDim(1)], 0);
    EXPECT_EQ(coord[MemDim(2)], 0);
}

TEST(MLIR_SubSpacesTests, IncrementPlane) {
    const MemShape dims = {2, 4, 8, 4};
    MemShape coord = {0, 0, 0, 0};

    subspace::incrementPlane(coord, dims, MemDim(1), MemDim(2));
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 0);
    EXPECT_EQ(coord[MemDim(2)], 0);
    EXPECT_EQ(coord[MemDim(3)], 1);

    subspace::incrementPlane(coord, dims, MemDim(2), MemDim(2));
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 0);
    EXPECT_EQ(coord[MemDim(2)], 0);
    EXPECT_EQ(coord[MemDim(3)], 2);

    subspace::incrementPlane(coord, dims, MemDim(2), MemDim(3));
    EXPECT_EQ(coord[MemDim(0)], 0);
    EXPECT_EQ(coord[MemDim(1)], 1);
    EXPECT_EQ(coord[MemDim(2)], 0);
    EXPECT_EQ(coord[MemDim(3)], 2);
}

TEST(MLIR_SubSpacesTests, GetTotalLines) {
    const MemShape dims = {2, 4, 8};

    const auto lines = subspace::getTotalLines(dims, MemDim(1));
    EXPECT_EQ(lines, 2 * 8);
}

TEST(MLIR_SubSpacesTests, GetTotalPlanes) {
    const MemShape dims = {2, 4, 8};

    const auto planes = subspace::getTotalPlanes(dims, MemDim(1), MemDim(2));
    EXPECT_EQ(planes, 2);
}

TEST(MLIR_SubSpacesTests, GetSizes) {
    const MemShape dims = {2, 4, 8};

    const auto sizes = subspace::getSizes(dims);
    ASSERT_EQ(sizes.size(), dims.size());
    EXPECT_EQ(sizes[MemDim(0)], 8 * 4);
    EXPECT_EQ(sizes[MemDim(1)], 8);
    EXPECT_EQ(sizes[MemDim(2)], 1);
}

TEST(MLIR_SubSpacesTests, ArrayElementExclude) {
    MemShape dims = {2, 4, 8};
    dims = subspace::arrayElementExclude(dims, MemDim(1));
    ASSERT_EQ(dims.size(), 2);
    EXPECT_EQ(dims[MemDim(0)], 2);
    EXPECT_EQ(dims[MemDim(1)], 8);
}

TEST(MLIR_SubSpacesTests, ArrayElementInclude) {
    MemShape dims = {2, 8};
    dims = subspace::arrayElementInclude(dims, MemDim(1), int64_t(4));
    ASSERT_EQ(dims.size(), 3);
    EXPECT_EQ(dims[MemDim(0)], 2);
    EXPECT_EQ(dims[MemDim(1)], 4);
    EXPECT_EQ(dims[MemDim(2)], 8);
}
