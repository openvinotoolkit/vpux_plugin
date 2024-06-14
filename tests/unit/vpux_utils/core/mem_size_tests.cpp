//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/mem_size.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(MLIR_MemSize, ValidCases) {
    const auto bits8 = 8_Bit;
    const auto bytes16 = 16_Byte;
    const auto kilobyte10 = 10_KB;
    const auto megabyte2 = 2_MB;
    const auto gigabyte1 = 1_GB;

    EXPECT_EQ(bits8.count(), 8);
    EXPECT_EQ(bytes16.count(), 16);
    EXPECT_EQ(kilobyte10.count(), 10);
    EXPECT_EQ(megabyte2.count(), 2);
    EXPECT_EQ(gigabyte1.count(), 1);

    EXPECT_EQ(bits8.to<Byte>().count(), 1);
    EXPECT_EQ(bytes16.to<Bit>().count(), 16 * 8);
    EXPECT_EQ(kilobyte10.to<Byte>().count(), 10 * 1024);
    EXPECT_EQ(megabyte2.to<KB>().count(), 2 * 1024);
    EXPECT_EQ(megabyte2.to<Byte>().count(), 2 * 1024 * 1024);
    EXPECT_EQ(gigabyte1.to<MB>().count(), 1 * 1024);
    EXPECT_EQ(gigabyte1.to<KB>().count(), 1 * 1024 * 1024);
    EXPECT_EQ(gigabyte1.to<Byte>().count(), 1 * 1024 * 1024 * 1024);
}

TEST(MLIR_MemSize, AlignMemSize) {
    const auto bits109 = 109_Bit;
    const auto bits128 = 128_Bit;
    const auto bytes16 = 16_Byte;
    const auto kilobyte16 = 16_KB;
    const auto megabyte24 = 24_MB;
    const auto gigabyte32 = 32_GB;

    EXPECT_EQ(alignMemSize(bits128, Bit(1)).count(), 128);
    EXPECT_EQ(alignMemSize(bits128, Bit(5)).count(), 130);
    EXPECT_EQ(alignMemSize(bits128, Byte(3)).count(), 144);

    EXPECT_EQ(alignMemSize(bits109, Byte(1)).count(), 112);

    EXPECT_EQ(alignMemSize(bytes16, Byte(1)).count(), 16);
    EXPECT_EQ(alignMemSize(bytes16, Byte(5)).count(), 20);
    EXPECT_EQ(alignMemSize(bytes16, KB(3)).count(), 3072);

    EXPECT_EQ(alignMemSize(kilobyte16, KB(1)).count(), 16);
    EXPECT_EQ(alignMemSize(kilobyte16, KB(5)).count(), 20);
    EXPECT_EQ(alignMemSize(kilobyte16, MB(3)).count(), 3072);

    EXPECT_EQ(alignMemSize(megabyte24, MB(1)).count(), 24);
    EXPECT_EQ(alignMemSize(megabyte24, MB(5)).count(), 25);
    EXPECT_EQ(alignMemSize(megabyte24, GB(3)).count(), 3072);

    EXPECT_EQ(alignMemSize(gigabyte32, GB(1)).count(), 32);
    EXPECT_EQ(alignMemSize(gigabyte32, GB(5)).count(), 35);
}
