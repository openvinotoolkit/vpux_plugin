//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/checked_cast.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(MLIR_CheckedCast, ValidCases) {
    const int8_t valI8 = 10;
    const int16_t valI16 = 10;
    const int32_t valI32 = 10;
    const int64_t valI64 = 10;

    const uint8_t valU8 = 10;
    const uint16_t valU16 = 10;
    const uint32_t valU32 = 10;
    const uint64_t valU64 = 10;

    const float valF32 = 10.0f;
    const double valF64 = 10.0;

    const auto test = [](auto val) -> void {
        EXPECT_EQ(checked_cast<int8_t>(val), static_cast<int8_t>(val));
        EXPECT_EQ(checked_cast<int16_t>(val), static_cast<int16_t>(val));
        EXPECT_EQ(checked_cast<int32_t>(val), static_cast<int32_t>(val));
        EXPECT_EQ(checked_cast<int64_t>(val), static_cast<int64_t>(val));

        EXPECT_EQ(checked_cast<uint8_t>(val), static_cast<uint8_t>(val));
        EXPECT_EQ(checked_cast<uint16_t>(val), static_cast<uint16_t>(val));
        EXPECT_EQ(checked_cast<uint32_t>(val), static_cast<uint32_t>(val));
        EXPECT_EQ(checked_cast<uint64_t>(val), static_cast<uint64_t>(val));
    };

    test(valI8);
    test(valI16);
    test(valI32);
    test(valI64);

    test(valU8);
    test(valU16);
    test(valU32);
    test(valU64);

    test(valF32);
    test(valF64);
}

TEST(MLIR_CheckedCast, InvalidThrowsException) {
    using Exception = vpux::Exception;

    const int8_t valI8 = -10;
    const int16_t valI16 = -10;
    const int32_t valI32 = -10;
    const int64_t valI64 = -10;

    const float valF32 = -10.0f;
    const double valF64 = -10.0;

    const auto test = [](auto val) -> void {
        EXPECT_THROW(checked_cast<uint8_t>(val), Exception);
        EXPECT_THROW(checked_cast<uint16_t>(val), Exception);
        EXPECT_THROW(checked_cast<uint32_t>(val), Exception);
        EXPECT_THROW(checked_cast<uint64_t>(val), Exception);
    };

    test(valI8);
    test(valI16);
    test(valI32);
    test(valI64);

    test(valF32);
    test(valF64);
}
