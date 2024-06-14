//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;
using namespace vpux::VPURegMapped;

template <uint8_t BIT_WIDTH, VPURegMapped::RegFieldDataType DATA_TYPE>
struct RegFieldPretender {
    static constexpr uint8_t getRegFieldWidth() {
        return BIT_WIDTH;
    }
    static constexpr VPURegMapped::RegFieldDataType getRegFieldDataType() {
        return DATA_TYPE;
    }
};

TEST(VPURegMapped_CheckedCastReg, ValidCasesUintReg) {
    // uint64_t
    EXPECT_EQ(uint64_t(0), (checked_cast_reg<RegFieldPretender<1, VPURegMapped::RegFieldDataType::UINT>>(uint64_t(0))));
    EXPECT_EQ(uint64_t(255),
              (checked_cast_reg<RegFieldPretender<8, VPURegMapped::RegFieldDataType::UINT>>(uint64_t(255))));
    EXPECT_EQ(uint64_t(0xFFFFFFFFFFFFFFFF),
              (checked_cast_reg<RegFieldPretender<64, VPURegMapped::RegFieldDataType::UINT>>(
                      uint64_t(0xFFFFFFFFFFFFFFFF))));

    // int64_t
    EXPECT_EQ(uint64_t(0), (checked_cast_reg<RegFieldPretender<1, VPURegMapped::RegFieldDataType::UINT>>(int64_t(0))));
    EXPECT_EQ(uint64_t(255),
              (checked_cast_reg<RegFieldPretender<8, VPURegMapped::RegFieldDataType::UINT>>(int64_t(255))));
    EXPECT_EQ(uint64_t(0x7FFFFFFFFFFFFFFF),
              (checked_cast_reg<RegFieldPretender<64, VPURegMapped::RegFieldDataType::UINT>>(
                      int64_t(0x7FFFFFFFFFFFFFFFll))));

    // enum
    enum class TestEnum { value_0, value_1, value_2 };
    EXPECT_EQ(uint64_t(0),
              (checked_cast_reg<RegFieldPretender<3, VPURegMapped::RegFieldDataType::UINT>>(TestEnum::value_0)));
    EXPECT_EQ(uint64_t(1),
              (checked_cast_reg<RegFieldPretender<3, VPURegMapped::RegFieldDataType::UINT>>(TestEnum::value_1)));
    EXPECT_EQ(uint64_t(2),
              (checked_cast_reg<RegFieldPretender<3, VPURegMapped::RegFieldDataType::UINT>>(TestEnum::value_2)));

    // bool
    EXPECT_EQ(uint64_t(0), (checked_cast_reg<RegFieldPretender<1, VPURegMapped::RegFieldDataType::UINT>>(false)));
    EXPECT_EQ(uint64_t(1), (checked_cast_reg<RegFieldPretender<1, VPURegMapped::RegFieldDataType::UINT>>(true)));
}

TEST(VPURegMapped_CheckedCastReg, InvalidCasesUintReg) {
    using Exception = vpux::Exception;
    // uint64_t
    EXPECT_THROW((checked_cast_reg<RegFieldPretender<8, VPURegMapped::RegFieldDataType::UINT>>(uint64_t(256))),
                 Exception);
    // int64_t
    EXPECT_THROW((checked_cast_reg<RegFieldPretender<8, VPURegMapped::RegFieldDataType::UINT>>(int64_t(256))),
                 Exception);
    EXPECT_THROW((checked_cast_reg<RegFieldPretender<2, VPURegMapped::RegFieldDataType::UINT>>(int64_t(-1))),
                 Exception);
    // enum
    enum class TestEnum { value_0, value_1, value_2 };

    EXPECT_THROW((checked_cast_reg<RegFieldPretender<1, VPURegMapped::RegFieldDataType::UINT>>(TestEnum::value_2)),
                 Exception);
}

TEST(VPURegMapped_CheckedCastReg, ValidCasesSintReg) {
    // uint64_t
    EXPECT_EQ(uint64_t(0), (checked_cast_reg<RegFieldPretender<2, VPURegMapped::RegFieldDataType::SINT>>(uint64_t(0))));
    EXPECT_EQ(uint64_t(127),
              (checked_cast_reg<RegFieldPretender<8, VPURegMapped::RegFieldDataType::SINT>>(uint64_t(127))));
    EXPECT_EQ(uint64_t(0x7FFFFFFFFFFFFFFF),
              (checked_cast_reg<RegFieldPretender<64, VPURegMapped::RegFieldDataType::SINT>>(
                      uint64_t(0x7FFFFFFFFFFFFFFF))));

    // int64_t
    EXPECT_EQ(uint64_t(3), (checked_cast_reg<RegFieldPretender<2, VPURegMapped::RegFieldDataType::SINT>>(int64_t(-1))));
    EXPECT_EQ(uint64_t(128),
              (checked_cast_reg<RegFieldPretender<8, VPURegMapped::RegFieldDataType::SINT>>(int64_t(-128))));
    EXPECT_EQ(uint64_t(384),
              (checked_cast_reg<RegFieldPretender<9, VPURegMapped::RegFieldDataType::SINT>>(int64_t(-128))));
    EXPECT_EQ(uint64_t(0x8000000000000000),
              (checked_cast_reg<RegFieldPretender<64, VPURegMapped::RegFieldDataType::SINT>>(
                      std::numeric_limits<int64_t>::min())));
}

TEST(VPURegMapped_CheckedCastReg, InvalidCasesSintReg) {
    using Exception = vpux::Exception;
    // uint64_t
    EXPECT_THROW((checked_cast_reg<RegFieldPretender<2, VPURegMapped::RegFieldDataType::SINT>>(uint64_t(2))),
                 Exception);
    // int64_t
    EXPECT_THROW((checked_cast_reg<RegFieldPretender<8, VPURegMapped::RegFieldDataType::SINT>>(int64_t(128))),
                 Exception);
    EXPECT_THROW((checked_cast_reg<RegFieldPretender<8, VPURegMapped::RegFieldDataType::SINT>>(int64_t(-129))),
                 Exception);
}

TEST(VPURegMapped_CheckedCastReg, ValidCasesFPReg) {
    // double
    EXPECT_EQ(uint64_t(0x3fb999999999999a),
              (checked_cast_reg<RegFieldPretender<64, VPURegMapped::RegFieldDataType::FP>>(double(0.1))));

    EXPECT_EQ(uint32_t(0xc142147b),
              (checked_cast_reg<RegFieldPretender<32, VPURegMapped::RegFieldDataType::FP>>(float(-12.13))));
}
