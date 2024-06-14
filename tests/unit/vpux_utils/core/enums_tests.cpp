//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <gtest/gtest.h>

enum class TestEnum { A, B };

vpux::StringLiteral stringifyEnum(TestEnum val) {
#define CASE(_val_)       \
    case TestEnum::_val_: \
        return #_val_

    switch (val) {
        CASE(A);
        CASE(B);
    default:
        return "<UNKNOWN>";
    }

#undef CASE
}

TEST(MLIR_PlainEnumTest, Format) {
    EXPECT_EQ(vpux::printToString("{0}", TestEnum::A), "A");
    EXPECT_EQ(vpux::printToString("{0}", TestEnum::B), "B");
}
