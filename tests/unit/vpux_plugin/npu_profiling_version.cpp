//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include "vpux/utils/profiling/metadata.hpp"

using ProfVersionUnitTests = ::testing::Test;
using namespace vpux::profiling;

namespace {
uint32_t getSectionEncodingWrapper(const std::vector<uint8_t> values) {
    return getProfilingSectionEncoding(values.data(), values.size());
}
}  // namespace

TEST_F(ProfVersionUnitTests, getSectionEncoding) {
    using namespace vpux::profiling;

    EXPECT_EQ(getSectionEncodingWrapper({0x1, 0x0, 0x0, 0x0}), 1);
    EXPECT_EQ(getSectionEncodingWrapper({0x10, 0x0, 0x0, 0x0, 0x1, 0x2, 0x3}), 16);
    EXPECT_NE(getSectionEncodingWrapper({0x0, 0x0, 0x0, 0x09, 0x1, 0x2}), 16);

    EXPECT_ANY_THROW(getProfilingSectionEncoding(nullptr, 0));
    EXPECT_ANY_THROW(getSectionEncodingWrapper({0x1}));
}
