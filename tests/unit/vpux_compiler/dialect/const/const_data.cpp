//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/const_data.hpp"
#include "common/utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(ConstDataTests, Allocate) {
    auto container = Const::ConstData::allocate<double>(10);
    ASSERT_TRUE(container.isMutable());
    ASSERT_EQ(container.size(), 10 * sizeof(double));

    auto rawData = container.data();
    ASSERT_EQ(rawData.size(), 10 * sizeof(double));

    auto mutableData = container.mutableData<double>();
    ASSERT_EQ(mutableData.size(), 10);

    auto typedPtr = reinterpret_cast<const double*>(rawData.data());
    ASSERT_EQ(typedPtr, mutableData.data()) << "these two should really point to the same place";

    // ensure it's writable/readable
    for (size_t i = 0; i < mutableData.size(); ++i) {
        mutableData[i] = static_cast<double>(i) + 42.;
        ASSERT_EQ(mutableData[i], typedPtr[i]);
    }
}

TEST(ConstDataTests, FromRawBuffer) {
    std::vector<double> realData = {0.0, 1.0, 2.0, 3.0, 4.0};
    auto container = Const::ConstData::fromRawBuffer(realData.data(), realData.size() * sizeof(double));
    ASSERT_FALSE(container.isMutable());
    ASSERT_EQ(container.data<double>(), ArrayRef<double>(realData));

    auto rawData = container.data();
    ASSERT_EQ(rawData.size() / sizeof(double), realData.size());

    auto doubleData = container.data<double>();
    ASSERT_EQ(doubleData.size(), realData.size());

    realData[2] = 42.;
    ASSERT_EQ(doubleData[2], realData[2]);
}

TEST(ConstDataTests, AllocateAndMove) {
    auto container = Const::ConstData::allocate<int>(5);
    container.mutableData<int>()[1] = 42;

    auto newContainer = std::move(container);
    ASSERT_EQ(newContainer.data<int>()[1], 42);
    ASSERT_EQ(newContainer.mutableData<int>()[1], 42);
}

TEST(ConstDataTests, FromRawBufferAndMove) {
    std::vector<int> realData = {0, 1, 2, 3, 4};
    auto container = Const::ConstData::fromRawBuffer(realData.data(), realData.size() * sizeof(int));
    realData[1] = 42;

    auto newContainer = std::move(container);
    ASSERT_EQ(newContainer.data<int>()[1], 42);
}

TEST(ConstDataTests, MoveAssign) {
    Const::ConstData container{};
    ASSERT_EQ(container.data(), ArrayRef<char>());
    ASSERT_EQ(container.size(), 0);

    container = Const::ConstData::allocate<int>(90);
    ASSERT_TRUE(container.isMutable());
    ASSERT_EQ(container.size(), 90 * sizeof(int));
    ASSERT_NE(container.data<int>().data(), nullptr);

    std::vector<int> realData = {0, 1, 2, 3, 4};
    container = Const::ConstData::fromRawBuffer(realData.data(), realData.size() * sizeof(int));
    ASSERT_FALSE(container.isMutable());
    ASSERT_EQ(container.size() / sizeof(int), realData.size());
    ASSERT_EQ(container.data<int>(), ArrayRef<int>(realData));
}
