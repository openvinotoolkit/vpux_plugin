//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace vpux;

template <class Range, typename T>
void cmpArrayRefWithRange(const Range& range, ArrayRef<T> arr) {
    size_t ind = 0;
    for (const auto& v : range) {
        ASSERT_LT(ind, arr.size());
        EXPECT_EQ(&v, &arr[ind]) << ind;
        ++ind;
    }
}

TEST(MLIR_ArrayRefTest, Empty) {
    ArrayRef<int> empty_arr1;
    EXPECT_TRUE(empty_arr1.empty());
}

TEST(MLIR_ArrayRefTest, FromPlainArray) {
    int elems[] = {1, 2, 3, 4, 5};

    ArrayRef<int> arr1(elems);
    ASSERT_EQ(5, arr1.size());
    cmpArrayRefWithRange(make_range(elems), arr1);

    ArrayRef<int> arr2(elems, 5);
    ASSERT_EQ(5, arr2.size());
    cmpArrayRefWithRange(make_range(elems), arr2);

    ArrayRef<int> arr3(elems, elems + 5);
    ASSERT_EQ(5, arr3.size());
    cmpArrayRefWithRange(make_range(elems), arr3);
}

template <class Cont>
class ArrayRefContTest : public testing::Test {
public:
    using cont_type = Cont;
    using value_type = typename Cont::value_type;
    using array_ref_type = ArrayRef<value_type>;

    void testFromContainer() {
        cont_type cont = {1, 2, 3, 4, 5};
        array_ref_type arr = cont;

        ASSERT_EQ(cont.size(), arr.size());
        cmpArrayRefWithRange(cont, arr);
    }
};

using ArrayRefContTestTypes = testing::Types<std::vector<int>, std::array<int, 5>, SmallVector<int, 10>,
                                             SmallVector<int, 2>, std::initializer_list<int>>;
TYPED_TEST_SUITE(ArrayRefContTest, ArrayRefContTestTypes);

TYPED_TEST(ArrayRefContTest, FromContainer) {
    this->testFromContainer();
}
