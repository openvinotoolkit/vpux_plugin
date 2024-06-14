//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include <gtest/gtest.h>
#include <climits>
#include <vector>
#include "vpux/compiler/utils/swizzle_transform.hpp"

using namespace vpux;
using namespace testing;

class SwizzlingTest_VPUX40XX : public TestWithParam<std::tuple<uint32_t, uint32_t>> {};

class SwizzlingTest_VPUX37XX : public TestWithParam<std::tuple<uint32_t, uint32_t>> {};

template <typename T>
bool compareBuffers(const SmallVector<T>& buf1, const MutableArrayRef<T>& buf2) {
    const auto buffer_size{buf1.size()};

    if (buf2.size() != buffer_size) {
        return true;
    }

    bool buffer_diff{false};

    for (size_t i{}; i < buffer_size; ++i) {
        buffer_diff |= buf1[i] != buf2[i];
    }

    return buffer_diff;
}

template <typename T>
bool swizzlingTest(uint32_t elements, uint32_t swizzleKey, VPU::ArchKind archKind) {
    BufferTransform::BufferSwizzleTransform bufferTransform{swizzleKey, archKind};
    const auto swizzlePatternStride{bufferTransform.getSwizzlePatternStride()};

    SmallVector<T> inputVector(elements);
    SmallVector<T> swizzledVector(elements, 0);
    SmallVector<T> deSwizzledVector(elements);
    bool result = true;

    elements = inputVector.size() * sizeof(T);
    elements = (elements + swizzlePatternStride - 1) / swizzlePatternStride;
    elements *= swizzlePatternStride;
    elements /= sizeof(T);

    // Resize the buffers
    inputVector.resize(elements);
    swizzledVector.resize(elements);
    deSwizzledVector.resize(elements);
    MutableArrayRef<T> swizzledArrayRef(swizzledVector);

    for (uint32_t e{}; e < elements; ++e) {
        inputVector[e] = static_cast<T>(rand() % 256);
    }
    auto inputRawData = ArrayRef(reinterpret_cast<const char*>(inputVector.data()), inputVector.size() * sizeof(T));

    const bool noSwizzlingApplied{elements * sizeof(T) <= swizzlePatternStride};
    bufferTransform.swizzle<T>(inputRawData, swizzledArrayRef);
    bufferTransform.deswizzle<T>(swizzledArrayRef, deSwizzledVector);

    auto buffersDifferent{compareBuffers<T>(inputVector, deSwizzledVector)};
    auto buffersShouldBeDifferent{compareBuffers<T>(inputVector, swizzledArrayRef)};

    buffersShouldBeDifferent |= (swizzleKey == 0);  // key=0 disable swizzling, so difference expected.

    // Actual Checks
    if ((!noSwizzlingApplied && (buffersDifferent || !buffersShouldBeDifferent)) ||
        (noSwizzlingApplied && buffersShouldBeDifferent && swizzleKey)) {
        result = false;
    }
    return result;
}

TEST_P(SwizzlingTest_VPUX37XX, swizzlingTest_VPUX37XX) {
    auto params = GetParam();
    auto swizzlingKey = std::get<0>(params);
    const auto elements = std::get<1>(params);

    bool result = false;
    EXPECT_TRUE(result = swizzlingTest<uint32_t>(elements, swizzlingKey, VPU::ArchKind::NPU37XX));
}

TEST_P(SwizzlingTest_VPUX40XX, swizzlingTest_VPUX40XX) {
    auto params = GetParam();
    auto swizzlingKey = std::get<0>(params);
    const auto elements = std::get<1>(params);

    bool result = false;
    EXPECT_TRUE(result = swizzlingTest<uint32_t>(elements, swizzlingKey, VPU::ArchKind::NPU40XX));
}

INSTANTIATE_TEST_SUITE_P(testAligned_VPUX40XX_Key0, SwizzlingTest_VPUX40XX, Combine(Values(0), Values(1024)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX40XX_Key1, SwizzlingTest_VPUX40XX, Combine(Values(1), Values(2048)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX40XX_Key2, SwizzlingTest_VPUX40XX, Combine(Values(2), Values(1024)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX40XX_Key3, SwizzlingTest_VPUX40XX, Combine(Values(3), Values(2048)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX40XX_Key4, SwizzlingTest_VPUX40XX, Combine(Values(4), Values(1024)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX40XX_Key5, SwizzlingTest_VPUX40XX, Combine(Values(5), Values(2048)));

INSTANTIATE_TEST_SUITE_P(testUnaligned_VPUX40XX_Key0, SwizzlingTest_VPUX40XX, Combine(Values(0), Values(1023)));
INSTANTIATE_TEST_SUITE_P(testUnaligned_VPUX40XX_Key1, SwizzlingTest_VPUX40XX, Combine(Values(1), Values(2049)));
INSTANTIATE_TEST_SUITE_P(testUnaligned_VPUX40XX_Key2, SwizzlingTest_VPUX40XX, Combine(Values(2), Values(1023)));
INSTANTIATE_TEST_SUITE_P(testUnaligned_VPUX40XX_Key3, SwizzlingTest_VPUX40XX, Combine(Values(3), Values(2049)));
INSTANTIATE_TEST_SUITE_P(testUnaligned_VPUX40XX_Key4, SwizzlingTest_VPUX40XX, Combine(Values(4), Values(1023)));
INSTANTIATE_TEST_SUITE_P(testUnaligned_VPUX40XX_Key5, SwizzlingTest_VPUX40XX, Combine(Values(5), Values(2049)));

INSTANTIATE_TEST_SUITE_P(testAligned_VPUX37XX_Key0, SwizzlingTest_VPUX37XX, Combine(Values(0), Values(1024)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX37XX_Key1, SwizzlingTest_VPUX37XX, Combine(Values(1), Values(2048)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX37XX_Key2, SwizzlingTest_VPUX37XX, Combine(Values(2), Values(1024)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX37XX_Key3, SwizzlingTest_VPUX37XX, Combine(Values(3), Values(2048)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX37XX_Key4, SwizzlingTest_VPUX37XX, Combine(Values(4), Values(1024)));
INSTANTIATE_TEST_SUITE_P(testAligned_VPUX37XX_Key5, SwizzlingTest_VPUX37XX, Combine(Values(5), Values(2048)));

INSTANTIATE_TEST_SUITE_P(testUnligned_VPUX37XX_Key0, SwizzlingTest_VPUX37XX, Combine(Values(0), Values(1023)));
INSTANTIATE_TEST_SUITE_P(testUnligned_VPUX37XX_Key1, SwizzlingTest_VPUX37XX, Combine(Values(1), Values(2049)));
INSTANTIATE_TEST_SUITE_P(testUnligned_VPUX37XX_Key2, SwizzlingTest_VPUX37XX, Combine(Values(2), Values(1023)));
INSTANTIATE_TEST_SUITE_P(testUnligned_VPUX37XX_Key3, SwizzlingTest_VPUX37XX, Combine(Values(3), Values(2049)));
INSTANTIATE_TEST_SUITE_P(testUnligned_VPUX37XX_Key4, SwizzlingTest_VPUX37XX, Combine(Values(4), Values(1022)));
INSTANTIATE_TEST_SUITE_P(testUnligned_VPUX37XX_Key5, SwizzlingTest_VPUX37XX, Combine(Values(5), Values(2049)));
