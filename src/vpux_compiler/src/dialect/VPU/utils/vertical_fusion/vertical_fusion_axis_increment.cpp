//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_axis_increment.hpp"
#include "vpux/compiler/utils/factors.hpp"

using namespace vpux::VPU;

int64_t DefaultVFAxisIncrement::getMiddleValue(int64_t min, int64_t max) const {
    return (max + min) / 2;
}

void DefaultVFAxisIncrement::increasedValue(int64_t& value, int64_t /*limit*/) const {
    ++value;
}

void DefaultVFAxisIncrement::decreasedValue(int64_t& value, int64_t /*limit*/) const {
    --value;
}

int64_t getMinValue(mlir::ArrayRef<int64_t> values) {
    auto minValueIt = std::min_element(values.begin(), values.end());
    VPUX_THROW_WHEN(minValueIt == values.end(), "Unable to get minimum of axis length");
    return *minValueIt;
}

int64_t DefaultVFAxisIncrement::getLimitValue(mlir::ArrayRef<int64_t> alignedValues,
                                              mlir::ArrayRef<int64_t> unalignedValues) const {
    VPUX_THROW_WHEN(alignedValues.empty() && unalignedValues.empty(),
                    "alignedValues and unalignedValues are both empty vectors");

    SmallVector<int64_t> unionVec;
    unionVec.reserve(alignedValues.size() + unalignedValues.size());
    unionVec.insert(unionVec.end(), alignedValues.begin(), alignedValues.end());
    unionVec.insert(unionVec.end(), unalignedValues.begin(), unalignedValues.end());

    return getMinValue(unionVec);
}

int64_t ChannelsVFAxisIncrement::getMiddleValue(int64_t min, int64_t max) const {
    auto factors = getFactorsList(max);
    if (factors.empty()) {
        return min;
    }

    for (int64_t i = factors.size() / 2; i >= 0; --i) {
        auto value = factors[i];

        if (value.first > min && value.first < max) {
            return value.first;
        }
    }

    return min;
}

void ChannelsVFAxisIncrement::increasedValue(int64_t& value, int64_t limit) const {
    auto factors = getFactorsList(limit);
    if (factors.empty()) {
        return;
    }
    // Search the tiling numbers ascending
    // In factor list, seconds are smaller than firsts
    for (auto factor : factors) {
        if (factor.second > value && factor.second < limit) {
            value = factor.second;
            return;
        }
    }
    for (auto factor : reverse(factors)) {
        if (factor.first > value && factor.first < limit) {
            value = factor.first;
            return;
        }
    }
}

void ChannelsVFAxisIncrement::decreasedValue(int64_t& value, int64_t limit) const {
    auto factors = getFactorsList(value);
    if (factors.empty()) {
        return;
    }
    for (auto factor : factors) {
        if (factor.first < value && factor.first > limit) {
            value = factor.first;
            break;
        }
    }
}

int64_t ChannelsVFAxisIncrement::getLimitValue(mlir::ArrayRef<int64_t> alignedValues,
                                               mlir::ArrayRef<int64_t> unalignedValues) const {
    VPUX_THROW_WHEN(alignedValues.empty() && unalignedValues.empty(),
                    "alignedValues and unalignedValues are both empty vectors");

    std::function<int64_t(int64_t, int64_t)> gcdNum = [&](int64_t one, int64_t two) {
        if (one == 0)
            return two;
        return gcdNum(two % one, one);
    };

    const auto gcdArray = [&]() {
        int64_t result = 1;
        if (alignedValues.empty()) {
            return result;
        }
        auto size = alignedValues.size();
        result = alignedValues.front();
        if (size == 1) {
            return result;
        }

        for (int64_t i : irange(size)) {
            result = gcdNum(result, alignedValues[i]);
        }
        return result;
    };

    // 1. if alignedValues array is empty, return minimal value in unalignedValues directly
    if (alignedValues.empty() && !unalignedValues.empty()) {
        return getMinValue(unalignedValues);
    }

    // 2. if alignedValues array is not empty and unalignedValues array is empty, return gcd of alignedValues array
    auto gcdValue = gcdArray();
    if (unalignedValues.empty()) {
        return gcdValue;
    }

    // 3. if both alignedValues and unalignedValues arrays are not empty, in order to ensure the final result is the
    // minimal element and can be divided by all elements in alignedValues array, we calculate gcdValue of the elements
    // in alignedValues array and compare it with the minimal element in unalignedValues array:
    //  a. if gcdValue <= minimal element in unalignedValues array, return gcdValue as the final result
    //  b. if gcdValue > minimal element in unalignedValues array, calculate the gcd of gcdValue and minimal element in
    //  unalignedValues array
    auto minUnalignedValue = getMinValue(unalignedValues);
    if (gcdValue <= minUnalignedValue) {
        return gcdValue;
    }

    return gcdNum(gcdValue, minUnalignedValue);
}
