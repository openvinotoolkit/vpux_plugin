//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_axis_increment.hpp"
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

int64_t DefaultVFAxisIncrement::getLimitValue(mlir::ArrayRef<int64_t> values) const {
    auto minAxisLength = std::min_element(values.begin(), values.end());

    VPUX_THROW_WHEN(minAxisLength == values.end(), "Unable to get minimum of axis length");

    return *minAxisLength;
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
    for (auto factor : reverse(factors)) {
        if (factor.first > value && factor.first < limit) {
            value = factor.first;
            break;
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

int64_t ChannelsVFAxisIncrement::getLimitValue(mlir::ArrayRef<int64_t> values) const {
    std::function<int64_t(int64_t, int64_t)> gcdNum = [&](int64_t one, int64_t two) {
        if (one == 0)
            return two;
        return gcdNum(two % one, one);
    };

    const auto gcdArray = [&]() {
        int64_t result = 1;
        if (values.empty()) {
            return result;
        }
        auto size = values.size();
        result = values.front();
        if (size == 1) {
            return result;
        }

        for (int64_t i : irange(size)) {
            result = gcdNum(result, values[i]);
        }
        return result;
    };
    return gcdArray();
}
