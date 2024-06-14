//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/factors.hpp"

#include <cmath>

using namespace vpux;

SmallVector<Factors> vpux::getFactorsList(int64_t n) {
    SmallVector<Factors> factors;
    const int64_t maxPossibleFactor = static_cast<int64_t>(sqrt(n));
    for (int64_t i = 1; i <= maxPossibleFactor; ++i) {
        if (n % i == 0) {
            factors.emplace_back(n / i, i);  // larger, smaller
        }
    }
    return factors;
}

SmallVector<Factors> vpux::getFactorsListWithLimitation(int64_t n, int64_t limit) {
    SmallVector<Factors> factors;
    const int64_t maxPossibleFactor = static_cast<int64_t>(sqrt(n));
    for (int64_t i = 1; i <= maxPossibleFactor; ++i) {
        if ((n % i == 0) && (i <= limit)) {
            factors.emplace_back(i, n / i);  // larger, smaller
        }
    }
    return factors;
}

SmallVector<int64_t> vpux::getPrimeFactors(int64_t n) {
    SmallVector<int64_t> factors;
    for (int64_t i = 2; i < n;) {
        if (n % i == 0) {
            factors.push_back(i);
            n = n / i;
        } else {
            i++;
        }
    }
    factors.push_back(n);
    return factors;
}

int64_t vpux::smallestDivisor(int64_t n) {
    int64_t i = 2;
    int64_t limit = sqrt(n);
    while (i <= limit) {
        if (n % i == 0) {
            return i;
        }
        i++;
    }
    return n;
}
