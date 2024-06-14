//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include "vpux/utils/algorithms/simulated_annealing.hpp"

using AlgorithmLibraryUnitTests = ::testing::Test;

/*
 * Test finds the minimum of function
 * f(x) = 2*x^3 - 6*x^2 + 10 on the range [0, 4)
 */
TEST_F(AlgorithmLibraryUnitTests, getFunctionMinimum) {
    int lowerBound = 0;
    int upperBound = 4;
    std::mt19937 gen(1);
    std::uniform_int_distribution<int> rng(lowerBound, upperBound);

    auto getState = [&](int /* temp */, double& /*cost*/, const int* const /*state*/) -> int {
        return rng(gen);
    };

    auto getCost = [](const int& state) -> double {
        return 2 * std::pow(state, 3) - 6 * std::pow(state, 2) + 10;
    };

    size_t counter = 0;
    auto successor = [&](const int& /*state*/) {
        ++counter;
    };

    auto temp = getCost(upperBound);

    int bestState = std::numeric_limits<int>::max();

    auto updateSolution = [&](const int& state) {
        bestState = state;
    };

    EXPECT_EQ(vpux::algorithm::simulatedAnnealing<int>(static_cast<size_t>(temp), 1, getState, getCost, nullptr,
                                                       successor, updateSolution),
              2);
    EXPECT_EQ(bestState, 2);

    EXPECT_LE(counter, temp);
}

/*
 * Test finds global minimum of function
 * f(x) = (x-5)^4 + 4*(x-5)^3 - 8*(x-5)^2 + 130 on the range [0, 8)
 * Function has 2 minima this range - 1 and 6, test checks that SA finds the global one
 */
TEST_F(AlgorithmLibraryUnitTests, getFunctionGlobalMinimum) {
    int lowerBound = 0;
    int upperBound = 8;

    auto getCost = [](const int& state) -> double {
        int shift = state - 5;
        return std::pow(shift, 4) + 4 * std::pow(shift, 3) - 8 * std::pow(shift, 2) + 130;
    };

    auto temperature = getCost(upperBound);

    auto getState = [&](int /* temp */, double& /*cost*/, const int* const state) -> int {
        if (state == nullptr) {
            return lowerBound;
        }

        auto newState = (*state) + 1;
        int i = newState % (upperBound - lowerBound);
        return i;
    };

    auto stopCondition = [&](size_t temperature) -> bool {
        return temperature < static_cast<size_t>(upperBound);
    };

    auto changeTemp = [&](size_t& temperature) {
        temperature -= (upperBound - lowerBound);
    };

    EXPECT_EQ(vpux::algorithm::simulatedAnnealing<int>(static_cast<size_t>(temperature), 1, getState, getCost, nullptr,
                                                       nullptr, nullptr, stopCondition, changeTemp),
              1);
}

/*
 * Test finds global minimum of function
 * f(x) = 3*sin(3x) + 3* cos(5x) + 10 on the range [0, 13)
 * Function has several minimas this range, test checks that SA finds the global one
 */
TEST_F(AlgorithmLibraryUnitTests, getFunctionMultipleGlobalMinimum) {
    int lowerBound = 0;
    int upperBound = 13;

    std::mt19937 gen(1);
    std::uniform_int_distribution<int> rng(lowerBound, upperBound);

    auto getCost = [](const int& state) -> double {
        return 3 * std::sin(3 * state) + 3 * std::cos(5 * state) + 10;
    };

    int temperature = upperBound * 10;

    auto getState = [&](int /* temp */, double& /*cost*/, const int* const /*state*/) -> int {
        return rng(gen);
    };

    EXPECT_EQ(vpux::algorithm::simulatedAnnealing<int>(static_cast<size_t>(temperature), 1, getState, getCost), 12);
}
