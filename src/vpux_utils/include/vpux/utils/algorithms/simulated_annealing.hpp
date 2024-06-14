//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <random>

#include <vpux/utils/core/error.hpp>

namespace vpux {
namespace algorithm {

extern bool defaultStopCondition(size_t temperature);
extern void defaultTemperatureCallback(size_t& temperature);
template <class T>
using StateGetter = std::function<const T(int, double&, const T* const)>;
template <class T>
using CostGetter = std::function<double(const T&)>;
template <class T>
using FullCostGetter = std::function<double()>;

template <class T>
using Successor = std::function<void(const T&)>;

template <class T>
using UpdateBestSolution = std::function<void(const T&)>;

using StopCondition = std::function<bool(size_t)>;
using TemperatureCallback = std::function<void(size_t&)>;

/*
 * Simulated Annealing Algorithm
 * Optimization algorithm for global minimum search
 * On each step it gets "neighbour" of current state and
 * tries to compare the cost of new state and current one.
 * If new solution is "better" it's accepted, if not,
 * it might also be accepted with a probability of e^(-delta/temperature).
 * Temperature decreases on each step, algorithm stops if it's zero.
 */
template <class State>
State simulatedAnnealing(size_t temperature, size_t iterations, StateGetter<State> getState, CostGetter<State> getCost,
                         FullCostGetter<State> getFullCost = nullptr, Successor<State> successCallback = nullptr,
                         UpdateBestSolution<State> solutionCallBack = nullptr,
                         StopCondition stopCondition = defaultStopCondition,
                         TemperatureCallback changeTemperature = defaultTemperatureCallback) {
    VPUX_THROW_WHEN(getState == nullptr || getCost == nullptr, "Functions for getting state and cost are not set");

    const double lowerBound = 0;
    const double upperBound = 1;
    std::uniform_real_distribution<double> dist(lowerBound, upperBound);

    std::mt19937 randomEngine(0);

    double currentFullCost = 0;
    auto currentState = getState(static_cast<int>(temperature), currentFullCost, nullptr);
    auto currentCost = getCost(currentState);

    // set default best state and cost and cost of full function.
    currentFullCost = currentCost;

    std::function<void(double)> getUpdatedFullCost = [&currentFullCost](double delta) {
        currentFullCost += delta;
    };

    if (getFullCost == nullptr) {
        getFullCost = [&]() {
            return currentCost;
        };
        getUpdatedFullCost = [&currentFullCost, &currentCost](double) {
            currentFullCost = currentCost;
        };
    }
    // get Full cost
    currentFullCost = getFullCost();
    auto bestState = currentState;
    double bestCost = currentFullCost;

    VPUX_THROW_WHEN(stopCondition == nullptr || changeTemperature == nullptr,
                    "Functions for checking stop condition and changing temperature are not set");

    while (!stopCondition(temperature)) {
        size_t index = iterations;
        while (index > 0) {
            auto neighbour = getState(static_cast<int>(temperature), currentFullCost, &currentState);
            const auto newCost = getCost(neighbour);
            const auto delta = newCost - currentCost;
            const double acceptance = dist(randomEngine);
            // use here Metropolis Criterion to decide acceptance of worse states
            if (delta <= 0 || exp(-(delta / temperature)) > acceptance) {
                currentState = neighbour;
                currentCost = newCost;
                // update full cost of solution
                if (successCallback != nullptr) {
                    successCallback(currentState);
                }

                getUpdatedFullCost(delta);

                if (currentFullCost <= bestCost) {
                    bestState = currentState;
                    bestCost = currentFullCost;

                    if (solutionCallBack != nullptr) {
                        // Callback with the updated best solution.
                        solutionCallBack(bestState);
                    }
                }
            }

            // get new state
            currentState = getState(static_cast<int>(temperature), currentFullCost, nullptr);
            currentCost = getCost(currentState);
            --index;
        }

        changeTemperature(temperature);
    }

    return bestState;
}

}  // namespace algorithm
}  // namespace vpux
