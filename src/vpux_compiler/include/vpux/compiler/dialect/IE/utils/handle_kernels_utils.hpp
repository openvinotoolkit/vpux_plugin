//
// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/factors.hpp"

namespace vpux {
namespace IE {

struct FactorsInfo final {
    Factors factors;
    int64_t padValue = 0;

    explicit FactorsInfo(Factors factors, int64_t padValue): factors(factors), padValue(padValue) {
    }
    explicit FactorsInfo(int64_t larger, int64_t smaller, int64_t padValue)
            : factors(larger, smaller), padValue(padValue) {
    }
};

struct KernelsInfo final {
    Shape firstKernel;
    Shape secondKernel;
    Shape padBegin;
    Shape padEnd;

    explicit KernelsInfo(ShapeRef firstKernel, ShapeRef secondKernel, ShapeRef padBegin, ShapeRef padEnd)
            : firstKernel(firstKernel.raw()),
              secondKernel(secondKernel.raw()),
              padBegin(padBegin.raw()),
              padEnd(padEnd.raw()) {
    }
};

bool hasSupportedKernels(ShapeRef kernelSize, int64_t maxKernelSize);
bool isPoolingKernelSizeValid(int64_t kernelSize, int64_t maxKernelSize);

std::optional<IE::KernelsInfo> calculateKernelsInfo(ShapeRef origKernel, int64_t maxKernelSize, Logger log);
std::optional<IE::FactorsInfo> getFactors(const int64_t kernelSize, const int64_t maxKernelSize);
bool checkFactors(const Factors& factors, int64_t kernelSize, int64_t maxKernelSize);

std::optional<Factors> getFactorsWithLimitation(int64_t val, int64_t limit);
std::optional<Factors> getFactorsAroundWithLimitation(int64_t val, int64_t aroundVal, int64_t limit);

}  // namespace IE
}  // namespace vpux
