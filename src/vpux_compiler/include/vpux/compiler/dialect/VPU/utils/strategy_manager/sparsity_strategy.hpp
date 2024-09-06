//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/Value.h>

namespace vpux {

namespace VPU {

namespace NCESparsity {

// The minimum ratio of weights values that should be sparse
// in order to gain any benefit from compressing the weights
constexpr double WEIGHTS_SPARSITY_FLOAT_RATIO_THRESHOLD = 0.0625;
constexpr double WEIGHTS_SPARSITY_INT_RATIO_THRESHOLD = 0.125;

class BaseWeightsSparsityStrategy {
public:
    virtual ~BaseWeightsSparsityStrategy() = default;

    virtual bool shouldSparsifyWeights(Logger& log, vpux::NDTypeInterface weightsType,
                                       ArrayRef<int64_t> numNonSparseElemsPerOC, bool hasFloatInput);
};

// Made decision on type of weights and ratio of zeros(or zero points)
class RatioBasedWeightsSparsityStrategy : public BaseWeightsSparsityStrategy {
public:
    RatioBasedWeightsSparsityStrategy(double floatRatioThreshold, double intRatioThreshold,
                                      std::optional<double> manualThreshold = std::nullopt);

    bool shouldSparsifyWeights(Logger& log, vpux::NDTypeInterface weightsType, ArrayRef<int64_t> numNonSparseElemsPerOC,
                               bool hasFloatInput) override;

private:
    double _floatRatioThreshold;
    double _intRatioThreshold;
    std::optional<double> _manualThreshold;
};

struct CMXBasedSparsityThreshold {
    static constexpr double DISABLED_SPARSITY_RATIO = std::numeric_limits<double>::max();

    CMXBasedSparsityThreshold(double cmxSizeRatio, double floatRatioThreshold, double intRatioThreshold);

    double _cmxSizeRatio;
    double _floatRatioThreshold;
    double _intRatioThreshold;
};

// TODO: Add NPU37XX specific intervals
const std::initializer_list<CMXBasedSparsityThreshold> CMX_BASED_STRATEGY_DEFAULT_INTERVALS = {
        CMXBasedSparsityThreshold(0., CMXBasedSparsityThreshold::DISABLED_SPARSITY_RATIO,
                                  CMXBasedSparsityThreshold::DISABLED_SPARSITY_RATIO),  // No sparsity for any ratio for
                                                                                        // 0-5% of CMX consumption
        CMXBasedSparsityThreshold(0.05, 0.3, 0.3),  // At least 30% sparse for 5-50% of CMX
        CMXBasedSparsityThreshold(0.5, 0.2, 0.2),   // At least 20% sparse for 50-100%
};

// Made decision on CMX consumption, and zeros/zero-points ratio
// For example :
// [0;5)% of CMX - no sparsity
// [5; 50)% of CMX - at least 30% of weights should be sparse
// [50; 100)% of CMX - at least 20% of weights should be sparse
// Details see in E#45206
class CMXConsumptionBasedWeightsSparsityStrategy : public BaseWeightsSparsityStrategy {
public:
    CMXConsumptionBasedWeightsSparsityStrategy(Byte cmxMemorySize,
                                               const std::initializer_list<CMXBasedSparsityThreshold>& intervals,
                                               std::optional<double> manualThreshold = std::nullopt);

    bool shouldSparsifyWeights(Logger& log, vpux::NDTypeInterface weightsType, ArrayRef<int64_t> numNonSparseElemsPerOC,
                               bool hasFloatInput) override;

private:
    Byte _cmxMemorySize;
    // Map ensures increasing order by key, which is start ratio for interval. Each interval contains int+float ratio
    std::map<double, std::pair<double, double>> _intervals;
    std::optional<double> _manualThreshold;
};

}  // namespace NCESparsity

}  // namespace VPU
}  // namespace vpux
