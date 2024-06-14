//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/strategy_manager/sparsity_strategy.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <limits>

using namespace vpux;
namespace vpux {
namespace VPU {

namespace NCESparsity {

using namespace VPU::NCESparsity;

bool BaseWeightsSparsityStrategy::shouldSparsifyWeights(Logger&, vpux::NDTypeInterface, ArrayRef<int64_t>, bool) {
    return false;
}

RatioBasedWeightsSparsityStrategy::RatioBasedWeightsSparsityStrategy(double floatRatioThreshold,
                                                                     double intRatioThreshold,
                                                                     std::optional<double> manualThreshold)
        : _floatRatioThreshold(floatRatioThreshold),
          _intRatioThreshold(intRatioThreshold),
          _manualThreshold(manualThreshold) {
}

bool RatioBasedWeightsSparsityStrategy::shouldSparsifyWeights(Logger& log, vpux::NDTypeInterface weightsType,
                                                              ArrayRef<int64_t> numNonSparseElemsPerOC,
                                                              bool hasFloatInput) {
    // This case requires tiling over input channels, which is not supported with the current representation of the
    // compression scheme
    if (weightsType.getShape()[Dims4D::Filter::IC] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        log.trace("Input channels are larger than 8K. Skipping");
        return false;
    }
    const auto actualSparsityRatio = getSparsityRatio(weightsType, numNonSparseElemsPerOC);
    if (isDoubleEqual(actualSparsityRatio, 1.0)) {
        log.trace("All weights are zero, so sparsity ratio is 1. Skipping");
        return false;
    }

    auto sparsityRatioThreshold = hasFloatInput ? _floatRatioThreshold : _intRatioThreshold;
    if (_manualThreshold.has_value()) {
        sparsityRatioThreshold = _manualThreshold.value();
    }

    log.trace("Sparsity ratio {0}, threshold {1}", actualSparsityRatio, sparsityRatioThreshold);
    return std::isgreaterequal(actualSparsityRatio, sparsityRatioThreshold);
}

constexpr double CMXBasedSparsityThreshold::DISABLED_SPARSITY_RATIO;

CMXBasedSparsityThreshold::CMXBasedSparsityThreshold(double cmxSizeRatio, double floatRatioThreshold,
                                                     double intRatioThreshold)
        : _cmxSizeRatio(cmxSizeRatio),
          _floatRatioThreshold(floatRatioThreshold),
          _intRatioThreshold(intRatioThreshold) {
}

CMXConsumptionBasedWeightsSparsityStrategy::CMXConsumptionBasedWeightsSparsityStrategy(
        Byte cmxMemorySize, const std::initializer_list<CMXBasedSparsityThreshold>& intervals,
        std::optional<double> manualThreshold)
        : _cmxMemorySize(cmxMemorySize), _manualThreshold(manualThreshold) {
    for (const auto& interval : intervals) {
        const auto cmxRatio = interval._cmxSizeRatio;
        VPUX_THROW_WHEN(cmxRatio < 0. || cmxRatio >= 1.,
                        "CMX consumption ratio should be in (0;1) interval, but got {0}", cmxRatio);
        _intervals.emplace(cmxRatio, std::make_pair(interval._intRatioThreshold, interval._floatRatioThreshold));
    }
    const auto disabledSparsityRatioInterval = std::make_pair(CMXBasedSparsityThreshold::DISABLED_SPARSITY_RATIO,
                                                              CMXBasedSparsityThreshold::DISABLED_SPARSITY_RATIO);
    // Disable sparsity for 0.-first interval
    _intervals.emplace(0., disabledSparsityRatioInterval);
    // Disable sparsity for >1. interval
    _intervals.emplace(1., disabledSparsityRatioInterval);
}

bool CMXConsumptionBasedWeightsSparsityStrategy::shouldSparsifyWeights(Logger& log, vpux::NDTypeInterface weightsType,
                                                                       ArrayRef<int64_t> numNonSparseElemsPerOC,
                                                                       bool hasFloatInput) {
    // This case requires tiling over input channels, which is not supported with the current representation of the
    // compression scheme
    if (weightsType.getShape()[Dims4D::Filter::IC] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        log.trace("Input channels are larger than 8K. Skipping");
        return false;
    }

    double sparsityRatioThreshold = 0.0;
    if (_manualThreshold.has_value()) {
        sparsityRatioThreshold = _manualThreshold.value();
    } else {
        const auto weightsSize = weightsType.getTotalAllocSize();
        const double cmxConsumptionRatio =
                checked_cast<double>(weightsSize.count()) / checked_cast<double>(_cmxMemorySize.count());
        log.trace("CMX consumption ratio {0}", cmxConsumptionRatio);
        auto intervalIt = _intervals.lower_bound(cmxConsumptionRatio);
        intervalIt--;
        const auto interval = intervalIt->second;
        sparsityRatioThreshold = hasFloatInput ? interval.second : interval.first;
    }

    const auto actualSparsityRatio = getSparsityRatio(weightsType, numNonSparseElemsPerOC);
    if (isDoubleEqual(actualSparsityRatio, 1.0)) {
        log.trace("All weights are zero, so sparsity ratio is 1. Skipping");
        return false;
    }

    log.trace("Sparsity ratio {0}, threshold {1}", actualSparsityRatio, sparsityRatioThreshold);
    return std::isgreaterequal(actualSparsityRatio, sparsityRatioThreshold);
}

}  // namespace NCESparsity

}  // namespace VPU
}  // namespace vpux
