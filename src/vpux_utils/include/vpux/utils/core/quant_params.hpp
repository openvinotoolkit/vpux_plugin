//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Quantization parameters
//

#pragma once

#include <string>
#include <unordered_map>
#include "vpux/utils/core/optional.hpp"

namespace vpux {

/**
 * @brief Quantization parameters
 */
struct QuantizationParam {
    explicit QuantizationParam(const float reverseScale = 1.f, const uint8_t zeroPoint = 0)
            : _reverseScale(reverseScale), _zeroPoint(zeroPoint) {
    }
    float _reverseScale;
    uint8_t _zeroPoint;
};

/**
 * @brief Quantization parameters map
 */
using QuantizationParamMap = std::unordered_map<std::string, std::optional<QuantizationParam>>;

}  // namespace vpux
