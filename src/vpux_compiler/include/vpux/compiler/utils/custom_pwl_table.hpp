//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/small_string.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <string>
#include <unordered_map>

namespace vpux {

struct PWLTableType {
    SmallString activation;
    mlir::Type dtype;
};

struct PWLTableEntry {
    SmallVector<int32_t> range;
    SmallVector<int32_t> shift;
    SmallVector<int32_t> bias;
    std::pair<double, double> floatRange;
    int32_t postShift;
};

struct PWLTableHash {
    std::size_t operator()(const PWLTableType& key) const {
        const auto h1 = std::hash<std::string>()(static_cast<std::string>(key.activation));
        const auto h2 = mlir::hash_value(key.dtype);

        return h1 ^ h2;
    }
};

struct PWLTableEq {
    bool operator()(const PWLTableType& key1, const PWLTableType& key2) const {
        return key1.activation == key2.activation && key1.dtype == key2.dtype;
    }
};

using PWLTableMap = std::unordered_map<PWLTableType, SmallVector<PWLTableEntry>, PWLTableHash, PWLTableEq>;

std::optional<vpux::PWLTableEntry> findCustomPWLTable(IE::PostOpAttr postOp, mlir::Type outElemType);
bool isSupportedNegativeSlope(const float reluSlope);
bool isSupportedPReLU(const float reluSlope, const int64_t zeroPoint);
}  // namespace vpux
