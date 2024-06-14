//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Pass/PassOptions.h>

#include <string>

namespace vpux {

using IntOption = mlir::detail::PassOptions::Option<int>;
using StrOption = mlir::detail::PassOptions::Option<std::string>;
using BoolOption = mlir::detail::PassOptions::Option<bool>;
using DoubleOption = mlir::detail::PassOptions::Option<double>;

std::optional<int> convertToOptional(const IntOption& intOption);
std::optional<std::string> convertToOptional(const StrOption& strOption);
bool isOptionEnabled(const BoolOption& option);

}  // namespace vpux
