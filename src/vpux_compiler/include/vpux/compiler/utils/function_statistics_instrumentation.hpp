//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Pass/PassManager.h>
#include "vpux/utils/core/logger.hpp"

namespace vpux {

void addFunctionStatisticsInstrumentation(mlir::PassManager& pm, const Logger& log);

};
