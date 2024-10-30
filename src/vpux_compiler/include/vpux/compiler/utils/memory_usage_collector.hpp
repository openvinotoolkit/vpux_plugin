//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {

void addMemoryUsageCollector(mlir::PassManager& pm, Logger log);

};
