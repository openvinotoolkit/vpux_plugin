//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {

std::vector<uint8_t> buildProfilingMetadataBuffer(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp, Logger log);

};  // namespace vpux
