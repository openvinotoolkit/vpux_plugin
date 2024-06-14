//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"

namespace vpux::IE {

bool isLegalQuantizeOp(IE::QuantizeOp quantizeOp, bool canUseCMajor);
bool isLegalDequantizeOp(IE::DequantizeOp dequantizeOp);

}  // namespace vpux::IE
