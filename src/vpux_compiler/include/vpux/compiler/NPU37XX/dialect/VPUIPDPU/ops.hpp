//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.hpp.inc>

namespace vpux::VPUIPDPU::arch37xx {

void registerVerifiersOpInterfaces(mlir::DialectRegistry&);

}  // namespace vpux::VPUIPDPU::arch37xx
