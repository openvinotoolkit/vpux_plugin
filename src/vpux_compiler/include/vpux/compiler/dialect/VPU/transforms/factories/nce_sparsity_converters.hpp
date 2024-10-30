//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <mlir/IR/Types.h>

namespace vpux {
namespace VPU {
namespace NCESparsity {

using PPEConverterCb = int32_t (*)(uint8_t, uint16_t, double, mlir::Type);
using BiasConverterCb = int32_t (*)(double, mlir::Type);

PPEConverterCb getPPEConverterCb(VPU::ArchKind arch);
BiasConverterCb getBiasConverterCb(VPU::ArchKind arch);

}  // namespace NCESparsity
}  // namespace VPU
}  // namespace vpux
