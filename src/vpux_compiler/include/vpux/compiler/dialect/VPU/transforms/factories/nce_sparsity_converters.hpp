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

using IntOrFloatType = std::variant<int32_t, float>;
using PPEConverterCb = IntOrFloatType (*)(uint8_t, int16_t, double, mlir::Type);
using BiasConverterCb = IntOrFloatType (*)(double, mlir::Type);

PPEConverterCb getPPEConverterCb(VPU::ArchKind arch);
BiasConverterCb getBiasConverterCb(VPU::ArchKind arch);

}  // namespace NCESparsity
}  // namespace VPU
}  // namespace vpux
