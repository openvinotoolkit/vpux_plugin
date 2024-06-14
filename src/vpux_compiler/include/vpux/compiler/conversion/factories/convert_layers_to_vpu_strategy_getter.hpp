//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {

std::unique_ptr<IGreedilyPassStrategy> CreateConvertLayers2VPUStrategy(VPU::ArchKind arch);

}  // namespace vpux
