//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/type_interfaces.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/pwl_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace VPU {

double calculateQuantScaleVectorForEltwise(vpux::NDTypeInterface input1ShapedType,
                                           vpux::NDTypeInterface input2ShapedType,
                                           vpux::NDTypeInterface outputShapedType, VPU::ArchKind arch,
                                           bool isMultiplyOp);

bool supportsPerInputEltwiseScale(const VPU::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
