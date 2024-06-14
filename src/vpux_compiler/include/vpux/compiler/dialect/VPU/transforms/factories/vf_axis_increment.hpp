//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/vf_axis_increment.hpp"

namespace vpux::VPU {

/*
   Find right class to calculate changes for each axis for VF
*/
std::unique_ptr<IVFAxisIncrement> getVFAxisIncrement(Dim axis);

}  // namespace vpux::VPU
