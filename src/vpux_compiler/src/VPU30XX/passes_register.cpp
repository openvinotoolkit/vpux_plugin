//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/passes_register.hpp"
#include "vpux/compiler/VPU30XX/conversion.hpp"
#include "vpux/compiler/VPU30XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/transforms/passes.hpp"

using namespace vpux;

//
// PassesRegistry30XX::registerPasses
//

void PassesRegistry30XX::registerPasses() {
    vpux::arch30xx::registerConversionPasses();
    vpux::IE::arch30xx::registerIEPasses();
    vpux::VPUIP::arch30xx::registerVPUIPPasses();
}
