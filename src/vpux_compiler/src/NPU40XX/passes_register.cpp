//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/passes_register.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"

using namespace vpux;

//
// PassesRegistry40XX::registerPasses
//

void PassesRegistry40XX::registerPasses() {
    vpux::IE::arch37xx::registerIEPasses();

    vpux::VPU::arch37xx::registerAdjustForOptimizedSwKernelPass();
    vpux::VPU::arch37xx::registerDecomposeMVNPass();
    vpux::VPU::arch37xx::registerSplitRealDFTOpsPass();
    vpux::VPU::arch37xx::registerAddProposalAuxiliaryBufferPass();
    vpux::VPU::arch40xx::registerVPUPasses();

    vpux::VPUIP::arch37xx::registerAddSwKernelCacheHandlingOpsPass();
    vpux::VPUIP::arch40xx::registerVPUIPPasses();

    vpux::arch40xx::registerConversionPasses();

    vpux::VPURT::arch37xx::registerVPURTPasses();
    vpux::VPURT::arch40xx::registerVPURTPasses();
}
