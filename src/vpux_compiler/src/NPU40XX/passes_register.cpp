//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/passes_register.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
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
    vpux::IE::arch37xx::registerConvertFFTToConv();
    vpux::IE::arch37xx::registerConvertToMixedPrecision();
    vpux::IE::arch37xx::registerFuseOutstandingDequant();
    vpux::IE::arch37xx::registerConvertWeightsToI8();
    vpux::IE::arch37xx::registerExpandActivationChannelsPass();  //
    vpux::IE::arch37xx::registerFuseStaticScale();
    vpux::IE::arch37xx::registerFusePermuteQuantizeExpand();
    vpux::IE::arch37xx::registerFuseReordersPass();
    vpux::IE::arch37xx::registerInsertIdentityPoolBeforeOp();
    vpux::IE::arch37xx::registerOptimizeNetworkInputConvert();
    vpux::IE::arch37xx::registerOptimizeSliceExpand();
    vpux::IE::arch37xx::registerProcessAsymmetricZeroPointsForConvolution();
    vpux::IE::arch37xx::registerPropagateExpand();
    vpux::IE::arch37xx::registerPropagateReorderToNCE();
    vpux::IE::arch37xx::registerSwapMaxPoolWithActivation();
    vpux::IE::arch37xx::registerUnrollBatch();
    vpux::IE::arch37xx::registerConvertSubGRUSequenceToConv();

    vpux::IE::arch40xx::registerIEPasses();
    vpux::VPU::arch37xx::registerAdjustForOptimizedLayersPass();
    vpux::VPU::arch37xx::registerDecomposeMVNPass();
    vpux::VPU::arch37xx::registerApplyTilingMVN1SumPass();
    vpux::VPU::arch37xx::registerSplitRealDFTOpsPass();
    vpux::VPU::arch37xx::registerAddProposalAuxiliaryBufferPass();
    vpux::VPU::arch40xx::registerVPUPasses();

    vpux::VPUIP::arch37xx::registerAddSwKernelCacheHandlingOpsPass();
    vpux::VPUIP::arch40xx::registerVPUIPPasses();

    vpux::arch40xx::registerConversionPasses();

    vpux::VPURT::arch37xx::registerVPURTPasses();
    vpux::VPURT::arch40xx::registerVPURTPasses();
}
