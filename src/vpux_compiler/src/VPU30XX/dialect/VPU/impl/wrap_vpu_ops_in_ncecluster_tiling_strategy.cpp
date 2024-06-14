//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/VPU/impl/wrap_vpu_ops_in_ncecluster_tiling_strategy.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/wrap_vpu_ops_in_ncecluster_tiling.hpp"

using namespace vpux::VPU::arch30xx;

//
// WrapVPUOpsInNCEClusterTilingStrategy
//

void WrapVPUOpsInNCEClusterTilingStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<VPU::NCEConvolutionRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEDepthConvolutionRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEMaxPoolRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEAveragePoolRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEEltwiseRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCESWRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCECompressConvolutionRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEInterpolateRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
}
