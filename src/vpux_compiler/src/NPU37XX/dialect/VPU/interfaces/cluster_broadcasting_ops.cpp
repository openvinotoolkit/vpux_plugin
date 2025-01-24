//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/cluster_broadcasting_utils.hpp"

void vpux::VPU::arch37xx::registerClusterBroadcastingOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEConvolutionOp::attachInterface<vpux::VPU::ClusterBroadcastingOpModelNCEOp>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<vpux::VPU::ClusterBroadcastingOpModelNCEOp>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<vpux::VPU::ClusterBroadcastingOpModelNCEOp>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<vpux::VPU::ClusterBroadcastingOpModelNCEOp>(*ctx);
        VPU::NCEEltwiseOp::attachInterface<vpux::VPU::ClusterBroadcastingOpModelNCEOp>(*ctx);
        VPU::NCEPermuteOp::attachInterface<vpux::VPU::ClusterBroadcastingOpModelNCEOp>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<vpux::VPU::ClusterBroadcastingOpModelNCEOp>(*ctx);
        VPU::NCEMatMulOp::attachInterface<vpux::VPU::ClusterBroadcastingOpModelNCEOp>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<vpux::VPU::ClusterBroadcastingOpModelNCEOp>(*ctx);
    });
}
