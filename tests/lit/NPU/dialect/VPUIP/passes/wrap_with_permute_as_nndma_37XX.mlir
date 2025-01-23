//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --wrap-with-permute-as-nndma %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x1x9x9xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_DepthToSpaceOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// Case 1: Do not wrap DepthToSpaceOp as MultiClusterDepthToSpaceDMA with single-cluster input and multi-cluster(SEGMENTED) output
// CHECK-LABEL: @NotWrapDepthToSpaceAsMultiClusterDMA
func.func @NotWrapDepthToSpaceAsMultiClusterDMA(%arg0: memref<1x9x3x3xf16, #NHWC, [@CMX_NN, 0]>)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x1x9x9xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%arg0 as %arg1: memref<1x9x3x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x1x9x9xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x9x9xf16, #NHWC, [@CMX_NN, 0]> {
       VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x9x3x3xf16, #NHWC, [@CMX_NN, 0]>, memref<1x1x9x9xf16, #NHWC, [@CMX_NN, 0]>
    }
    %2 = memref.alloc() : memref<1x1x9x9xf16, #NHWC>
    %3 = VPUIP.Copy inputs(%1 : memref<1x1x9x9xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x9x9xf16, #NHWC>) -> memref<1x1x9x9xf16, #NHWC>
    %4 = VPURT.AllocDistributed -> !OutputDistributedType
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x1x9x9xf16, #NHWC>) outputs(%4 as %arg2: memref<1x1x9x9xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
       %6 = VPUIP.Copy inputs(%arg1 : memref<1x1x9x9xf16, #NHWC>) outputs(%arg2 : memref<1x1x9x9xf16, #NHWC, @CMX_NN>) -> memref<1x1x9x9xf16, #NHWC, @CMX_NN>
    }

    return %5: !OutputDistributedType

    // CHECK: [[OUT_MEMREF:%.*]] = memref.alloc() : memref<1x1x9x9xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[RESULT:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%arg0 as %arg1: memref<1x9x3x3xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUT_MEMREF]] as %arg2: memref<1x1x9x9xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x9x9xf16, #NHWC, [@CMX_NN, 0]>
}
