//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-copies %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x30x120x120xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!InputStub_CMX = memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x3x120x120xf16, #NHWC, @DDR>

func.func @NotFuseCMXCopyToTheFrontOfTillingCopyDueToCMXSizeLimitation() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !SpilledOutput_DDR
  %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg1: !SpilledOutput_DDR) -> !SpilledOutput_DDR {
      VPUIP.Copy inputs(%arg0: memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs(%arg1: !SpilledOutput_DDR) -> !SpilledOutput_DDR
  }

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !SpilledOutput_DDR) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x120x120xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[BUF_1:%.*]] = memref.alloc() : memref<1x3x120x120xf16, #NHWC, @DDR>
  // CHECK:   [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[BUF_0]] as %arg0: memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs([[BUF_1]] as %arg1: memref<1x3x120x120xf16, #NHWC, @DDR>) -> memref<1x3x120x120xf16, #NHWC, @DDR> {
  // CHECK:       VPUIP.Copy inputs(%arg0 : memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x3x120x120xf16, #NHWC, @DDR>) -> memref<1x3x120x120xf16, #NHWC, @DDR>
  // CHECK:   }
  // CHECK:   [[BUF_2:%.*]] = memref.alloc() : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   [[COPY_1:%.*]] = VPUIP.Copy inputs([[COPY_0]] : memref<1x3x120x120xf16, #NHWC, @DDR>) outputs([[BUF_2]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   return [[COPY_1]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
}
