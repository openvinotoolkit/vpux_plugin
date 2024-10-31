//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --optimize-copies %s | FileCheck %s
// REQUIRES: arch-NPU40XX

IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x30x120x120xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!InputStub_CMX = memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x30x120x120xf16, #NHWC, @DDR>

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
  // CHECK:   [[BUF_1:%.*]] = memref.alloc() : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[BUF_0]] as %arg0: memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs([[BUF_1]] as %arg1: memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]> {
  // CHECK:       VPUIP.Copy inputs(%arg0 : memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   }
  // CHECK:   return [[COPY_0]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
}
// CHECK-LABEL: func.func @NotEraseCMX2CMXCopyAfterSubviewDueToCMXSizeLimitation
// CHECK-SAME:      [[DATA:%.+]]: memref<8000x32xf16>
// CHECK-SAME:      [[INDICES:%.+]]: memref<64000x1xi64, [@CMX_NN, 0]>
func.func @NotEraseCMX2CMXCopyAfterSubviewDueToCMXSizeLimitation(%data : memref<8000x32xf16>, %indices : memref<64000x1xi64, [@CMX_NN, 0]>)
                              -> memref<16000x32xf16, [@CMX_NN, 0]>
{
  %subview_indices = VPUIP.SubView %indices [0, 0] [16000, 1] : memref<64000x1xi64, [@CMX_NN, 0]> to memref<16000x1xi64, [@CMX_NN, 0]>
  %alloc_subview_indices = memref.alloc() : memref<16000x1xi64, [@CMX_NN, 0]>
  %subview_indices_in = VPUIP.Copy inputs(%subview_indices : memref<16000x1xi64, [@CMX_NN, 0]>) outputs(%alloc_subview_indices : memref<16000x1xi64, [@CMX_NN, 0]>) -> memref<16000x1xi64, [@CMX_NN, 0]>
  %alloc_gather = memref.alloc() : memref<16000x32xf16, [@CMX_NN, 0]>
  %gather_out = VPUIP.GatherDMA {channelType = 0 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64} inputs(%data : memref<8000x32xf16>) indices(%subview_indices_in : memref<16000x1xi64, [@CMX_NN, 0]>) outputs(%alloc_gather : memref<16000x32xf16, [@CMX_NN, 0]>) -> memref<16000x32xf16, [@CMX_NN, 0]>
  return %gather_out : memref<16000x32xf16, [@CMX_NN, 0]>

  // CHECK:   [[INDICES_IN:%.+]] = VPUIP.ViewOp [[INDICES]] : memref<64000x1xi64, [@CMX_NN, 0]> to memref<16000x1xi64, [@CMX_NN, 0]>
  // CHECK-NOT:   memref.alloc()
  // CHECK-NOT:   VPUIP.Copy
  // CHECK:   [[ALLOC_GATHER:%.+]] = memref.alloc() : memref<16000x32xf16, [@CMX_NN, 0]>
  // CHECK:   [[GATHER_OUT:%.+]] = VPUIP.GatherDMA {channelType = 0 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64} inputs([[DATA]] : memref<8000x32xf16>) indices([[INDICES_IN]] : memref<16000x1xi64, [@CMX_NN, 0]>) outputs([[ALLOC_GATHER]] : memref<16000x32xf16, [@CMX_NN, 0]>) -> memref<16000x32xf16, [@CMX_NN, 0]>
  // CHECK:   return [[GATHER_OUT]] : memref<16000x32xf16, [@CMX_NN, 0]>
}
