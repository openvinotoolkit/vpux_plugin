//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @spaceToDepth {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x1x16x256xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x1x16x256xf16>
  }
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL0:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL1:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL0]] : !VPURegMapped.Index<0:1:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL2:.*]] = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL1]] : !VPURegMapped.Index<0:1:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %6 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL3:.*]] = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL2]] : !VPURegMapped.Index<0:1:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:3>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL4:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL3]] : !VPURegMapped.Index<0:1:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:4>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL5:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL4]] : !VPURegMapped.Index<0:1:4>)  start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @spaceToDepth {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x1x16x256xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x1x16x256xf16>
  }
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %2 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL0:.*]] = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %3 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL1:.*]] = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL0]] : !VPURegMapped.Index<0:1:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL2:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL1]] : !VPURegMapped.Index<0:1:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL3:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL2]] : !VPURegMapped.Index<0:1:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:3>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %6 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL4:.*]] = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL3]] : !VPURegMapped.Index<0:1:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:4>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %7 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.SpaceToDepth
    // CHECK: %[[VAL5:.*]] = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 7 : i64, len = 1792 : i64, srcWidth = 256 : i64, srcStride = 512 : i64, srcPlaneStride = 3584 : i64, dstWidth = 1792 : i64, dstStride = 1 : i64, dstPlaneStride = 3584 : i64>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL4]] : !VPURegMapped.Index<0:1:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}
