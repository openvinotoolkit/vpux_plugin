//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module @UpsamplingDMAOp {
  func.func @main(%arg0: memref<16x256xf16, @DDR>, %arg1: memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<16x256xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<256x16xf16, @DDR>
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %4 =  VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) -> memref<256x16xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %6 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) -> memref<256x16xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %7 =  VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %8 =  VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    return %arg1 : memref<16x256xf16, @DDR>
  }
}

// CHECK-NOT: VPUIP.NNDMA
// CHECK: %[[VAL5:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: %[[VAL4:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) nextDMAIdx(%[[VAL5]] : !VPURegMapped.Index<0:0:5>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
// CHECK-NOT: VPUIP.UpsamplingDMAOp
// CHECK: %[[VAL3:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) nextDMAIdx(%[[VAL4]] : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
// CHECK-NOT: VPUIP.UpsamplingDMAOp
// CHECK: %[[VAL2:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) nextDMAIdx(%[[VAL3]] : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: %[[VAL1:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) nextDMAIdx(%[[VAL2]] : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: %[[VAL0:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) nextDMAIdx(%[[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

// -----

module @UpsamplingDMAOp {
  func.func @main(%arg0: memref<16x256xf16, @DDR>, %arg1: memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<16x256xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<256x16xf16, @DDR>

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %2 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) -> memref<256x16xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %3 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) -> memref<256x16xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %5 =  VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %6 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) -> memref<256x16xf16, @DDR>
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %7 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) -> memref<256x16xf16, @DDR>
    }

    return %arg1 : memref<16x256xf16, @DDR>
  }
}

// CHECK-NOT: VPUIP.UpsamplingDMAOp
// CHECK: %[[VAL5:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>
// CHECK-NOT: VPUIP.UpsamplingDMAOp
// CHECK: %[[VAL4:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) nextDMAIdx(%[[VAL5]] : !VPURegMapped.Index<0:0:5>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: %[[VAL3:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) nextDMAIdx(%[[VAL4]] : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
// CHECK-NOT: VPUIP.NNDMA
// CHECK: %[[VAL2:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) nextDMAIdx(%[[VAL3]] : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
// CHECK-NOT: VPUIP.UpsamplingDMAOp
// CHECK: %[[VAL1:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) nextDMAIdx(%[[VAL2]] : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
// CHECK-NOT: VPUIP.UpsamplingDMAOp
// CHECK: %[[VAL0:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 16 : i64, len = 512 : i64, srcWidth = 512 : i64, srcStride = 2 : i64, srcPlaneStride = 512 : i64, dstWidth = 2 : i64, dstStride = 32 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<256x16xf16, @DDR>) nextDMAIdx(%[[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
