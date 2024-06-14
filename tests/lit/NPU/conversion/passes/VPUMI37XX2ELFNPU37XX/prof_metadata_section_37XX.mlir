//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --lower-VPUIP-to-ELF %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

module @dmaSwProfiling {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x2x3x4xf16>
  } profilingOutputsInfo : {
    DataInfo "profilingOutput" {
      VPUIP.ProfilingSection type 4 : 16 bytes from 0
    } : tensor<4xui32>
  }

  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>, %arg2: memref<4xui32>) -> (memref<1x2x3x4xf16, @DDR>, memref<4xui32>) {
    %profReg = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register>

    %profSlotStart = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1xui64, [@CMX_NN, 0]>
    %profSlotEnd = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> memref<1xui64, [@CMX_NN, 0]>

    %profBufCmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2xui64, [@CMX_NN, 0]>
    %profOutput = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<2xui64>

    %profCmxToOut = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profBufCmx : memref<2xui64, [@CMX_NN, 0]>) outputs(%profOutput : memref<2xui64>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %profDmaEnd = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profReg : memref<1xui64, @Register>) outputs(%profSlotEnd : memref<1xui64, [@CMX_NN, 0]>) nextDMAIdx(%profCmxToOut : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %dma = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) nextDMAIdx(%profDmaEnd : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %profDmaStart = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profReg : memref<1xui64, @Register>) outputs(%profSlotStart : memref<1xui64, [@CMX_NN, 0]>) nextDMAIdx(%dma : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    return %arg1, %arg2 : memref<1x2x3x4xf16, @DDR>, memref<4xui32>

    // CHECK: ELFNPU37XX.CreateProfilingSection secFlags("SHF_NONE") {secAddrAlign = 1 : i64, secInfo = 0 : i64, secName = ".profiling"}
    // CHECK-NEXT: VPUMI37XX.ProfilingMetadata
  }
}
