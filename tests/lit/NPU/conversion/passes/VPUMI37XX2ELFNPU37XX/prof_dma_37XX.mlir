//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --lower-VPUIP-to-ELF %s | FileCheck %s
// REQUIRES: arch-NPU37XX

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
    %dma = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) nextDMAIdx(%profDmaEnd : !VPURegMapped.Index<0:0:2>)  start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %profDmaStart = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profReg : memref<1xui64, @Register>) outputs(%profSlotStart : memref<1xui64, [@CMX_NN, 0]>) nextDMAIdx(%dma : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    return %arg1, %arg2 : memref<1x2x3x4xf16, @DDR>, memref<4xui32>

    // CHECK:       [[DMA_PROF_TO_OUT:%.*]] = VPUMI37XX.NNDMA
    // CHECK:       [[DMA_PROF_END:%.*]] = VPUMI37XX.NNDMA
    // CHECK:       [[DMA:%.*]] = VPUMI37XX.NNDMA
    // CHECK:       [[DMA_PROF_START:%.*]] = VPUMI37XX.NNDMA

    // CHECK:       [[SYM_IN:%.*]] = ELFNPU37XX.Symbol %arg0 name("input") size(48) : memref<1x2x3x4xf16, @DDR>
    // CHECK:       [[SYM_OUT:%.*]] = ELFNPU37XX.Symbol %arg1 name("output") size(48) : memref<1x2x3x4xf16, @DDR>
    // CHECK:       [[SYM_PROF:%.*]] = ELFNPU37XX.Symbol %arg2 name("profilingOutput") size(16) : memref<4xui32>
    // CHECK:       [[SYMTAB_IN:%.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section {
    // CHECK-NEXT:         ELFNPU37XX.PutOpInSection [[SYM_IN]] : !ELFNPU37XX.Symbol
    // CHECK:       [[SYMTAB_OUT:%.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section {
    // CHECK-NEXT:         ELFNPU37XX.PutOpInSection [[SYM_OUT]] : !ELFNPU37XX.Symbol
    // CHECK:       [[SYMTAB_PROF:%.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.prof_output") secFlags(VPU_SHF_PROFOUTPUT) -> !ELFNPU37XX.Section {
    // CHECK-NEXT:         ELFNPU37XX.PutOpInSection [[SYM_PROF]] : !ELFNPU37XX.Symbol

    // CHECK:       [[RELOCSEC_PROF:%.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_ProfOutput0") sourceSymbolTableSection([[SYMTAB_PROF]])
    // CHECK-NEXT:         ELFNPU37XX.Reloc baseOp([[DMA_PROF_TO_OUT]] : !VPURegMapped.Index<0:0:3>) offset(24) <R_VPU_64> [[SYM_PROF]] 0

    // CHECK:       return %arg1, %arg2 : memref<1x2x3x4xf16, @DDR>, memref<4xui32>
  }
}
