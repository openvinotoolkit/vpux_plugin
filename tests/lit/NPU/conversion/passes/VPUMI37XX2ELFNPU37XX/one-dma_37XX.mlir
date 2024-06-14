//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --lower-VPUIP-to-ELF %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

module @OneDMA {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x2x3x4xf16>
  }

  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    // CHECK:       %[[VAL0:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }

    // CHECK:       %[[VAL1:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELFNPU37XX.Section

    // CHECK:       ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput0")
    // CHECK:           ELFNPU37XX.Reloc
    // CHECK:       ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput0")
    // CHECK:           ELFNPU37XX.Reloc

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}

// -----

module @DMANetworkOutputAsInput {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x2x3x4xf16>
  }

  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %input = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[INPUT:.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x2x3x4xf16, @DDR>
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL0:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[INPUT]] : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:       ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELFNPU37XX.Section

    // CHECK:       %[[VAL2:.*]] = ELFNPU37XX.Symbol %arg0 name("input") size(48) : memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[VAL3:.*]] = ELFNPU37XX.Symbol %arg1 name("output") size(48) : memref<1x2x3x4xf16, @DDR>

    // CHECK:       ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section
    // CHECK:           ELFNPU37XX.PutOpInSection %[[VAL2]]
    // CHECK:       %[[VAL4:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section
    // CHECK:           ELFNPU37XX.PutOpInSection %[[VAL3]]

    // CHECK:       ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection(%[[VAL4]])
    // CHECK:           ELFNPU37XX.Reloc
    // CHECK-SAME:        %[[VAL3]]
    // CHECK:           ELFNPU37XX.Reloc
    // CHECK-SAME:        %[[VAL3]]

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}
