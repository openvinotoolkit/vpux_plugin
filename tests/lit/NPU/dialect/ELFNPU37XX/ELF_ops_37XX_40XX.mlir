//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @SingleLayer attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>}  {
  module @UsedMemory  {
    IE.MemoryResource 2048 bytes of @DDR
  }
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1000xf16>
  } outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1000xf16>
  }
  func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    // CHECK:       %[[VAL0:.*]] = ELFNPU37XX.CreateSection secType(SHT_LOUSER) secFlags(SHF_ALLOC) {secAddrAlign = 4 : i64, secInfo = 1 : i64, secName = ".data.Weights"} -> !ELFNPU37XX.Section
    %0 = ELFNPU37XX.CreateSection secType(SHT_LOUSER) secFlags(SHF_ALLOC) {secAddrAlign = 4 : i64, secInfo = 1 : i64, secName = ".data.Weights"} -> !ELFNPU37XX.Section  {
    }
    // CHECK:    %[[VAL1:.*]] = ELFNPU37XX.Symbol %[[VAL2:.*]] name("outputCNN") : memref<1x1000xf16>
    %1 = ELFNPU37XX.Symbol %arg1 name("outputCNN") : memref<1x1000xf16>
    // CHECK:    %[[VAL3:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".output.symbolTableSection") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section
    %2 = ELFNPU37XX.CreateSymbolTableSection secName(".output.symbolTableSection") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section  {
      // CHECK:      ELFNPU37XX.PutOpInSection %[[VAL1]] : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %1 : !ELFNPU37XX.Symbol
    }
    // CHECK:    %[[VAL4:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rela.output") sourceSymbolTableSection(%[[VAL3]]) targetSection(%[[VAL0]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT")
    %3 = ELFNPU37XX.CreateRelocationSection secName(".rela.output") sourceSymbolTableSection(%2) targetSection(%0) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section  {
      // CHECK:      ELFNPU37XX.Reloc offset(24) <R_VPU_64> %[[VAL1]] 0
      ELFNPU37XX.Reloc offset(24) <R_VPU_64> %1 0
    }
    // CHECK:    return %[[VAL2]] : memref<1x1000xf16>
    return %arg1 : memref<1x1000xf16>
  }
}
