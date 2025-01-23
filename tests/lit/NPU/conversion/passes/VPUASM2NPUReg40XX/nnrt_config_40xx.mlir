//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUASM-to-NPUReg40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @Test attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 6 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateSection @program.nnrt_config aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {
        VPUASM.nnrtConfig {actShaveRt = @shave.runtime::@ActShaveRt, elfMemOffsetAttrKey = 0 : ui64, isActKernelInvocations} @MappedInference_nnrtConfigManaged : dmaHwpBase(@buffer.CMX_NN.0::@DeclareBuffer6)
      }
    }
    return
  }
}

//CHECK: "NPUReg40XX.NNrtConfig"()
//CHECK-SAME: <{actShaveRt = @shave.runtime::@ActShaveRt
//CHECK-SAME: dmaHwpBase = @buffer.CMX_NN.0::@DeclareBuffer6
//CHECK-SAME: isActKernelInvocations
//CHECK-SAME: sym_name = "MappedInference_nnrtConfigManaged
