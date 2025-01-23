//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUASM-to-NPUReg40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @OneM2IWithoutAttributes {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x256x256x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x256x256x4xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.M2I0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_M2I_0 idx(!VPURegMapped.Index<0:0:0>) <M2I>
      }
      ELF.CreateSection @text.dma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.M2I @M2I_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.M2I0::@DeclareTaskBuffer_M2I_0) inputs(@DeclareBuffer0) outputs(@DeclareBuffer1) {clean_after = 2 : ui64, do_norm, inFmt = #VPU.m2i_color_fmt<PL_YUV420_8>, norm = [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01], outFmt = #VPU.m2i_color_fmt<IL_RGB888>, chroma_out_reverse_channels, scale_factor_x = 131072 : ui32, scale_factor_y = 131072 : ui32, start_after = 1 : ui64, updateBarriers = [], waitBarriers = []}
        // CHECK-NOT:   VPUASM.M2I
        // CHECK:       NPUReg40XX.M2I
        // CHECK:  UINT ls = 0x100
        // CHECK:  UINT width = 0x7F
        // CHECK:  UINT height = 0x54
        // CHECK:  UINT ls = 0x80
        // CHECK:  UINT width = 0x7F
        // CHECK:  UINT height = 0x54
        // CHECK:  UINT ls = 0x80
        // CHECK:  UINT inFormat = 1
        // CHECK:  UINT outFormat = 0x1A
        // CHECK:  UINT numRois = 1
        // CHECK:  UINT operations = 0xB
        // CHECK:  UINT IFC = 1
        // CHECK:  UINT IRQMask = 0x8000
        // CHECK:  UINT NormFact0 = 0x4900
        // CHECK:  UINT NormFact1 = 0x4980
        // CHECK:  UINT NormFact2 = 0x4A00
        // CHECK:  UINT NormFact3 = 0x4A80
        // CHECK:  UINT NormFact0 = 0x4D00
        // CHECK:  UINT NormFact1 = 0x4D40
        // CHECK:  UINT NormFact2 = 0x4D80
        // CHECK:  UINT NormFact3 = 0x4DC0
        // CHECK:  UINT NormFact0 = 0x4F80
        // CHECK:  UINT NormFact1 = 0x4FC0
        // CHECK:  UINT NormFact2 = 0x5000
        // CHECK:  UINT NormFact3 = 0x5020
        // CHECK:  UINT inPS = 0xAA00
        // CHECK:  UINT outFormatLocal = 0x1A
        // CHECK:  UINT OFC = 2
        // CHECK:  UINT X_coord = 0
        // CHECK:  UINT Y_coord = 0
        // CHECK:  UINT roiWidth = 0xFF
        // CHECK:  UINT roiHeight = 0xA9
        // CHECK:  UINT outScale0_width = 0xFF
        // CHECK:  UINT outScale0_height = 0xFF
        // CHECK:  UINT outScale1_width = 0
        // CHECK:  UINT outScale1_height = 0
        // CHECK:  UINT psSc0Y = 0x30000
        // CHECK:  UINT psSc1Y = 0
        // CHECK:  UINT psSc0UV = 0
        // CHECK:  UINT psSc1UV = 0
        // CHECK:  UINT lsSc0Y = 0x300
        // CHECK:  UINT lsSc1Y = 0
        // CHECK:  UINT lsSc0UV = 0
        // CHECK:  UINT lsSc1UV = 0
        // CHECK:  UINT vSc_offset = 0
        // CHECK:  UINT hSc_offset = 0
        // CHECK:  UINT vSc_factor = 0x20000
        // CHECK:  UINT hSc_factor = 0x20000
        // CHECK:  UINT start_after_ = 1
        // CHECK:  UINT clean_after_ = 2
      }
    }
    return
  }
}
