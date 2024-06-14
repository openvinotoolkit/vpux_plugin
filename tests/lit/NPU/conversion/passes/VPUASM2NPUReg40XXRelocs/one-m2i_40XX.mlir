//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUASM-to-NPUReg40XX-relocs %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

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
        // CHECK:  UINT ls at 32 size 24 = 0x100
        // CHECK:  UINT width at 0 size 16 = 0x7F
        // CHECK:  UINT height at 16 size 16 = 0x54
        // CHECK:  UINT ls at 32 size 24 = 0x80
        // CHECK:  UINT width at 0 size 16 = 0x7F
        // CHECK:  UINT height at 16 size 16 = 0x54
        // CHECK:  UINT ls at 32 size 24 = 0x80
        // CHECK:  UINT inFormat at 0 size 8 = 1
        // CHECK:  UINT outFormat at 8 size 8 = 0x1A
        // CHECK:  UINT numRois at 16 size 16 = 1
        // CHECK:  UINT operations at 36 size 4 = 0xB
        // CHECK:  UINT IFC at 40 size 8 = 1
        // CHECK:  UINT IRQMask at 48 size 16 = 0x8000
        // CHECK:  UINT NormFact0 at 0 size 16 = 0x4900
        // CHECK:  UINT NormFact1 at 16 size 16 = 0x4980
        // CHECK:  UINT NormFact2 at 32 size 16 = 0x4A00
        // CHECK:  UINT NormFact3 at 48 size 16 = 0x4A80
        // CHECK:  UINT NormFact0 at 0 size 16 = 0x4D00
        // CHECK:  UINT NormFact1 at 16 size 16 = 0x4D40
        // CHECK:  UINT NormFact2 at 32 size 16 = 0x4D80
        // CHECK:  UINT NormFact3 at 48 size 16 = 0x4DC0
        // CHECK:  UINT NormFact0 at 0 size 16 = 0x4F80
        // CHECK:  UINT NormFact1 at 16 size 16 = 0x4FC0
        // CHECK:  UINT NormFact2 at 32 size 16 = 0x5000
        // CHECK:  UINT NormFact3 at 48 size 16 = 0x5020
        // CHECK:  UINT inPS at 0 size 32 = 0xAA00
        // CHECK:  UINT outFormatLocal at 32 size 8 = 0x1A
        // CHECK:  UINT OFC at 44 size 8 = 2
        // CHECK:  UINT X_coord at 0 size 16 = 0
        // CHECK:  UINT Y_coord at 16 size 16 = 0
        // CHECK:  UINT roiWidth at 32 size 16 = 0xFF
        // CHECK:  UINT roiHeight at 48 size 16 = 0xA9
        // CHECK:  UINT outScale0_width at 0 size 16 = 0xFF
        // CHECK:  UINT outScale0_height at 16 size 16 = 0xFF
        // CHECK:  UINT outScale1_width at 32 size 16 = 0
        // CHECK:  UINT outScale1_height at 48 size 16 = 0
        // CHECK:  UINT psSc0Y at 0 size 32 = 0x30000
        // CHECK:  UINT psSc1Y at 32 size 32 = 0
        // CHECK:  UINT psSc0UV at 0 size 32 = 0
        // CHECK:  UINT psSc1UV at 32 size 32 = 0
        // CHECK:  UINT lsSc0Y at 0 size 16 = 0x300
        // CHECK:  UINT lsSc1Y at 16 size 16 = 0
        // CHECK:  UINT lsSc0UV at 32 size 16 = 0
        // CHECK:  UINT lsSc1UV at 48 size 16 = 0
        // CHECK:  UINT vSc_offset at 0 size 32 = 0
        // CHECK:  UINT hSc_offset at 32 size 32 = 0
        // CHECK:  UINT vSc_factor at 0 size 32 = 0x20000
        // CHECK:  UINT hSc_factor at 32 size 32 = 0x20000
        // CHECK:  UINT start_after_ at 0 size 32 = 1
        // CHECK:  UINT clean_after_ at 32 size 32 = 2
      }
    }
    return
  }
}
