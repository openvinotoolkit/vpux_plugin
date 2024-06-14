//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --move-io-buffers-to-elf-sections %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.CNNNetwork entryPoint : @multiple_clusters_dpu_soh_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x32x32xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x16x32xf16>
    DataInfo "output_1" : tensor<1x64x16x32xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @output_1_buffDecl !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @multiple_clusters_dpu_soh_f16_f16_f16() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "NetworkInput"[0] <32768> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
    }
    return
  }
}

//CHECK: ELF.Main @ELFMain

//CHECK-DAG: ELF.CreateLogicalSection [[I0SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT) {
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "NetworkInput"[0] <32768> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>

//CHECK-DAG: ELF.CreateLogicalSection [[O0SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) {
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>

//CHECK-DAG: ELF.CreateLogicalSection [[O1SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) {
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>

//CHECK-DAG: ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags(VPU_SHF_USERINPUT) {
//CHECK-NEXT: ELF.Symbol {{.*}} of([[I0SECNAME]]) type(<STT_SECTION>) size(65536) value(0)

//CHECK-DAG: ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags(VPU_SHF_USEROUTPUT) {
//CHECK-NEXT: ELF.Symbol {{.*}} of([[O0SECNAME]]) type(<STT_SECTION>) size(65536) value(0)
//CHECK-NEXT: ELF.Symbol {{.*}} of([[O1SECNAME]]) type(<STT_SECTION>) size(65536) value(0)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.CNNNetwork entryPoint : @multiple_clusters_dpu_soh_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x32x32xf16>
    DataInfo "input_1" : tensor<1x32x32x32xf16>
    DataInfo "input_2" : tensor<1x32x32x32xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x16x32xf16>
    DataInfo "output_1" : tensor<1x64x16x32xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @input_1_buffDecl !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x32x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @input_2_buffDecl !VPUASM.Buffer< "NetworkInput"[2] <0> : memref<1x32x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @output_1_buffDecl !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @multiple_clusters_dpu_soh_f16_f16_f16() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "NetworkInput"[0] <32768> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer7 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer8 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer9 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer10 !VPUASM.Buffer< "NetworkInput"[2] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer11 !VPUASM.Buffer< "NetworkInput"[2] <32> : memref<1x1x1x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer12 !VPUASM.Buffer< "NetworkInput"[2] <64> : memref<1x1x1x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer13 !VPUASM.Buffer< "NetworkInput"[2] <128> : memref<1x1x1x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
    }
    return
  }
}

//CHECK: ELF.Main @ELFMain

//CHECK-DAG: ELF.CreateLogicalSection [[I0SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "NetworkInput"[0] <32768> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>

//CHECK-DAG: ELF.CreateLogicalSection [[I1SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer8 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer9 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>

//CHECK-DAG: ELF.CreateLogicalSection [[I2SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer10 !VPUASM.Buffer< "NetworkInput"[2] <0> : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer11 !VPUASM.Buffer< "NetworkInput"[2] <32> : memref<1x1x1x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer12 !VPUASM.Buffer< "NetworkInput"[2] <64> : memref<1x1x1x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer13 !VPUASM.Buffer< "NetworkInput"[2] <128> : memref<1x1x1x32xf16, #NHWC, [@DDR, 0]> :  swizzling(0)>

//CHECK-DAG: ELF.CreateLogicalSection [[O0SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>

//CHECK-DAG: ELF.CreateLogicalSection [[O1SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer7 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x64x16x32xf16, #NHWC, @DDR> :  swizzling(0)>

//CHECK-DAG: ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags(VPU_SHF_USERINPUT)
//CHECK-NEXT: ELF.Symbol {{.*}} of([[I0SECNAME]]) type(<STT_SECTION>) size(65536) value(0)
//CHECK-NEXT: ELF.Symbol {{.*}} of([[I1SECNAME]]) type(<STT_SECTION>) size(65536) value(0)
//CHECK-NEXT: ELF.Symbol {{.*}} of([[I2SECNAME]]) type(<STT_SECTION>) size(65536) value(0)

//CHECK-DAG: ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags(VPU_SHF_USEROUTPUT)
//CHECK-NEXT: ELF.Symbol {{.*}} of([[O0SECNAME]]) type(<STT_SECTION>) size(65536) value(0)
//CHECK-NEXT: ELF.Symbol {{.*}} of([[O1SECNAME]]) type(<STT_SECTION>) size(65536) value(0)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x1x15x15xf16>
  } outputsInfo : {
    DataInfo "pool1" : tensor<1x1x14x14xf16>
  } profilingOutputsInfo : {
    DataInfo "0_dpu_32_pad_64_pll" : tensor<32xui32>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @data_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x15x15xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @pool1_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x14x14xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
    VPUASM.DeclareBuffer @"0_dpu_32_pad_64_pll_buffDecl" !VPUASM.Buffer< "ProfilingOutput"[0] <0> : memref<32xui32> :  swizzling(0)>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x15x15xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x15x15xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x14x14xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x14x14xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "ProfilingOutput"[0] <0> : memref<32xui32> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "ProfilingOutput"[0] <0> : memref<4xui64> :  swizzling(0)>
    }
    return
  }
}

//CHECK: ELF.Main @ELFMain

//CHECK-DAG: ELF.CreateLogicalSection [[I0SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT) {
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x15x15xf16, @DDR> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x15x15xf16, #NHWC, @DDR> :  swizzling(0)>

//CHECK-DAG: ELF.CreateLogicalSection [[O0SECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) {
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x14x14xf16, @DDR> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x14x14xf16, #NHWC, @DDR> :  swizzling(0)>

//CHECK-DAG: ELF.CreateLogicalSection [[POSECNAME:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_PROFOUTPUT) {
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "ProfilingOutput"[0] <0> : memref<32xui32> :  swizzling(0)>
//CHECK-NEXT: VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "ProfilingOutput"[0] <0> : memref<4xui64> :  swizzling(0)>

//CHECK-DAG: ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags(VPU_SHF_USERINPUT) {
//CHECK-NEXT: ELF.Symbol {{.*}} of([[I0SECNAME]]) type(<STT_SECTION>) size(450) value(0)

//CHECK-DAG: ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags(VPU_SHF_USEROUTPUT) {
//CHECK-NEXT: ELF.Symbol {{.*}} of([[O0SECNAME]]) type(<STT_SECTION>) size(392) value(0)

//CHECK-DAG: ELF.CreateSymbolTableSection @symtab.io.ProfilingOutput secFlags(VPU_SHF_PROFOUTPUT) {
//CHECK-NEXT: ELF.Symbol {{.*}} of([[POSECNAME]]) type(<STT_SECTION>) size(128) value(0)
