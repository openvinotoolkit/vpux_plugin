//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --move-ops-into-sections %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @mainModule attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @oneDma inputsInfo : {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x2x3x4xf16>
  }
  func.func @oneDma() {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
    VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
    VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
    VPUASM.MappedInference @MappedInference : dmas([[@NNDMA_0_0_0]]) dmaCount([[1, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(0)
    return
  }
}

//CHECK: ELF.Main @ELFMain

//CHECK-DAG: ELF.CreateLogicalSection [[MetadataTaskSec:@.*]] aligned(1) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE")
//CHECK-NEXT: VPUASM.DeclareTaskBuffer {{.*}} idx(!VPURegMapped.Index<0:0:0>) <DMA>

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkInput:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer {{.*}} !VPUASM.Buffer< "NetworkInput"[0]

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkOutput:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer {{.*}} !VPUASM.Buffer< "NetworkOutput"[0]

//CHECK-DAG: ELF.CreateSection [[DMA0SEC:@.*]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
//CHECK-NEXT: VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>)

//CHECK-DAG: ELF.CreateSection [[MappedInferenceSection:@.*]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK-NEXT: VPUASM.MappedInference

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @twoDma inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
    DataInfo "output_1" : tensor<1x16x16x16xf16>
  }
  func.func @twoDma() {
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_10 idx(!VPURegMapped.Index<0:1:0>) <DMA>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_11 idx(!VPURegMapped.Index<0:1:1>) <DMA>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_12 idx(!VPURegMapped.Index<0:1:2>) <DMA>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_00 idx(!VPURegMapped.Index<0:0:0>) <DMA>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_01 idx(!VPURegMapped.Index<0:0:1>) <DMA>
    VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_02 idx(!VPURegMapped.Index<0:0:2>) <DMA>
    VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x16x16x16xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x16x16x16xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x16x16x16xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "CMX_NN"[1] <0> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]> :  swizzling(0)>
    VPUASM.ConfigureBarrier @ConfigureBarrier0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(2 : 2)
    VPUASM.ConfigureBarrier @ConfigureBarrier1 idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(2 : 2)
    VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@DeclareTaskBuffer_DMA_00) links(@NNDMA_0_0_1) input(@DeclareBuffer0) outputs([@DeclareBuffer3]) waits([]) updates([0 : ui8]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
    VPUASM.NNDMA @NNDMA_0_0_1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@DeclareTaskBuffer_DMA_01) links(@NNDMA_0_0_2) input(@DeclareBuffer0) outputs([@DeclareBuffer3]) waits([0 : ui8]) updates([1 : ui8]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
    VPUASM.NNDMA @NNDMA_0_0_2 idx(!VPURegMapped.Index<0:0:2>) taskLocation(@DeclareTaskBuffer_DMA_02) input(@DeclareBuffer3) outputs([@DeclareBuffer1]) waits([1 : ui8]) updates([]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
    VPUASM.NNDMA @NNDMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) taskLocation(@DeclareTaskBuffer_DMA_10) links(@NNDMA_0_1_1) input(@DeclareBuffer0) outputs([@DeclareBuffer4]) waits([]) updates([0 : ui8]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
    VPUASM.NNDMA @NNDMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) taskLocation(@DeclareTaskBuffer_DMA_11) links(@NNDMA_0_1_2) input(@DeclareBuffer0) outputs([@DeclareBuffer4]) waits([0 : ui8]) updates([1 : ui8]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
    VPUASM.NNDMA @NNDMA_0_1_2 idx(!VPURegMapped.Index<0:1:2>) taskLocation(@DeclareTaskBuffer_DMA_12) input(@DeclareBuffer4) outputs([@DeclareBuffer2]) waits([1 : ui8]) updates([]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
    VPUASM.MappedInference @MappedInference : dmas([[@NNDMA_0_0_0, @NNDMA_0_1_0]]) barriers(@ConfigureBarrier0) dmaCount([[3, 3]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(2)
    return
  }
}

//CHECK: ELF.Main @ELFMain {

//CHECK-DAG: ELF.CreateLogicalSection [[MetadataSec:@.*]] aligned(1) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE")
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF10:@.*]] idx(!VPURegMapped.Index<0:1:0>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF11:@.*]] idx(!VPURegMapped.Index<0:1:1>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF12:@.*]] idx(!VPURegMapped.Index<0:1:2>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF00:@.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF01:@.*]] idx(!VPURegMapped.Index<0:0:1>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF02:@.*]] idx(!VPURegMapped.Index<0:0:2>) <DMA>

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkInput:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer {{.*}} !VPUASM.Buffer< "NetworkInput"[0]

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkOutput0:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer {{.*}} !VPUASM.Buffer< "NetworkOutput"[0]

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkOutput1:@.*]] aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT)
//CHECK-NEXT: VPUASM.DeclareBuffer {{.*}} !VPUASM.Buffer< "NetworkOutput"[1]

//CHECK-DAG: ELF.CreateLogicalSection [[NNCMX0:@.*]] aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
//CHECK-NEXT: VPUASM.DeclareBuffer [[BUFF0:@.*]] !VPUASM.Buffer< "CMX_NN"[0]

//CHECK-DAG: ELF.CreateLogicalSection [[NNCMX1:@.*]] aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
//CHECK-NEXT: VPUASM.DeclareBuffer [[BUFF1:@.*]] !VPUASM.Buffer< "CMX_NN"[1]

//CHECK-DAG: ELF.CreateSection [[BARRSEC:@.*]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
//CHECK-NEXT: VPUASM.ConfigureBarrier [[BARR0:@.*]] idx(!VPURegMapped.Index<0:0:0>)
//CHECK-NEXT: VPUASM.ConfigureBarrier [[BARR1:@.*]] idx(!VPURegMapped.Index<0:0:1>)

//CHECK-DAG: ELF.CreateSection [[DMA0SEC:@.*]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK-NEXT: VPUASM.NNDMA [[DMA00:@.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation([[MetadataSec]]::[[DMATASKBUFF00]])
    //CHECK-SAME: outputs([
    //CHECK-SAME: [[NNCMX0]]::[[BUFF0]]])

//CHECK-NEXT: VPUASM.NNDMA [[DMA01:@.*]] idx(!VPURegMapped.Index<0:0:1>) taskLocation([[MetadataSec]]::[[DMATASKBUFF01]])
    //CHECK-SAME: outputs([
    //CHECK-SAME: [[NNCMX0]]::[[BUFF0]]])

//CHECK-NEXT: VPUASM.NNDMA [[DMA02:@.*]] idx(!VPURegMapped.Index<0:0:2>) taskLocation([[MetadataSec]]::[[DMATASKBUFF02]])
    //CHECK-SAME: input([[NNCMX0]]::[[BUFF0]])

//CHECK-DAG: ELF.CreateSection [[DMA1SEC:@.*]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC)
//CHECK-NEXT: VPUASM.NNDMA [[DMA10:@.*]] idx(!VPURegMapped.Index<0:1:0>) taskLocation([[MetadataSec]]::[[DMATASKBUFF10]])
    //CHECK-SAME: outputs([
    //CHECK-SAME: [[NNCMX1]]::[[BUFF1]]])

//CHECK-NEXT: VPUASM.NNDMA [[DMA11:@.*]] idx(!VPURegMapped.Index<0:1:1>) taskLocation([[MetadataSec]]::[[DMATASKBUFF11]])
    //CHECK-SAME: outputs([
    //CHECK-SAME: [[NNCMX1]]::[[BUFF1]]])

//CHECK-NEXT: VPUASM.NNDMA [[DMA12:@.*]] idx(!VPURegMapped.Index<0:1:2>) taskLocation([[MetadataSec]]::[[DMATASKBUFF12]])
    //CHECK-SAME: input([[NNCMX1]]::[[BUFF1]])

//CHECK-DAG: ELF.CreateSection [[MappedInferenceSection:@.*]] aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
//CHECK-NEXT: VPUASM.MappedInference @MappedInference
    //CHECK-SAME: dmas([
    //CHECK-SAME: [
    //CHECK-SAME: [[DMA0SEC]]::[[DMA00]], [[DMA1SEC]]::[[DMA10]]]])
    //CHECK-SAME: barriers([[BARRSEC]]::[[BARR0]])
