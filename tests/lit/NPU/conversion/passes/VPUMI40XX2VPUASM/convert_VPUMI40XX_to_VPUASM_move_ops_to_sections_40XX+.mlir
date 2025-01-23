//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @mainModule attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @oneDma inputsInfo : {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x2x3x4xf16>
  }
  func.func @oneDma() {
    %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, {order = #NHWC}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, {order = #NHWC}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>) inputs(%1 : memref<1x2x3x4xf16, {order = #NHWC}, @DDR>) outputs(%2 : memref<1x2x3x4xf16, {order = #NHWC}, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
    VPUMI40XX.MappedInference dmas((%3) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[1, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

//CHECK: ELF.Main @ELFMain

//CHECK-DAG: ELF.CreateLogicalSection [[MetadataTaskSec:@.*]] aligned(1) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE")
//CHECK-NEXT: VPUASM.DeclareTaskBuffer {{.*}} idx(!VPURegMapped.Index<0:0:0>) <DMA>

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkInput:@.*]] aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT")
//CHECK-NEXT: VPUASM.DeclareBuffer {{.*}} !VPUASM.Buffer< "NetworkInput"[0]

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkOutput:@.*]] aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT")
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
    %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>
    %2 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:2>
    %3 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:0>
    %4 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:1>
    %5 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:2>
    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
    %8 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %13 = VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>) inputs(%6 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64} taskLocation(%1 : !VPURegMapped.Index<0:0:1>) inputs(%6 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %15 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%2 : !VPURegMapped.Index<0:0:2>) inputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%7 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %16 = VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64} taskLocation(%3 : !VPURegMapped.Index<0:1:0>) inputs(%6 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %17 = VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64} taskLocation(%4 : !VPURegMapped.Index<0:1:1>) inputs(%6 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
    %18 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%5 : !VPURegMapped.Index<0:1:2>) inputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%8 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>
    %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
    VPUMI40XX.MappedInference dmas((%13, %16) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) barriers(%11: !VPURegMapped.Index<0:0:0>) dmaCount([[3, 3]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(2) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>)-> !VPURegMapped.Index<0:0:0>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

//CHECK: ELF.Main @ELFMain {

//CHECK-DAG: ELF.CreateLogicalSection [[MetadataSec:@.*]] aligned(1) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE")
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF00:@.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF01:@.*]] idx(!VPURegMapped.Index<0:0:1>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF02:@.*]] idx(!VPURegMapped.Index<0:0:2>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF10:@.*]] idx(!VPURegMapped.Index<0:1:0>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF11:@.*]] idx(!VPURegMapped.Index<0:1:1>) <DMA>
//CHECK-NEXT: VPUASM.DeclareTaskBuffer [[DMATASKBUFF12:@.*]] idx(!VPURegMapped.Index<0:1:2>) <DMA>

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkInput:@.*]] aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT")
//CHECK-NEXT: VPUASM.DeclareBuffer {{.*}} !VPUASM.Buffer< "NetworkInput"[0]

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkOutput0:@.*]] aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT")
//CHECK-NEXT: VPUASM.DeclareBuffer {{.*}} !VPUASM.Buffer< "NetworkOutput"[0]

//CHECK-DAG: ELF.CreateLogicalSection [[NetworkOutput1:@.*]] aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT")
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
