//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUMI37XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU37XX

IE.CNNNetwork entryPoint : @oneDma inputsInfo : {
    DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x2x3x4xf16>
}

func.func @oneDma() {
    %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, @DDR>
    %3 = VPUMI37XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>) inputs(%1 : memref<1x2x3x4xf16, @DDR>) outputs(%2 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %4 = VPUMI37XX.MappedInference dmas(%3 : !VPURegMapped.Index<0:0:0>) dmaCount([1, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
    return
}

// CHECK: func.func @oneDma()
// CHECK: ELF.CreateLogicalSection @[[SECMETA:.*]] aligned
// CHECK-NEXT: VPUASM.DeclareTaskBuffer @[[TB0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>
// CHECK: ELF.CreateLogicalSection @[[SECIN0:.*]] aligned
// CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF0:.*]] !VPUASM.Buffer< "NetworkInput"[0] <0>
// CHECK: ELF.CreateLogicalSection @[[SECOUT0:.*]] aligned
// CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF1:.*]] !VPUASM.Buffer< "NetworkOutput"[0] <0>
// CHECK: ELF.CreateSection @[[SECDMA00:.*]] aligned
// CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TB0]]) input(@[[SECIN0]]::@[[SYMBUF0]]) outputs([@[[SECOUT0]]::@[[SYMBUF1]]])
// CHECK: VPUASM.MappedInference_37XX @MappedInference : dmas([@[[SECDMA00]]::@[[SYMDMA0]]]) dmaCount([1, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0)
