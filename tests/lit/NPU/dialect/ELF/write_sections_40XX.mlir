//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --move-ops-into-sections %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

module @Test attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "softmax" : tensor<1x1000xf16>
  }
  module @VPU.SW {
    func.func private @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x1x1000xf16> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @softmax_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x1x1000xf16> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x1x1000xf16> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x1x1000xf16> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "DDR"[0] <0> : memref<1048576xi8, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "CMX_NN"[0] <2000> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareKernelText @DeclareKernelText_0_0 : "softmax"
    VPUASM.DeclareKernelEntry @DeclareKernelEntry_0_0 : "softmax"
    VPUASM.DeclareKernelData @DeclareKernelArgs_0_0 : "softmax"
    VPUASM.KernelParams @KernelParams_0_0 inputs([@DeclareBuffer3]) outputs([@DeclareBuffer4]) kernel_type("softmax") kernel_params(dense<[0, 0, 32, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<80xui8>)
    VPUASM.ConfigureBarrier @ConfigureBarrier_0_0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(1 : 1)
    VPUASM.ConfigureBarrier @ConfigureBarrier_0_1 idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(1 : 1)
    VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@DeclareTaskBuffer_DMA_0_0_0) input(@DeclareBuffer0) outputs([@DeclareBuffer3]) waits([]) updates([0 : ui8]) start_after(1) clean_after(0) descriptor(<numPlanes = 0 : i32, len = 2000 : i32, srcWidth = 2000 : i32, srcStride = 2000 : i32, srcPlaneStride = 0 : i32, dstWidth = 2000 : i32, dstStride = 2000 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>) tile_indexes([0])
    VPUASM.NNDMA @NNDMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) taskLocation(@DeclareTaskBuffer_DMA_0_1_0) input(@DeclareBuffer4) outputs([@DeclareBuffer1]) waits([1 : ui8]) updates([]) start_after(2) clean_after(2) descriptor(<numPlanes = 0 : i32, len = 2000 : i32, srcWidth = 2000 : i32, srcStride = 2000 : i32, srcPlaneStride = 0 : i32, dstWidth = 2000 : i32, dstStride = 2000 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
    VPUASM.ActKernelRange @ActKernelRange_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@DeclareTaskBuffer_ActKernelRange_0_0_0) kernelTaskType(@COMPUTE) calls @DeclareKernelText_0_0 : @DeclareKernelEntry_0_0
    VPUASM.ActKernelInvocation @ActKernelInvocation_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@DeclareTaskBuffer_ActKernelInvocation_0_0_0) -> @DeclareTaskBuffer_ActKernelRange_0_0_0(kernel_data : @DeclareKernelArgs_0_0, kernel_params : @KernelParams_0_0) waits([0 : ui8]) updates([1 : ui8]) tile(0) start_after(2) clean_after(1) range_index(0)
    VPUASM.MappedInference @MappedInference : dmas([[@NNDMA_0_0_0, @NNDMA_0_1_0]]) actKernelRanges([@ActKernelRange_0_0]) actKernelInvocations([@ActKernelInvocation_0_0]) barriers(@ConfigureBarrier_0_0) actShaveRt(@DeclareBuffer2) dmaCount([[1, 1], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([1, 0, 0, 0, 0, 0]) actKernelInvocationsCount([1, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(2)
    return
  }
}

// CHECK: ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC")
// CHECK: ELF.CreateSection @shave.data aligned(1024) secType(SHT_PROGBITS) secFlags("SHF_WRITE|SHF_ALLOC")
