//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUASM-to-NPUReg40XX-relocs %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @SingleHswishFP16 attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @single_hswish inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "hswish" : tensor<1x1000xf16>
  }
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
    module @VPU.SW {
      func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x1x1000xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x1x1000xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @single_hswish() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @io.NetworkInput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT) {
        VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x1x1000xf16> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkOutput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) {
        VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1x1x1000xf16> :  swizzling(0)>
      }
      VPUASM.DeclareKernelEntry @DeclareKernelEntry0 : "activation_hswish"
      ELF.CreateLogicalSection @program.ActKernelRange.cmx.0.0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelRange>
      }
      ELF.CreateLogicalSection @program.ActKernelInvocation.cmx.0.0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelInvocation>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "CMX_NN"[0] <2000> : memref<1x1x1x1000xf16, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @text.shave aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareKernelText @DeclareKernelText0 : "activation_hswish"
        // CHECK:   VPUASM.DeclareKernelText
      }
      ELF.CreateSection @program.shave.data aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareKernelData @DeclareKernelArgs0 : "activation_hswish"
        // CHECK:   VPUASM.DeclareKernelData
      }
      ELF.CreateSection @progra.shave.parameter aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.KernelParams @KernelParams0 inputs([@buffer.CMX_NN.0::@DeclareBuffer2]) outputs([@buffer.CMX_NN.0::@DeclareBuffer3]) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>)
        // CHECK:   VPUASM.KernelParams
      }
      ELF.CreateSection @program.barrier aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.ConfigureBarrier @ConfigureBarrier0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier1 idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(1 : 1)
      }
      ELF.CreateSection @task.shave.range.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.ActKernelRange @ActKernelRange0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange0_0_0) kernelTaskType(@COMPUTE) calls @text.shave::@DeclareKernelText0 : @DeclareKernelEntry0
        // CHECK-NOT:   VPUASM.ActKernelRange
        // CHECK:       NPUReg40XX.ActKernelRange
        // CHECK:  type offset 0 size 8 = UINT 0,
        // CHECK:  kernel_entry offset 8 size 64 = UINT 0x1D000000,
      }
      ELF.CreateSection @task.shave.invocation.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.ActKernelInvocation @ActKernelInvocation0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.ActKernelInvocation.cmx.0.0::@DeclareTaskBuffer_ActKernelInvocation_0_0_0) -> @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange0_0_0(kernel_data : @program.shave.data::@DeclareKernelArgs0, kernel_params : @progra.shave.parameter::@KernelParams0) waits([0 : ui8]) updates([1 : ui8]) tile(0) start_after(0) clean_after(0) range_index(0)
        // CHECK-NOT:   VPUASM.ActKernelInvocation
        // CHECK:       NPUReg40XX.ActKernelInvocation
        // CHECK:  range offset 0 size 64 = UINT 0x200000,
        // CHECK:  perf_packet_out offset 24 size 64 = UINT 0,
        // CHECK:  UINT barriers_wait_mask_hi_act at 0 size 32 = 0
        // CHECK:  barriers_wait_mask_lo_act offset 40 size 64 = UINT 1,
        // CHECK:  UINT barriers_post_mask_hi_act at 0 size 32 = 0
        // CHECK:  barriers_post_mask_lo_act offset 56 size 64 = UINT 2,
        // CHECK:  UINT group_act at 0 size 8 = 1,
        // CHECK:  UINT mask_act at 8 size 8 = 1
        // CHECK:  UINT start_after_ at 0 size 32 = 0,
        // CHECK:  UINT clean_after_ at 32 size 32 = 0
        // CHECK:  invo_index offset 80 size 32 = UINT 0,
        // CHECK:  invo_tile offset 84 size 32 = UINT 0,
        // CHECK:  kernel_range_index offset 88 size 32 = UINT 0,
      }
      ELF.CreateSection @task.shave.runtime aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.ActShaveRt @ActKernelRt0 kernel("nnActEntry")
        // CHECK-NOT:   VPUASM.ActShaveRt
        // CHECK:       NPUReg40XX.ActShaveRt
      }
    }
    return
  }
}
