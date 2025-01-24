//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% %s | FileCheck %s
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
      }
      ELF.CreateSection @program.shave.data aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareKernelData @DeclareKernelArgs0 : "activation_hswish"
      }
      ELF.CreateSection @progra.shave.parameter aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.KernelParams @KernelParams0 inputs([@buffer.CMX_NN.0::@DeclareBuffer2]) outputs([@buffer.CMX_NN.0::@DeclareBuffer3]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>)
      }
      ELF.CreateSection @program.barrier aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.ConfigureBarrier @ConfigureBarrier0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier1 idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(1 : 1)
      }
      ELF.CreateSection @task.shave.range.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        "NPUReg40XX.ActKernelRange"() <{act_range_descriptor = #NPUReg40XX.VpuActKernelRange<
          VpuActKernelRange {
            type = UINT 0,
            kernel_entry = UINT 0x1D000000 requires 1:1:1,
            text_window_base = UINT 0,
            code_size = UINT 0x750,
            reserved_akr = UINT 0,
            kernel_invo_count = UINT 0,
            pad1_4 = UINT 0
          }
        >, kernel_entry = @DeclareKernelEntry0, kernel_text = @text.shave::@DeclareKernelText0, sym_name = "ActKernelRange0", task_location = @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange0_0_0}> : () -> ()
        // CHECK: act_range_descriptor = #NPUReg40XX.VpuActKernelRange<
        // CHECK: VpuActKernelRange {
        // CHECK:   type = UINT 0,
        // CHECK:   kernel_entry = UINT 0x1D000000 requires 1:1:1,
        // CHECK:   text_window_base = UINT 0,
        // CHECK:   code_size = UINT 0x750,
        // CHECK:   reserved_akr = UINT 0,
        // CHECK:   kernel_invo_count = UINT 0,
        // CHECK:   pad1_4 = UINT 0
        // CHECK: }
        // CHECK: >    
      }
      ELF.CreateSection @task.shave.invocation.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        "NPUReg40XX.ActKernelInvocation"() <{act_invo_descriptor = #NPUReg40XX.VpuActKernelInvocation<
          VpuActKernelInvocation {
            range = UINT 0x200000,
            kernel_args = UINT 0 requires 1:1:1,
            data_window_base = UINT 0,
            perf_packet_out = UINT 0,
            barriers_wait_mask_hi_act {
              UINT barriers_wait_mask_hi_act = 0
            },
            barriers_wait_mask_lo_act = UINT 1,
            barriers_post_mask_hi_act {
              UINT barriers_post_mask_hi_act = 0
            },
            barriers_post_mask_lo_act = UINT 2,
            barriers_group_mask_act {
              UINT group_act = 1,
              UINT mask_act = 1
            },
            act_invo_barriers_sched {
              UINT start_after_ = 0,
              UINT clean_after_ = 0
            },
            reserved_aki = UINT 0,
            invo_tile = UINT 0,
            kernel_range_index = UINT 0,
            next_aki_wl_addr = UINT 0
          }
        >, kernel_data = @program.shave.data::@DeclareKernelArgs0, kernel_params = @progra.shave.parameter::@KernelParams0, kernel_range = @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange0_0_0, sym_name = "ActKernelInvocation0", task_location = @program.ActKernelInvocation.cmx.0.0::@DeclareTaskBuffer_ActKernelInvocation_0_0_0}> : () -> ()
        // CHECK: act_invo_descriptor = #NPUReg40XX.VpuActKernelInvocation<
        // CHECK: VpuActKernelInvocation {
        // CHECK:   range = UINT 0x200000,
        // CHECK:   kernel_args = UINT 0 requires 1:1:1,
        // CHECK:   data_window_base = UINT 0,
        // CHECK:   perf_packet_out = UINT 0,
        // CHECK:   barriers_wait_mask_hi_act {
        // CHECK:     UINT barriers_wait_mask_hi_act = 0
        // CHECK:   },
        // CHECK:   barriers_wait_mask_lo_act = UINT 1,
        // CHECK:   barriers_post_mask_hi_act {
        // CHECK:     UINT barriers_post_mask_hi_act = 0
        // CHECK:   },
        // CHECK:   barriers_post_mask_lo_act = UINT 2,
        // CHECK:   barriers_group_mask_act {
        // CHECK:     UINT group_act = 1,
        // CHECK:     UINT mask_act = 1
        // CHECK:   },
        // CHECK:   act_invo_barriers_sched {
        // CHECK:     UINT start_after_ = 0,
        // CHECK:     UINT clean_after_ = 0
        // CHECK:   },
        // CHECK:   reserved_aki = UINT 0,
        // CHECK:   invo_tile = UINT 0,
        // CHECK:   kernel_range_index = UINT 0,
        // CHECK:   next_aki_wl_addr = UINT 0
        // CHECK: }
        // CHECK: >
      }
    }
    return
  }
}
