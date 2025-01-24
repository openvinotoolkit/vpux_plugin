//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --expand-dpu-config %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.CNNNetwork entryPoint : @PPE_FP16_FP16_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScalePreluMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE){{$}}
    // CHECK-NEXT:  }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.CNNNetwork entryPoint : @PPE_FP16_FP16_CONV_LRELU inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_CONV_LRELU() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.250000e-01 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScalePreluMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> prelu_alpha(-0.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.CNNNetwork entryPoint : @PPE_FP16_FP16_CONV_LPRELU inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_CONV_LPRELU() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.250000e-01 : f64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScalePreluMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> prelu_alpha(1.250000e-01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.CNNNetwork entryPoint : @PPE_FP16_BFP16_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xbf16>
  }

  func.func @PPE_FP16_BFP16_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xbf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScalePreluMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(BF16) bf16_round_mode(RNE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.CNNNetwork entryPoint : @PPE_FP16_FP16_MAXPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_MAXPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(NONE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(NONE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.CNNNetwork entryPoint : @PPE_FP16_FP16_ELTWISE inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_ELTWISE() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x1x1xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [2.500000e-01]>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScalePreluMult scale_static(2.500000e-01){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.CNNNetwork entryPoint : @PPE_FP16_FP16_AVEPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_FP16_FP16_AVEPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x1x1xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd bias_static(0.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScalePreluMult scale_static(1.000000e+00){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  IE.CNNNetwork entryPoint : @PPE_U8_FP16_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_U8_FP16_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_high = 4200 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(NONE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_high(4200)
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(FP16)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  IE.CNNNetwork entryPoint : @PPE_FP16_U8_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_FP16_U8_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpScalePreluMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(I32){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647){{$}}
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE){{$}}
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  IE.CNNNetwork entryPoint : @PPE_U8_U8_CONV inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_CONV() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(NONE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift %arg1 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647)
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  IE.CNNNetwork entryPoint : @PPE_U8_U8_ELTWISE_LRELU inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_ELTWISE_LRELU() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <LRELU>, clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 5 : i64, lrelu_shift = 3 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(NONE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(16384)
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(10)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647)
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
module {
  IE.CNNNetwork entryPoint : @PPE_U8_U8_AVEPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_AVEPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<AVEPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, quant_mult = [12288], quant_shift = [9], quant_post_shift = 0>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(NONE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(12288)
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(9)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(RNE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(10)
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647)
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
module {
  IE.CNNNetwork entryPoint : @PPE_U8_U8_MAXPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xui8>
  }

  func.func @PPE_U8_U8_MAXPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, quant_mult = [12288], quant_shift = [9], quant_post_shift = 0>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(NONE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(1)
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(NONE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647)
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(NONE)
    // CHECK-NEXT:  }

    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  IE.CNNNetwork entryPoint : @PPE_U8_FP16_MAXPOOL inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16>
  }

  func.func @PPE_U8_FP16_MAXPOOL() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16x!qElemType, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui8]) updates([1 : ui8]) {clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, variant_count = 1 : ui64} PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, quant_mult = [12288], quant_shift = [9], quant_post_shift = 0>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.PPECfg {
    // CHECK-NEXT:    VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
    // CHECK-NEXT:    VPUIPDPU.PPEFpConvert convert_mode(NONE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntBiasAdd bias_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleMult scale_static(12288)
    // CHECK-NEXT:    VPUIPDPU.PPEIntScaleShift shift_static(9)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
    // CHECK-NEXT:    VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntRound round_mode(NONE)
    // CHECK-NEXT:    VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
    // CHECK-NEXT:    VPUIPDPU.PPEIntClamp clamp_high(2147483647)
    // CHECK-NEXT:    VPUIPDPU.PPEIntConvert convert_mode(FP16)
    // CHECK-NEXT:  }

    }
    return
  }
}
