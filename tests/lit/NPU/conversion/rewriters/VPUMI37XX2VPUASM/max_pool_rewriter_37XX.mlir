//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUMI37XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
IE.CNNNetwork entryPoint : @maxpool_f16_f16 inputsInfo : {
  DataInfo "input_0" : tensor<1x64x16x16xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x64x8x8xf16>
}

func.func @maxpool_f16_f16() {
  %0 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
  %1 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>

  %8 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [0] <40976> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>

  %20 = VPUMI37XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} taskLocation(%1 : !VPURegMapped.Index<0:0:0>) input(%8 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%10 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%11 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
    VPUMI37XX.PPETask {ppe = #VPU.PPEStub<>}
  }

  %21 = "VPUMI37XX.DPUVariant"(%0, %20) {end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
  %22 = VPUMI37XX.MappedInference invariants(%20 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(1) variantCount(1) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: func.func @maxpool_f16_f16

//CHECK: ELF.CreateLogicalSection @[[SECMETA:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareTaskBuffer @[[TBVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>

//CHECK: ELF.CreateLogicalSection @[[SECCMX0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF2:.*]] !VPUASM.Buffer< "CMX_NN"[0] <8192>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF3:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF4:.*]] !VPUASM.Buffer< "CMX_NN"[0] <8192>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF5:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF7:.*]] !VPUASM.Buffer< "CMX_NN"[0] <40976>

//CHECK: ELF.CreateSection @[[SECINV:.*]] aligned
//CHECK-NEXT: VPUASM.DPUInvariant_37XX @[[SYMINV0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBIVAR0]])
    // CHECK-SAME: input(@[[SECCMX0]]::@[[SYMBUFF2]]) weight_table(@[[SECCMX0]]::@[[SYMBUFF7]]) parent_input(@[[SECCMX0]]::@[[SYMBUFF4]])
    // CHECK-SAME: parent_output(@[[SECCMX0]]::@[[SYMBUFF5]]) outputs([@[[SECCMX0]]::@[[SYMBUFF3]]]) waits([]) updates([])

//CHECK: ELF.CreateSection @[[SECVAR:.*]] aligned
//CHECK-NEXT: VPUASM.DPUVariant_37XX @[[SYMVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBVAR0]])
    // CHECK-SAME: calls @[[SECINV]]::@[[SYMINV0]]

//CHECK: VPUASM.MappedInference_37XX @MappedInference : invariants(@[[SECINV]]::@[[SYMINV0]]) variants(@[[SECVAR]]::@[[SYMVAR0]])
//CHECK-SAME: dmaCount([0, 0]) invariantCount(1) variantCount(1) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0)
