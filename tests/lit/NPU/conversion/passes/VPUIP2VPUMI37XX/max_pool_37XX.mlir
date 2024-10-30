//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// max_pool_1x64x16x16xfp16_2x2_pads_1x0x1x0_strides_2x2_fp16

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  IE.CNNNetwork entryPoint : @maxpool_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x9x8xf16>
  }

  func.func private @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x64x9x8xf16, #NHWC, @DDR>) -> memref<1x64x9x8xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <41984> -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <42000> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    VPURT.Task {
      %8 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%0 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%3 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {outEnd = [7, 8, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
      } PPE : { PPETask {opaque_ppe = #VPU.PPEStub<>} }
    }
    return %arg1 : memref<1x64x9x8xf16, #NHWC, @DDR>
  }

}

// CHECK: func.func private @maxpool_f16_f16
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NOT: VPURT.Task
// CHECK: VPUMI37XX.DPUInvariant
// CHECK: VPUMI37XX.DPUVariant
