//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NcePermute
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x3x224x224xf16, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>
func.func @NcePermute(%arg0: tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>)
        -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 7.000000e+00 : f64>
    } -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    }

    return %0 : tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK: [[VIEW_OP_IN:%.*]] = VPUIP.ViewOp [[ARG0]] : memref<1x3x224x224xf16, @CMX_NN>
    // CHECK-SAME:  to memref<1x224x3x224xf16, #NHWC, @CMX_NN>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>

    // CHECK:       [[PERMUTE_RES:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weights([[VIEW_OP_IN]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>

    // CHECK:       PPETask {ppe = #VPU.PPEInt<
    // CHECK-SAME:      mode = <ADD>,
    // CHECK-SAME:      clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:      quant_scale = [3.500000e+00],
    // CHECK-SAME:      fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:  >}

    // CHECK: [[VIEW_OP_OUT:%.*]] = VPUIP.ViewOp [[PERMUTE_RES]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>
    // CHECK-SAME: to memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>

    // CHECK: return [[VIEW_OP_OUT]] : memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>
}
