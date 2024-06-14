//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

// CHECK-LABEL:  func.func @ConvertFP32ToFP16
func.func @ConvertFP32ToFP16(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf16>
    return %0 : tensor<1x3x4x4xf16>

    // CHECK-NOT:   VPUIP.SW.Kernel
    // CHECK:       return
    // CHECK:       <1x3x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ConvertFP16ToFP32UsingSW
func.func @ConvertFP16ToFP32UsingSW(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
    return %0 : tensor<1x3x4x4xf32>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x3x4x4xf16>) outputs([[VAR0]] : memref<1x3x4x4xf16, [@CMX_NN, 0]>) -> memref<1x3x4x4xf16, [@CMX_NN, 0]>

    // CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x3x4x4xf32, [@CMX_NN, 0]>
    // CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[VAR1]] as %arg1: memref<1x3x4x4xf16, [@CMX_NN, 0]>) outputs([[VAR2]] as %arg2: memref<1x3x4x4xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x4x4xf32, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run(%arg1, %arg2) : memref<1x3x4x4xf16, [@CMX_NN, 0]>, memref<1x3x4x4xf32, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x3x4x4xf32>
    // CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x3x4x4xf32, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x3x4x4xf32>) -> memref<1x3x4x4xf32>
}
