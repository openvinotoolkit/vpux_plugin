//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: func.func @ConvertFP32ToFP16UsingConvertDMA
// CHECK-SAME: ([[ARG:%.+]]: memref<1x3x4x4xf32>)
func.func @ConvertFP32ToFP16UsingConvertDMA(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf16>
    return %0 : tensor<1x3x4x4xf16>

    // CHECK:       [[ALLOC:%.*]] = memref.alloc() : memref<1x3x4x4xf16>
    // CHECK:       [[OUT:%.*]] = VPUIP.ConvertDMA inputs([[ARG]] : memref<1x3x4x4xf32>) outputs([[ALLOC]] : memref<1x3x4x4xf16>) -> memref<1x3x4x4xf16>
}

// -----
// CHECK-LABEL: func.func @ConvertFP64ToFP16UsingSWKernel
// CHECK-SAME: ([[ARG:%.+]]: memref<1x1x1x1xf64>)
func.func @ConvertFP64ToFP16UsingSWKernel(%arg0: tensor<1x1x1x1xf64>) -> tensor<1x1x1x1xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x1x1x1xf64> -> tensor<1x1x1x1xf16>
    return %0 : tensor<1x1x1x1xf16>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1xf64, [@CMX_NN, 0]>
    // CHECK:   [[ARG_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x1x1x1xf64>) outputs([[ALLOC]] : memref<1x1x1x1xf64, [@CMX_NN, 0]>) -> memref<1x1x1x1xf64, [@CMX_NN, 0]>
    // CHECK:   [[ALLOC_0:%.+]] = memref.alloc() : memref<1x1x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[ARG_CMX]] as [[ARG_1:%.+]]: memref<1x1x1x1xf64, [@CMX_NN, 0]>) outputs([[ALLOC_0]] as [[ARG_2:%.+]]: memref<1x1x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1xf16, [@CMX_NN, 0]>{
    // CHECK:       VPUIP.SW.Kernel.run([[ARG_1]], [[ARG_2]]) : memref<1x1x1x1xf64, [@CMX_NN, 0]>, memref<1x1x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[ALLOC_1:%.+]] = memref.alloc() : memref<1x1x1x1xf16>
    // CHECK:   [[OUT:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC_1]] : memref<1x1x1x1xf16>) -> memref<1x1x1x1xf16>
    // CHECK:   [[OUT]] : memref<1x1x1x1xf16>
}

// -----

// CHECK-LABEL: @GatherDMA
// CHECK-SAME: [[INPUT_0:%arg[0-9]]]: memref<1x1x8404x512xf16>
// CHECK-SAME: [[INPUT_1:%arg[0-9]]]: memref<1x1000x1x1xsi32>
func.func @GatherDMA(%arg0: tensor<1x1x8404x512xf16>, %arg1:  tensor<1x1000x1x1xsi32>) -> tensor<1x1x1000x512xf16> {
    %0 = VPU.Reshape(%arg1) {shape_value = [1, 1, 1000, 1]} : tensor<1x1000x1x1xsi32> -> tensor<1x1x1000x1xsi32>
    %1 = VPU.Convert(%0) {dstElemType = i64} : tensor<1x1x1000x1xsi32> -> tensor<1x1x1000x1xi64>
    %2 = VPU.GatherDMA(%arg0, %1) {axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x8404x512xf16>, tensor<1x1x1000x1xi64> -> tensor<1x1x1000x512xf16>
    %3 = VPU.Reshape(%2) {shape_value = [1, 1, 1000, 512]} : tensor<1x1x1000x512xf16> -> tensor<1x1x1000x512xf16>

    return %3 : tensor<1x1x1000x512xf16>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT_1]] : memref<1x1000x1x1xsi32>) -> memref<1x1x1000x1xsi32>
    // CHECK:       [[CONVERT_IN_ALLOC:%.+]] = memref.alloc() : memref<1x1x1000x1xsi32, [@CMX_NN, 0]>
    // CHECK:       [[CONVERT_IN_COPY:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : memref<1x1x1000x1xsi32>)
    // CHECK-SAME:                                       outputs([[CONVERT_IN_ALLOC]] : memref<1x1x1000x1xsi32, [@CMX_NN, 0]>)
    // CHECK:       [[CONVERT_OUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1000x1xi64, [@CMX_NN, 0]>
    // CHECK:       [[CONVERT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert
    // CHECK-SAME:                      inputs([[CONVERT_IN_COPY]] as [[INNER_IN:%.+]]: memref<1x1x1000x1xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:                      outputs([[CONVERT_OUT_ALLOC]] as [[INNER_OUT:%.+]]: memref<1x1x1000x1xi64, [@CMX_NN, 0]>)
    // CHECK:         VPUIP.SW.Kernel.run([[INNER_IN]], [[INNER_OUT]]) : memref<1x1x1000x1xsi32, [@CMX_NN, 0]>, memref<1x1x1000x1xi64, [@CMX_NN, 0]>
    // CHECK:       [[CONVERT_DDR_ALLOC:%.+]] = memref.alloc() : memref<1x1x1000x1xi64>
    // CHECK:       [[CONVERT_OUT_COPY:%.+]] = VPUIP.Copy inputs([[CONVERT]] : memref<1x1x1000x1xi64, [@CMX_NN, 0]>)
    // CHECK-SAME:                                   outputs([[CONVERT_DDR_ALLOC]] : memref<1x1x1000x1xi64>)

    // CHECK:       [[INDICES_ALLOC:%.+]] = memref.alloc() : memref<1x1x1000x1xi64, [@CMX_NN, 0]>
    // CHECK:       [[INDICES_IN:%.+]] = VPUIP.Copy inputs([[CONVERT_OUT_COPY]] : memref<1x1x1000x1xi64>)
    // CHECK-SAME:                                  outputs([[INDICES_ALLOC]] : memref<1x1x1000x1xi64, [@CMX_NN, 0]>)
    // CHECK:       [[GATHE_OUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1000x512xf16, [@CMX_NN, 0]>
    // CHECK:       [[GATHER_DMA:%.+]] = VPUIP.GatherDMA {channelType = 0 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}
    // CHECK-SAME:                          inputs([[INPUT_0]] : memref<1x1x8404x512xf16>
    // CHECK-SAME:                          indices([[INDICES_IN]] : memref<1x1x1000x1xi64, [@CMX_NN, 0]>
    // CHECK-SAME:                          outputs([[GATHE_OUT_ALLOC]] : memref<1x1x1000x512xf16, [@CMX_NN, 0]>
    // CHECK:       [[GATHER_DDR_ALLOC:%.+]] = memref.alloc() : memref<1x1x1000x512xf16>
    // CHECK:       [[GATHER_OUT_COPY:%.+]] = VPUIP.Copy inputs([[GATHER_DMA]] : memref<1x1x1000x512xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:                                outputs([[GATHER_DDR_ALLOC]] : memref<1x1x1000x512xf16>) -> memref<1x1x1000x512xf16>

    // CHECK:       return [[GATHER_OUT_COPY]] : memref<1x1x1000x512xf16>
}
