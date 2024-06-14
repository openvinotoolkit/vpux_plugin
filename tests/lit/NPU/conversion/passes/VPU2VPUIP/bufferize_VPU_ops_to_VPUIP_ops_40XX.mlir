//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

func.func @ConvertFP32ToFP16UsingConvertDMA(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf16>
    return %0 : tensor<1x3x4x4xf16>

    // CHECK:       [[ALLOC:%.*]] = memref.alloc() : memref<1x3x4x4xf16>
    // CHECK:       [[OUT:%.*]] = VPUIP.ConvertDMA inputs({{[^:]+}} : memref<1x3x4x4xf32>) outputs([[ALLOC]] : memref<1x3x4x4xf16>) -> memref<1x3x4x4xf16>
}

// CHECK-LABEL: @GatherDMA
// CHECK-SAME: ([[ARG0:%.*]]: memref<30522x22xf16>, [[ARG1:%.*]]:  memref<512x1xi64>)
func.func @GatherDMA(%arg0: tensor<30522x22xf16>, %arg1:  tensor<512x1xi64>) -> tensor<1x512x22xf16, {mem_space = @DDR, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> {
    %2 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<512x1xi64> -> tensor<512x1xi64, {mem_space = @CMX_NN, order = affine_map<(d0, d1) -> (d0, d1)>}>
    %3 = VPU.GatherDMA(%arg0, %2) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<30522x22xf16>, tensor<512x1xi64, {mem_space = @CMX_NN, order = affine_map<(d0, d1) -> (d0, d1)>}> -> tensor<512x22xf16, {mem_space = [@CMX_NN, 0], order = affine_map<(d0, d1) -> (d0, d1)>}>
    %4 = VPU.Reshape(%3) {shape_value = [1, 512, 22]} : tensor<512x22xf16, {mem_space = [@CMX_NN, 0], order = affine_map<(d0, d1) -> (d0, d1)>}> -> tensor<1x512x22xf16, {mem_space = [@CMX_NN, 0], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    %5 = VPU.Copy(%4) {out_mem_space = @DDR} : tensor<1x512x22xf16, {mem_space = [@CMX_NN, 0], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> -> tensor<1x512x22xf16, {mem_space = @DDR, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    return %5 : tensor<1x512x22xf16, {mem_space = @DDR, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    // CHECK:       [[VARALLOC:%.*]] = memref.alloc() : memref<512x1xi64, @CMX_NN>
    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy inputs([[ARG1]] : memref<512x1xi64>) outputs([[VARALLOC]] : memref<512x1xi64, @CMX_NN>) -> memref<512x1xi64, @CMX_NN>

    // CHECK:       [[VARALLOC0:%.*]] = memref.alloc() : memref<512x22xf16, [@CMX_NN, 0]>
    // CHECK:       [[VAR3:%.*]] = VPUIP.GatherDMA {channelType = 0 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64} inputs([[ARG0]] : memref<30522x22xf16>) indices([[VAR2]] : memref<512x1xi64, @CMX_NN>) outputs([[VARALLOC0]] : memref<512x22xf16, [@CMX_NN, 0]>) -> memref<512x22xf16, [@CMX_NN, 0]>
    // CHECK:       [[VAR4:%.*]] = VPUIP.GenericReshape inputs([[VAR3]] : memref<512x22xf16, [@CMX_NN, 0]>) -> memref<1x512x22xf16, [@CMX_NN, 0]>
    // CHECK:       [[VARALLOC1:%.*]] = memref.alloc() : memref<1x512x22xf16, @DDR>
    // CHECK:       [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR4]] : memref<1x512x22xf16, [@CMX_NN, 0]>) outputs([[VARALLOC1]] : memref<1x512x22xf16, @DDR>) -> memref<1x512x22xf16, @DDR>

    // CHECK: return [[VAR5]] : memref<1x512x22xf16, @DDR>
}
