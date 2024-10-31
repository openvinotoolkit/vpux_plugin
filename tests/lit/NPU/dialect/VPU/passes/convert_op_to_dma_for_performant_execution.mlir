
//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-op-to-dma-for-performant-execution %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @GatherMoveToDMA
// CHECK-SAME:  [[ARG0:%.+]]: tensor<30522x21xf16>, [[ARG1:%.+]]: tensor<1x512xsi32>
func.func @GatherMoveToDMA(%arg0: tensor<30522x21xf16>, %arg1: tensor<1x512xsi32>) -> tensor<1x512x21xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64
        } : tensor<30522x21xf16>, tensor<1x512xsi32> -> tensor<1x512x21xf16>

    return %0 :  tensor<1x512x21xf16>

    // CHECK:   [[VAL0:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [512, 1]} : tensor<1x512xsi32> -> tensor<512x1xsi32>
    // CHECK:   [[VAL1:%.+]] = VPU.Convert([[VAL0]]) {dstElemType = i64} : tensor<512x1xsi32> -> tensor<512x1xi64>
    // CHECK:   [[VAL2:%.+]] = VPU.Copy([[VAL1]]) {out_mem_space = [@CMX_NN, 0]} : tensor<512x1xi64> -> tensor<512x1xi64, {mem_space = [@CMX_NN, 0], order = #NC}>
    // CHECK:   [[VAL3:%.+]] = VPU.GatherDMA([[ARG0]], [[VAL2]]) {
    // CHECK-SAME:              axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<30522x21xf16>, tensor<512x1xi64, {mem_space = [@CMX_NN, 0], order = #NC}> -> tensor<512x21xf16, {mem_space = [@CMX_NN, 0], order = #NC}>
    // CHECK:   [[VAL4:%.+]] = VPU.Copy([[VAL3]]) {out_mem_space = @DDR} : tensor<512x21xf16, {mem_space = [@CMX_NN, 0], order = #NC}> -> tensor<512x21xf16>
    // CHECK:   [[VAL5:%.+]] = VPU.Reshape([[VAL4]]) {shape_value = [1, 512, 21]} : tensor<512x21xf16> -> tensor<1x512x21xf16>

    // CHECK:   return [[VAL5]] : tensor<1x512x21xf16>
}

// -----

#NCH = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @GatherMoveToDMATooManyIndices
// CHECK-SAME:  [[ARG0:%.+]]: tensor<30522x8xf16>, [[ARG1:%.+]]: tensor<1x65537xsi32>
func.func @GatherMoveToDMATooManyIndices(%arg0: tensor<30522x8xf16>, %arg1: tensor<1x65537xsi32>) -> tensor<1x65537x8xf16, {mem_space = @DDR, order = #NCH}> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64
        } : tensor<30522x8xf16>, tensor<1x65537xsi32> -> tensor<1x65537x8xf16, {mem_space = @DDR,order = #NCH}>

    return %0 :  tensor<1x65537x8xf16, {mem_space = @DDR,order = #NCH}>

    // CHECK:   [[VAL0:%.+]] = VPU.Gather([[ARG0]], [[ARG1]]) {
    // CHECK-SAME:              axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<30522x8xf16>, tensor<1x65537xsi32> -> tensor<1x65537x8xf16, {mem_space = @DDR, order = #CHW}

    // CHECK:   return [[VAL0]] : tensor<1x65537x8xf16, {mem_space = @DDR, order = #CHW}>
}

// -----

#NCH = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @GatherMoveToDMATooLarge
// CHECK-SAME:  [[ARG0:%.+]]: tensor<30522x4096xf16>, [[ARG1:%.+]]: tensor<1x512xsi32>
func.func @GatherMoveToDMATooLarge(%arg0: tensor<30522x4096xf16>, %arg1: tensor<1x512xsi32>) -> tensor<1x512x4096xf16, {mem_space = @DDR, order = #NCH}> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64
        } : tensor<30522x4096xf16>, tensor<1x512xsi32> -> tensor<1x512x4096xf16, {mem_space = @DDR,order = #NCH}>

    return %0 :  tensor<1x512x4096xf16, {mem_space = @DDR,order = #NCH}>

    // CHECK:   [[VAL0:%.+]] = VPU.Gather([[ARG0]], [[ARG1]]) {
    // CHECK-SAME:              axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<30522x4096xf16>, tensor<1x512xsi32> -> tensor<1x512x4096xf16, {mem_space = @DDR, order = #CHW}

    // CHECK:   return [[VAL0]] : tensor<1x512x4096xf16, {mem_space = @DDR, order = #CHW}>
}

// -----

// CHECK-LABEL: @GatherMoveToDMAWithMultipleIndices
// CHECK-SAME:  [[ARG0:%.+]]: tensor<8404x512xf16>, [[ARG1:%.+]]: tensor<100x10xsi32>
func.func @GatherMoveToDMAWithMultipleIndices(%arg0: tensor<8404x512xf16>, %arg1: tensor<100x10xsi32>) -> tensor<100x10x512xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64
        } : tensor<8404x512xf16>, tensor<100x10xsi32> -> tensor<100x10x512xf16>

    return %0 :  tensor<100x10x512xf16>

    // CHECK:   [[VAL0:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1000, 1]} : tensor<100x10xsi32> -> tensor<1000x1xsi32>
    // CHECK:   [[VAL1:%.+]] = VPU.Convert([[VAL0]]) {dstElemType = i64} : tensor<1000x1xsi32> -> tensor<1000x1xi64>
    // CHECK:   [[VAL2:%.+]] = VPU.Copy([[VAL1]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1000x1xi64> -> tensor<1000x1xi64, {mem_space = [@CMX_NN, 0], order = #NC}>
    // CHECK:   [[VAL3:%.+]] = VPU.GatherDMA([[ARG0]], [[VAL2]]) {
    // CHECK-SAME:              axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<8404x512xf16>, tensor<1000x1xi64, {mem_space = [@CMX_NN, 0], order = #NC}> -> tensor<1000x512xf16, {mem_space = [@CMX_NN, 0], order = #NC}>
    // CHECK:   [[VAL4:%.+]] = VPU.Copy([[VAL3]]) {out_mem_space = @DDR} : tensor<1000x512xf16, {mem_space = [@CMX_NN, 0], order = #NC}> -> tensor<1000x512xf16>
    // CHECK:   [[VAL5:%.+]] = VPU.Reshape([[VAL4]]) {shape_value = [100, 10, 512]} : tensor<1000x512xf16> -> tensor<100x10x512xf16>

    // CHECK:   return [[VAL5]] : tensor<100x10x512xf16>
}

// -----

// CHECK-LABEL: @Gather4DMoveToDMA
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x30522x21xf16>, [[ARG1:%.+]]: tensor<1x512x1x1xsi32>
func.func @Gather4DMoveToDMA(%arg0: tensor<1x1x30522x21xf16>, %arg1: tensor<1x512x1x1xsi32>) -> tensor<1x1x512x21xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x30522x21xf16>, tensor<1x512x1x1xsi32> -> tensor<1x1x512x21xf16>

    return %0 :  tensor<1x1x512x21xf16>

    // CHECK:   [[VAL0:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 512, 1]} : tensor<1x512x1x1xsi32> -> tensor<1x1x512x1xsi32>
    // CHECK:   [[VAL1:%.+]] = VPU.Convert([[VAL0]]) {dstElemType = i64} : tensor<1x1x512x1xsi32> -> tensor<1x1x512x1xi64>
    // CHECK:   [[VAL2:%.+]] = VPU.Copy([[VAL1]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x1x512x1xi64> -> tensor<1x1x512x1xi64, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:   [[VAL3:%.+]] = VPU.GatherDMA([[ARG0]], [[VAL2]]) {
    // CHECK-SAME:              axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x30522x21xf16>, tensor<1x1x512x1xi64, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x512x21xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:   [[VAL4:%.+]] = VPU.Copy([[VAL3]]) {out_mem_space = @DDR} : tensor<1x1x512x21xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x512x21xf16>
    // CHECK:   [[VAL5:%.+]] = VPU.Reshape([[VAL4]]) {shape_value = [1, 1, 512, 21]} : tensor<1x1x512x21xf16> -> tensor<1x1x512x21xf16>

    // CHECK:   return [[VAL5]] : tensor<1x1x512x21xf16>
}

// -----

// CHECK-LABEL: @Gather4DMoveToDMATooManyIndices
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x30522x8xf16>, [[ARG1:%.+]]: tensor<1x65537x1x1xsi32>
func.func @Gather4DMoveToDMATooManyIndices(%arg0: tensor<1x1x30522x8xf16>, %arg1: tensor<1x65537x1x1xsi32>) -> tensor<1x1x65537x8xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x30522x8xf16>, tensor<1x65537x1x1xsi32> -> tensor<1x1x65537x8xf16>

    return %0 :  tensor<1x1x65537x8xf16>

    // CHECK:   [[VAL0:%.+]] = VPU.Gather([[ARG0]], [[ARG1]]) {
    // CHECK-SAME:              axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x30522x8xf16>, tensor<1x65537x1x1xsi32> -> tensor<1x1x65537x8xf16>

    // CHECK:   return [[VAL0]] : tensor<1x1x65537x8xf16>
}

// -----

// CHECK-LABEL: @Gather4DMoveToDMATooLarge
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x30522x4096xf16>, [[ARG1:%.+]]: tensor<1x512x1x1xsi32>
func.func @Gather4DMoveToDMATooLarge(%arg0: tensor<1x1x30522x4096xf16>, %arg1: tensor<1x512x1x1xsi32>) -> tensor<1x1x512x4096xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x30522x4096xf16>, tensor<1x512x1x1xsi32> -> tensor<1x1x512x4096xf16>

    return %0 :  tensor<1x1x512x4096xf16>

    // CHECK:   [[VAL0:%.+]] = VPU.Gather([[ARG0]], [[ARG1]]) {
    // CHECK-SAME:              axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x30522x4096xf16>, tensor<1x512x1x1xsi32> -> tensor<1x1x512x4096xf16>

    // CHECK:   return [[VAL0]] : tensor<1x1x512x4096xf16>
}

// -----

// CHECK-LABEL: @Gather4DMoveToDMAWithMultipleIndices
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x8404x512xf16>, [[ARG1:%.+]]: tensor<1x1000x1x1xsi32>
func.func @Gather4DMoveToDMAWithMultipleIndices(%arg0: tensor<1x1x8404x512xf16>, %arg1: tensor<1x1000x1x1xsi32>) -> tensor<1x1x1000x512xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x8404x512xf16>, tensor<1x1000x1x1xsi32> -> tensor<1x1x1000x512xf16>

    return %0 :  tensor<1x1x1000x512xf16>

    // CHECK:   [[VAL0:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 1000, 1]} : tensor<1x1000x1x1xsi32> -> tensor<1x1x1000x1xsi32>
    // CHECK:   [[VAL1:%.+]] = VPU.Convert([[VAL0]]) {dstElemType = i64} : tensor<1x1x1000x1xsi32> -> tensor<1x1x1000x1xi64>
    // CHECK:   [[VAL2:%.+]] = VPU.Copy([[VAL1]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x1x1000x1xi64> -> tensor<1x1x1000x1xi64, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:   [[VAL3:%.+]] = VPU.GatherDMA([[ARG0]], [[VAL2]]) {
    // CHECK-SAME:              axis_value = 2 : i64, batch_dims = 1 : i64} : tensor<1x1x8404x512xf16>, tensor<1x1x1000x1xi64, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x1000x512xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:   [[VAL4:%.+]] = VPU.Copy([[VAL3]]) {out_mem_space = @DDR} : tensor<1x1x1000x512xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x1000x512xf16>
    // CHECK:   [[VAL5:%.+]] = VPU.Reshape([[VAL4]]) {shape_value = [1, 1, 1000, 512]} : tensor<1x1x1000x512xf16> -> tensor<1x1x1000x512xf16>

    // CHECK:   return [[VAL5]] : tensor<1x1x1000x512xf16>
}
