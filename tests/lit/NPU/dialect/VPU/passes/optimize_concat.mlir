//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-concat %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: EliminateConcat
func.func @EliminateConcat(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %slice_0 = VPU.Slice %concat [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    %slice_1 = VPU.Slice %concat [0, 0, 126, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>

    return %slice_0, %slice_1 : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>

    // CHECK-NOT: Concat
    // CHECK: [[SLICE_0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice %arg1 [0, 0, 1, 0] [1, 16, 64, 128] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // return [[SLICE_0]], [[SLICE_1]] : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotOptimizeConcatWithNotSliceUser
func.func @NotOptimizeConcatWithNotSliceUser(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x48x250x250xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %slice_0 = VPU.Slice %concat [0, 16, 0, 0] [1, 16, 250, 250] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x250x250xf16, {order = #NHWC}>
    %slice_1 = VPU.Slice %concat [0, 0, 126, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    %concat_1 = VPU.Concat(%concat, %slice_0) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : tensor<1x32x250x250xf16, {order = #NHWC}>,
        tensor<1x16x250x250xf16, {order = #NHWC}>
            -> tensor<1x48x250x250xf16, {order = #NHWC}>

    return %concat_1, %slice_1 : tensor<1x48x250x250xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>

    // CHECK: [[CONCAT_0:%.+]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 125, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x125x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE_0:%.+]] = VPU.Slice [[CONCAT_0]] [0, 16, 0, 0] [1, 16, 250, 250] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[CONCAT_0]] [0, 0, 126, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: [[CONCAT_1:%.+]] = VPU.Concat([[CONCAT_0]], [[SLICE_0]])
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 32, 0, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x250x250xf16, {order = #NHWC}>, tensor<1x16x250x250xf16, {order = #NHWC}> -> tensor<1x48x250x250xf16, {order = #NHWC}>
    // CHECK: return [[CONCAT_1]], [[SLICE_1]] : tensor<1x48x250x250xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotEliminateConcatWithNotSubTensorDueToShape
func.func @NotEliminateConcatWithNotSubTensorDueToShape(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x32x126x250xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %slice = VPU.Slice %concat [0, 0, 0, 0] [1, 32, 126, 250] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x32x126x250xf16, {order = #NHWC}>
    return %slice : tensor<1x32x126x250xf16, {order = #NHWC}>

    // CHECK: [[CONCAT:%.+]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 125, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x125x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 0, 0] [1, 32, 126, 250] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x32x126x250xf16, {order = #NHWC}>
    // CHECK: return [[SLICE]] : tensor<1x32x126x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotEliminateConcatWithNotSubTensorDueToOffset
func.func @NotEliminateConcatWithNotSubTensorDueToOffset(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x16x64x128xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %slice = VPU.Slice %concat [0, 0, 100, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    return %slice : tensor<1x16x64x128xf16, {order = #NHWC}>

    // CHECK: [[CONCAT:%.+]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 125, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x125x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 100, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: return [[SLICE]] : tensor<1x16x64x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotEliminateConcatWithNotAllSubTensors
func.func @NotEliminateConcatWithNotAllSubTensors(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    // SubTensor
    %slice_0 = VPU.Slice %concat [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // Not SubTensor
    %slice_1 = VPU.Slice %concat [0, 0, 100, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    return %slice_0, %slice_1 : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>

    // CHECK: [[CONCAT:%.+]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 125, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x125x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE_0:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 100, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: return [[SLICE_0]], [[SLICE_1]] : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotEliminateConcatWithInputMultiUsers
func.func @NotEliminateConcatWithInputMultiUsers(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x32x164x250xf16, {order = #NHWC}>) {

    %input_slice = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 100, 250] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x32x100x250xf16, {order = #NHWC}>
    %concat = VPU.Concat(%input_slice, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 100, 0]
        ]
    } : tensor<1x32x100x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x225x250xf16, {order = #NHWC}>

    %output_slice = VPU.Slice %concat [0, 0, 0, 0] [1, 32, 64, 250] : tensor<1x32x225x250xf16, {order = #NHWC}> to tensor<1x32x64x250xf16, {order = #NHWC}>
    %output_concat = VPU.Concat(%input_slice, %output_slice) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 100, 0]
        ]
    } : tensor<1x32x100x250xf16, {order = #NHWC}>,
        tensor<1x32x64x250xf16, {order = #NHWC}>
            -> tensor<1x32x164x250xf16, {order = #NHWC}>

    return %output_concat : tensor<1x32x164x250xf16, {order = #NHWC}>

    // CHECK: [[INPUT_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 100, 250] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x32x100x250xf16, {order = #NHWC}>
    // CHECK: [[CONCAT:%.*]] = VPU.Concat([[INPUT_SLICE]], %arg1) {
    // CHECK:    static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK:    tensor<1x32x100x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x225x250xf16, {order = #NHWC}>

    // CHECK: [[OUTPUT_SLICE:%.*]] = VPU.Slice [[CONCAT]] [0, 0, 0, 0] [1, 32, 64, 250] : tensor<1x32x225x250xf16, {order = #NHWC}> to tensor<1x32x64x250xf16, {order = #NHWC}>
    // CHECK: [[OUTPUT_CONCAT:%.*]] = VPU.Concat([[INPUT_SLICE]], [[OUTPUT_SLICE]]) {
    // CHECK:   static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK:    tensor<1x32x100x250xf16, {order = #NHWC}>, tensor<1x32x64x250xf16, {order = #NHWC}> -> tensor<1x32x164x250xf16, {order = #NHWC}>
    // CHECK: return [[OUTPUT_CONCAT]] : tensor<1x32x164x250xf16, {order = #NHWC}>
}
