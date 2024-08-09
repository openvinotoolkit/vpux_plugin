//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --bufferize-IE %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

func.func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x1000xf16>
    // CHECK: [[VAR2:%.*]] = IERT.SoftMax
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x1000xf16>)

    // CHECK: return
    // CHECK-SAME: tensor<1x1000xf16>
}

// -----

func.func @ConstantLayer() -> tensor<1x2x2x2xf16> {
    %0 = const.Declare tensor<1x2x2x2xf16> =
      dense<1.0> : tensor<1x2x2x2xf32>, [#const.ConvertElemType<f16>]
    return %0 : tensor<1x2x2x2xf16>

    // CHECK-DAG:       [[VAR0:%.*]] = const.Declare tensor<1x2x2x2xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x2x2xf32>, [#const.ConvertElemType<f16>]

    // CHECK:       return
    // CHECK-SAME: tensor<1x2x2x2xf16>
}

// -----

func.func @Reshape(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 512] } : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>

    // CHECK: [[VAR1:%.*]] = IERT.GenericReshape inputs({{[^:]+}} : memref<1x512x1x1xf32>) -> memref<1x512xf32>

    // CHECK: return
    // CHECK-SAME: tensor<1x512xf32>
}

// -----

func.func @Split(%tensor: tensor<2x6x4x2xf32>) -> (tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>) {
    %0:2 = IE.Split(%tensor) {num_splits = 2, axis_value = 0} : tensor<2x6x4x2xf32> -> tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>
    return %0#0, %0#1 : tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x6x4x2xf32>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x6x4x2xf32>

    // CHECK:       [[VAR2:%.*]] = IERT.SubView {{[^:]+}} [0, 0, 0, 0] [1, 6, 4, 2]
    // CHECK-SAME:      : memref<2x6x4x2xf32> to memref<1x6x4x2xf32>
    // CHECK:       [[VAR4:%.*]] = IERT.Copy inputs([[VAR2]] : memref<1x6x4x2xf32>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x6x4x2xf32>)
    // CHECK:       [[VAR3:%.*]] = IERT.SubView {{[^:]+}} [1, 0, 0, 0] [1, 6, 4, 2]
    // CHECK-SAME:      : memref<2x6x4x2xf32> to memref<1x6x4x2xf32>
    // CHECK:       [[VAR5:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x6x4x2xf32>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x6x4x2xf32>)

    // CHECK: return
    // CHECK-SAME:  tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>
}

// -----

func.func @Concat(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1>} :
        tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    return %0 : tensor<1x4x3x4xf32>


    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<1x4x3x4xf32>

    // CHECK:       [[VAR3:%.*]] = IERT.SubView [[VAR2]] [0, 0, 0, 0] [1, 2, 3, 4]
    // CHECK-SAME:      : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>
    // CHECK:       [[VAR4:%.*]] = IERT.Copy inputs({{[^:]+}} : memref<1x2x3x4xf32>)
    // CHECK-SAME:      outputs([[VAR3]] : memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>)

    // CHECK:       [[VAR5:%.*]] = IERT.SubView [[VAR2]] [0, 2, 0, 0] [1, 2, 3, 4]
    // CHECK-SAME:      : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>
    // CHECK:       [[VAR6:%.*]] = IERT.Copy inputs({{[^:]+}} : memref<1x2x3x4xf32>)
    // CHECK-SAME:      outputs([[VAR5]] : memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>)

    // CHECK:       [[VAR7:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[VAR4]], [[VAR6]] :
    // CHECK-SAME:          memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>,
    // CHECK-SAME:          memref<1x2x3x4xf32, {order = #NCHW,  strides = [48, 12, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x4x3x4xf32>)

    // CHECK: return
    // CHECK-SAME: tensor<1x4x3x4xf32>
}

// -----

func.func @ConcatWithStride(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1, offset = 1, stride = 2>} :
        tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    return %0 : tensor<1x4x3x4xf32>

    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<1x4x3x4xf32>

    // CHECK:       [[VAR3:%.*]] = IERT.SubView [[VAR2]] [0, 0, 0, 0] [1, 2, 3, 4] [1, 2, 1, 1]
    // CHECK-SAME:      : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>
    // CHECK:       [[VAR4:%.*]] = IERT.Copy inputs({{[^:]+}} : memref<1x2x3x4xf32>)
    // CHECK-SAME:      outputs([[VAR3]] : memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>)

    // CHECK:       [[VAR5:%.*]] = IERT.SubView [[VAR2]] [0, 1, 0, 0] [1, 2, 3, 4] [1, 2, 1, 1]
    // CHECK-SAME:      : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>
    // CHECK:       [[VAR6:%.*]] = IERT.Copy inputs({{[^:]+}} : memref<1x2x3x4xf32>)
    // CHECK-SAME:      outputs([[VAR5]] : memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>)

    // CHECK:       [[VAR7:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[VAR4]], [[VAR6]] :
    // CHECK-SAME:          memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>,
    // CHECK-SAME:          memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x4x3x4xf32>) -> memref<1x4x3x4xf32>

    // CHECK: return
    // CHECK-SAME: tensor<1x4x3x4xf32>
}

// -----

func.func @ExpandToSubview(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x3x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[VIEW1:%.*]] = IERT.SubView [[VAR1]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.*]] = IERT.Copy inputs({{[^:]+}} : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.*]] = IERT.SubView [[VAR1]] [0, 3, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.*]] = IERT.Copy inputs({{[^:]+}} : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW_IN:%.*]] = IERT.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 4, 4]
    // CHECK-SAME:      : memref<1x3x4x4xf16> to memref<1x2x4x4xf16, {order = #NCHW, strides = [48, 16, 4, 1]}>
    // CHECK:       [[VIEW_TAIL:%.*]] = IERT.SubView [[VAR1]] [0, 6, 0, 0] [1, 2, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x2x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY3:%.*]] = IERT.Copy inputs([[VIEW_IN]] : memref<1x2x4x4xf16, {order = #NCHW, strides = [48, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VIEW_TAIL]] : memref<1x2x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]], [[COPY3]] :
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x2x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK: return
    // CHECK-SAME: tensor<1x8x4x4xf16>
}

// -----

func.func @ExpandToSubviewWithoutTail(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x4x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>


    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[VIEW1:%.*]] = IERT.SubView [[VAR1]] [0, 0, 0, 0] [1, 4, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.*]] = IERT.Copy inputs({{[^:]+}} : memref<1x4x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW1:%.*]] = IERT.SubView [[VAR1]] [0, 4, 0, 0] [1, 4, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.*]] = IERT.Copy inputs({{[^:]+}} : memref<1x4x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK: return
    // CHECK-SAME: tensor<1x8x4x4xf16>
}

// -----

func.func @ExpandToSubviewOnlyWithTail(%arg0: tensor<1x5x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x5x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>


    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[VIEW1:%.*]] = IERT.SubView [[VAR1]] [0, 0, 0, 0] [1, 5, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.*]] = IERT.Copy inputs({{[^:]+}} : memref<1x5x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW_IN:%.*]] = IERT.SubView {{[^:]+}} [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x5x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>
    // CHECK:       [[VIEW_TAIL:%.*]] = IERT.SubView [[VAR1]] [0, 5, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.*]] = IERT.Copy inputs([[VIEW_IN]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VIEW_TAIL]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK: return
    // CHECK-SAME： tensor<1x8x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @WithMemSpace(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16> {
    %0 = IE.ReLU(%arg0) : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16, {order = #NHWC, mem_space = @CMX_NN}>
    %1 = IE.Tanh(%0) : tensor<1x2x3x4xf16, {order = #NHWC, mem_space = @CMX_NN}> -> tensor<1x2x3x4xf16>
    return %1 : tensor<1x2x3x4xf16>


    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR2:%.*]] = IERT.ReLU
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x2x3x4xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x2x3x4xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[VAR3:%.*]] = memref.alloc() : memref<1x2x3x4xf16>
    // CHECK:       [[VAR4:%.*]] = IERT.Tanh
    // CHECK-SAME:      inputs([[VAR2]] : memref<1x2x3x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAR3]] : memref<1x2x3x4xf16>)

    // CHECK: return
    // CHECK-SAME: tensor<1x2x3x4xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @PermuteCast(%arg0: tensor<1x12x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x12xf16> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x12x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x12xf16>
    return %0 : tensor<1x16x16x12xf16>


    //CHECK:        [[VAR1:%.*]] = IERT.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x12x16x16xf16, #NHWC>) -> memref<1x16x16x12xf16>

    //CHECK: return
    // CHECK-SAME: tensor<1x16x16x12xf16>
}

// -----

// CHECK-LABEL: @Roll
func.func @Roll(%arg0: tensor<3x10x100x200xf16>) -> tensor<3x10x100x200xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %cst_0 = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.Roll(%arg0, %cst, %cst_0) : tensor<3x10x100x200xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<3x10x100x200xf16>
    return %0 : tensor<3x10x100x200xf16>

    // CHECK-DAG:   [[VAR1:%.*]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK-DAG:   [[VAR2:%.*]] = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       [[VAR3:%.*]] = memref.alloc() : memref<3x10x100x200xf16>

    // CHECK:       [[VAR4:%.*]] = IERT.Roll
    // CHECK-SAME:     inputs({{[^:]+}} : memref<3x10x100x200xf16>, {{[^:]+}} : memref<1xsi32>, {{[^:]+}} : memref<2xsi32>)
    // CHECK-SAME:     outputs([[VAR3]] : memref<3x10x100x200xf16>) -> memref<3x10x100x200xf16>

    // CHECK:       return
    // CHECK-SAME: tensor<3x10x100x200xf16>
}

// -----

// CHECK-LABEL: @CTCGreedyDecoder
func.func @CTCGreedyDecoder(%arg0: tensor<20x8x128xf16>, %arg1: tensor<20x8xf16>) -> tensor<8x20x1x1xf16> {
    %0 = IE.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    return %0 : tensor<8x20x1x1xf16>

    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<8x20x1x1xf16>
    // CHECK:       [[VAR3:%.*]] = IERT.CTCGreedyDecoder {mergeRepeated}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<20x8x128xf16>, {{[^:]+}} : memref<20x8xf16>)
    // CHECK-SAME：     outputs([[VAR2]] : memref<8x20x1x1xf16>) -> memref<8x20x1x1xf16>

    //CHECK: return
    // CHECK-SAME: tensor<8x20x1x1xf16>
}

// -----

// CHECK-LABEL: @ExtractImagePatches
func.func @ExtractImagePatches(%arg0: tensor<64x3x10x10xf32>) -> tensor<64x27x2x2xf32> {
    %0 = IE.ExtractImagePatches(%arg0) {sizes = [3, 3], strides = [5, 5], rates = [1, 1], autoPad = #IE.pad_type<VALID>} : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    return %0 : tensor<64x27x2x2xf32>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<64x27x2x2xf32>
    // CHECK:       [[VAR2:%.*]] = IERT.ExtractImagePatches {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [3, 3], strides = [5, 5]}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<64x3x10x10xf32>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<64x27x2x2xf32>) -> memref<64x27x2x2xf32>

    // CHECK: return
    // CHECK-SAME: tensor<64x27x2x2xf32>
}

// -----
// CHECK-LABEL: @ReduceL2
func.func @ReduceL2(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.ReduceL2(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR2:%.*]] = IERT.ReduceL2 {keep_dims}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>, {{[^:]+}} : memref<1xsi32>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>

    // CHECK: return
    // CHECK-SAME: tensor<1x32x112x1xf16>
}

// -----
// CHECK-LABEL: @EmbeddingBagOffsetsSum
func.func @EmbeddingBagOffsetsSum(%arg0: tensor<5x6x4xsi32>) -> tensor<2x6x4xsi32> {
    %0 = IE.EmbeddingBagOffsetsSum(%arg0) {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>,
    per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} : tensor<5x6x4xsi32> -> tensor<2x6x4xsi32>
    return %0 : tensor<2x6x4xsi32>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<2x6x4xsi32>
    // CHECK:       [[VAR2:%.*]] = IERT.EmbeddingBagOffsetsSum {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 2],
    // CHECK-SAME:  weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]}
    // CHECK-SAME:  inputs({{[^:]+}} : memref<5x6x4xsi32>)
    // CHECK-SAME:  outputs([[VAR1]] : memref<2x6x4xsi32>) -> memref<2x6x4xsi32>

    // CHECK: return
    // CHECK-SAME: tensor<2x6x4xsi32>
}

// -----
// CHECK-LABEL: @EmbeddingSegmentsSum
func.func @EmbeddingSegmentsSum(%arg0: tensor<5x6x4xsi32>) -> tensor<7x6x4xsi32> {
    %0 = IE.EmbeddingSegmentsSum(%arg0) {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01],
        segment_ids_value = [0, 1, 2, 3, 4]} : tensor<5x6x4xsi32> -> tensor<7x6x4xsi32>
    return %0 : tensor<7x6x4xsi32>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<7x6x4xsi32>
    // CHECK:       [[VAR2:%.*]] = IERT.EmbeddingSegmentsSum {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32,
    // CHECK-SAME:       per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01], segment_ids_value = [0, 1, 2, 3, 4]}
    // CHECK-SAME:       inputs({{[^:]+}} : memref<5x6x4xsi32>) outputs([[VAR1]] : memref<7x6x4xsi32>) -> memref<7x6x4xsi32>

    // CHECK: return
    // CHECK-SAME: tensor<7x6x4xsi32>
}

// -----
// CHECK-LABEL: @DeformablePSROIPooling
  func.func @DeformablePSROIPooling(%arg0: tensor<1x441x8x8xf32>, %arg1: tensor<30x5xf32>) -> tensor<30x49x3x3xf32> {
    %0 = IE.DeformablePSROIPooling(%arg0, %arg1) {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 4 : i64, spatial_bins_y = 4 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.10000000149011612 : f64} : tensor<1x441x8x8xf32>, tensor<30x5xf32> -> tensor<30x49x3x3xf32>
    return %0 : tensor<30x49x3x3xf32>

    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<30x49x3x3xf32>
    // CHECK:       [[VAR3:%.*]] = IERT.DeformablePSROIPooling {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 4 : i64,
    // CHECK-SAME:       spatial_bins_y = 4 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.10000000149011612 : f64}
    // CHECK-SAME:       inputs({{[^:]+}} : memref<1x441x8x8xf32>, {{[^:]+}} : memref<30x5xf32>)
    // CHECK-SAME:       outputs([[VAR2]] : memref<30x49x3x3xf32>) -> memref<30x49x3x3xf32>

    // CHECK: return
    // CHECK-SAME: tensor<30x49x3x3xf32>
}

// -----

// CHECK-LABEL: @NonMaxSuppression
func.func @NonMaxSuppression(%arg0: tensor<3x100x4xf16>, %arg1: tensor<3x5x100xf16>) -> (tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>) {
    %0, %1, %2 = IE.NonMaxSuppression(%arg0, %arg1) {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} : tensor<3x100x4xf16>, tensor<3x5x100xf16> -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    return %0, %1, %2 : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>

    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<300x3xsi32>
    // CHECK:       [[VAR3:%.*]] = memref.alloc() : memref<300x3xf16>
    // CHECK:       [[VAR4:%.*]] = memref.alloc() : memref<1xsi32>
    // CHECK:       [[VAR5:%.+]], [[VAR6:%.+]], [[VAR7:%.+]] = IERT.NonMaxSuppression {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} inputs({{[^:]+}} : memref<3x100x4xf16>, {{[^:]+}} : memref<3x5x100xf16>) outputs([[VAR2]] : memref<300x3xsi32>, [[VAR3]] : memref<300x3xf16>, [[VAR4]] : memref<1xsi32>) -> memref<300x3xsi32>, memref<300x3xf16>, memref<1xsi32>

    // CHECK: return
    // CHECK-SAME: tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
}

// -----

// CHECK-LABEL: @ScatterUpdate
func.func @ScatterUpdate(%arg0: tensor<10x16x12x15xf16>, %arg1:tensor<8xsi32> , %arg2:tensor<8x16x12x15xf16>) -> (tensor<10x16x12x15xf16>) {
    %0 = IE.ScatterUpdate(%arg0, %arg1, %arg2) {axis_value = 0 : i64}:tensor<10x16x12x15xf16>, tensor<8xsi32>, tensor<8x16x12x15xf16> -> tensor<10x16x12x15xf16>
    return %0: tensor<10x16x12x15xf16>

    // CHECK:       [[VAR3:%.*]] = memref.alloc() : memref<10x16x12x15xf16>
    // CHECK:       [[VAR4:%.*]] = IERT.ScatterUpdate {axis_value = 0 : i64} inputs({{[^:]+}} : memref<10x16x12x15xf16>, {{[^:]+}} : memref<8xsi32>, {{[^:]+}} : memref<8x16x12x15xf16>) outputs([[VAR3]] : memref<10x16x12x15xf16>) -> memref<10x16x12x15xf16>

    // CHECK: return
    // CHECK-SAME: tensor<10x16x12x15xf16>
}

// -----

// CHECK-LABEL: @Tan
func.func @Tan(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16> {
    %0 = IE.Tan(%arg0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    return %0 : tensor<1x32x112x112xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x32x112x112xf16>
    // CHECK:       [[VAR2:%.*]] = IERT.Tan
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x32x112x112xf16>) -> memref<1x32x112x112xf16>

    // CHECK: return
    // CHECK-SAME: tensor<1x32x112x112xf16>
}
