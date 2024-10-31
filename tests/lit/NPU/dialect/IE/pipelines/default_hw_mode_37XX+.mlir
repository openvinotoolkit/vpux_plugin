//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FuseConstDivideToMatMul
module @FuseConstDivideToMatMul {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x3x24xf16>
        DataInfo "input" : tensor<1x64x3x24xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK-LABEL: @main
    func.func @main(%arg0: tensor<1x3x64x24xf16>, %arg1: tensor<1x3x64x24xf16>) -> tensor<1x3x64x64xf16> {
        %cst_0 = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
        %cst_1 = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>
        %cst_16 = const.Declare tensor<1xf16> = dense<-8.01463317> : tensor<1xf16>
        %cst_17 = const.Declare tensor<1xf16> = dense<7.95201873> : tensor<1xf16>
        %cst_18 = const.Declare tensor<1xf16> = dense<2.460000e+02> : tensor<1xf16>
        %cst_fq = IE.FakeQuantize(%cst_18, %cst_0, %cst_1, %cst_16, %cst_17) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
        } : tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>

        %28 = IE.MatMul(%arg0, %arg1) {transpose_b}
            : tensor<1x3x64x24xf16>, tensor<1x3x64x24xf16> -> tensor<1x3x64x64xf16>

        %29 = IE.Divide(%28, %cst_fq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x3x64x64xf16>, tensor<1xf16> -> tensor<1x3x64x64xf16>

        return %29 : tensor<1x3x64x64xf16>

        // CHECK: [[CONV0:%.+]] = IE.Convolution
        // CHECK-SAME:  static_scale = 0.135327876 : f32
        // CHECK: [[CONV0_RESHAPE:%.+]] = IE.AffineReshape([[CONV0]])
        // CHECK-SAME: -> tensor<1x64x64x1xf16, {order = #NHWC}>
        // CHECK: [[CONV0_PERMUTE:%.+]] = IE.PermuteCast([[CONV0_RESHAPE]])

        // CHECK: [[CONV1:%.+]] = IE.Convolution
        // CHECK-SAME:  static_scale = 0.135327876 : f32
        // CHECK: [[CONV1_RESHAPE:%.+]] = IE.AffineReshape([[CONV1]])
        // CHECK-SAME: -> tensor<1x64x64x1xf16, {order = #NHWC}>
        // CHECK: [[CONV1_PERMUTE:%.+]] = IE.PermuteCast([[CONV1_RESHAPE]])

        // CHECK: [[CONV2:%.+]] = IE.Convolution
        // CHECK-SAME:  static_scale = 0.135327876 : f32
        // CHECK: [[CONV2_RESHAPE:%.+]] = IE.AffineReshape([[CONV2]])
        // CHECK-SAME: -> tensor<1x64x64x1xf16, {order = #NHWC}>
        // CHECK: [[CONV2_PERMUTE:%.+]] = IE.PermuteCast([[CONV2_RESHAPE]])


        // CHECK: [[CONV0_RES:%.+]] = IE.AffineReshape([[CONV0_PERMUTE]])
        // CHECK: [[CONV1_RES:%.+]] = IE.AffineReshape([[CONV1_PERMUTE]])
        // CHECK: [[CONV2_RES:%.+]] = IE.AffineReshape([[CONV2_PERMUTE]])
        // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0_RES]], [[CONV1_RES]], [[CONV2_RES]])
        // CHECK-NEXT: return [[CONCAT]]
    }
}

// -----

// CHECK-LABEL: @SoftMax
module @SoftMax {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
        %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
        return %0 : tensor<1x1000xf16>
        // CHECK:               [[RESHAPE_RES:%.+]] = IE.AffineReshape([[ARG0]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[SOFTMAX_RES:%.+]] = IE.SoftMax([[RESHAPE_RES]]) {axisInd = 3 : i64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[OUT:%.+]] = IE.AffineReshape([[SOFTMAX_RES]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        // CHECK:               return [[OUT]] : tensor<1x1000xf16>
    }
}

// -----

// CHECK-LABEL: @RepeatingBlocks
module @RepeatingBlocks {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    // CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func private @main_fn1(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
        %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu = IE.ReLU(%conv) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %relu : tensor<1x48x60x60xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK:       [[SHAPECAST1:%.+]] = IE.ShapeCast {shape = [1, 48, 225, 16]} inputs([[ARG0]] : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        // CHECK:       [[PERMQUANT:%.+]] = IE.PermuteQuantize([[SHAPECAST1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
        // CHECK-SAME:      : tensor<1x48x225x16xf16> -> tensor<1x48x225x16xf16, {order = #NHWC}>
        // CHECK:       [[SHAPECAST2:%.+]] = IE.ShapeCast {shape = [1, 48, 60, 60]} inputs([[PERMQUANT]] : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[SHAPECAST2]], [[CST]]) {
        // CHECK-SAME:          dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]
        // CHECK-SAME:      } : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        // CHECK:       return [[CONV]]
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax = IE.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %call1 = call @main_fn1(%softmax) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %call2 = call @main_fn1(%call1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %call2 : tensor<1x48x60x60xf32>

        // CHECK:       [[CONVERT1:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        // CHECK:       [[PERMUTECAST1:%.+]] = IE.PermuteCast([[CONVERT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x48x60x60xf16> -> tensor<1x60x48x60xf16, {order = #NHWC}>
        // CHECK:       [[SHAPECAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 48, 225]} inputs([[PERMUTECAST1]] : tensor<1x60x48x60xf16, {order = #NHWC}>) -> tensor<1x16x48x225xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL1:%.+]] = IE.MaxPool([[SHAPECAST1]]) {
        // CHECK-SAME:          kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
        // CHECK-SAME:      } : tensor<1x16x48x225xf16, {order = #NHWC}> -> tensor<1x16x48x225xf16, {order = #NWCH}>
        // CHECK:       [[PERMUTECAST2:%.+]] = IE.PermuteCast([[MAXPOOL1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x48x225xf16, {order = #NWCH}> -> tensor<1x48x225x16xf16, {order = #NHWC}>
        // CHECK:       [[SHAPECAST2:%.+]] = IE.ShapeCast {shape = [1, 48, 60, 60]} inputs([[PERMUTECAST2]] : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[SHAPECAST2]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL2:%.+]] = IE.MaxPool([[SOFTMAX]]) {
        // CHECK-SAME:          kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
        // CHECK-SAME:      } : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>

        // CHECK:       [[CALL1:%.+]] = call @main_fn1([[MAXPOOL2]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>

        // CHECK:       [[CONVERT2:%.+]] = IE.Convert([[CALL2]]) {dstElemType = f32} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf32>
        // CHECK:       return [[CONVERT2]]
    }
}

// -----

// CHECK-LABEL: @GroupConvolution
module @GroupConvolution {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2x2x96xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x64x2x96xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x2x2x96xf16>) -> tensor<1x64x2x96xf16>
    func.func @main(%arg0: tensor<1x2x2x96xf16>) -> tensor<1x64x2x96xf16> {
        %cst = const.Declare tensor<64x1x3x3xf16> = dense<1.0> : tensor<64x1x3x3xf16>
        %1 = IE.GroupConvolution(%arg0, %cst) {
            dilations = [1, 1],
            groups = 2 : i64,
            pads_begin = [2, 1],
            pads_end = [0, 1],
            strides = [1, 1]
        } : tensor<1x2x2x96xf16>, tensor<64x1x3x3xf16> -> tensor<1x64x2x96xf16>

        return %1 : tensor<1x64x2x96xf16>

        // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<64x2x3x3xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 14, 0, 0]>]
        // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<1x2x1x96xf16> = dense<0.000000e+00> : tensor<1x2x1x96xf16>
        // CHECK:           [[CONCAT:%.+]] = IE.Concat([[CST_0]], %arg0) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x2x1x96xf16>, tensor<1x2x2x96xf16> -> tensor<1x2x3x96xf16>
        // CHECK:           [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[CONCAT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 14, 0, 0]} : tensor<1x2x3x96xf16> -> tensor<1x16x3x96xf16, {order = #NHWC}>
        // CHECK:           [[CONV:%.+]] = IE.Convolution([[PERMUTEQUANTIZE]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x3x96xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x2x96xf16>
        // CHECK:        return [[CONV]] : tensor<1x64x2x96xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BroadcastAdd
module @BroadcastAdd {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x16x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x16x32xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x16x16x32xf16>) -> tensor<1x16x16x32xf16> {
    func.func @main(%arg0: tensor<1x16x16x32xf16>) -> tensor<1x16x16x32xf16> {
        %cst = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>, [#const.CastElemType<f16>]
        %0 = IE.Add(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>

        return %0 : tensor<1x16x16x32xf16>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
        // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<f16>]
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16, {order = #NHWC}>
        // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[PERM]], [[CST]], [[CST_0]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x32xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>
        // CHECK:       return [[GROUP_CONV]] : tensor<1x16x16x32xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAddToScaleShift
module @ConvertAddToScaleShift {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x16x32xf16>
        DataInfo "input1" : tensor<1x16x1x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x16x32xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x16x16x32xf16>, [[ARG1:%.+]]: tensor<1x16x1x1xf16>) -> tensor<1x16x16x32xf16> {
    func.func @main(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x16x32xf16> {
        %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>

        return %0 : tensor<1x16x16x32xf16>

        // CHECK:       [[PERM_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf16> -> tensor<1x1x16x1xf16, {order = #NHWC}>
        // CHECK:       [[TILE:%.+]] = IE.Tile([[PERM_1]]) {repeats_values = [1, 32, 1, 16]} : tensor<1x1x16x1xf16, {order = #NHWC}> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[PERM_2:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x32xf16> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[ADD:%.+]] = IE.Add([[PERM_2]], [[TILE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[PERM_3:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16>
        // CHECK:       return [[PERM_3]] : tensor<1x16x16x32xf16>

    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// This test indicates there is a dependancy between ConvertStridedSlice2Conv and AdjustConvolutionShape
// for converting a StridedSlice to Convolution
// CHECK-LABEL: @StridedSlice
module @StridedSlice {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x640x640xf16, {order = #NHWC}>
    }
    outputsInfo : {
        DataInfo "stridedslice" : tensor<1x3x640x320xf16, {order = #NHWC}>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x640x320xf16, {order = #NHWC}> {
    func.func @main(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x640x320xf16, {order = #NHWC}> {
        %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x640x320xf16, {order = #NHWC}>
        return %0 : tensor<1x3x640x320xf16, {order = #NHWC}>

        // CHECK [[CST_0:%.+]] = const.Declare tensor<48x96x1x1xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<48x96x1x1xf16, {order = #NHWC}>
        // CHECK [[CST_1:%.+]] = const.Declare tensor<1x3x640x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x640x1xf32>, [#const.Reorder<#NHWC>, #const.CastElemType<f16>]
        // CHECK [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 1] [1, 3, 640, 639] : tensor<1x3x640x640xf16, {order = #NHWC}> to tensor<1x3x640x639xf16, {order = #NHWC}>
        // CHECK [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST_0]]) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 639]]} : tensor<1x3x640x639xf16, {order = #NHWC}>, tensor<1x3x640x1xf16, {order = #NHWC}> -> tensor<1x3x640x640xf16, {order = #NHWC}>
        // CHECK [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 96, 640, 20]} inputs([[CONCAT]] : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x96x640x20xf16, {order = #NHWC}>
        // CHECK [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_IN]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x640x20xf16, {order = #NHWC}>, tensor<48x96x1x1xf16, {order = #NHWC}> -> tensor<1x48x640x20xf16, {order = #NHWC}>
        // CHECK [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 640, 320]} inputs([[CONV]] : tensor<1x48x640x20xf16, {order = #NHWC}>) -> tensor<1x3x640x320xf16, {order = #NHWC}>
        // CHECK return [[SHAPE_CAST_OUT]] : tensor<1x3x640x320xf16, {order = #NHWC}>
    }
}

// -----

// CHECK-LABEL: @MultiNonTrivialDimMultiplyToConv
module @MultiNonTrivialDimMultiplyToConv {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x19x80x80xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x19x80x80xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x19x80x80xf16>) -> tensor<1x19x80x80xf16> {
    func.func @main(%arg0: tensor<1x19x80x80xf16>) -> tensor<1x19x80x80xf16> {
        %MUL_WEIGHTS = const.Declare tensor<1x1x80x80xf16> = dense<2.000000e+00> : tensor<1x1x80x80xf16>
        %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>
        } : tensor<1x19x80x80xf16>, tensor<1x1x80x80xf16> -> tensor<1x19x80x80xf16>

        return %MUL : tensor<1x19x80x80xf16>

        // CHECK-DAG:       [[MUL_WEIGHTS:%.+]] = const.Declare tensor<1600x1x1x1xf16, {order = #NHWC}> = dense<2.000000e+00>
        // CHECK-SAME           : tensor<1x1x80x80xf16>, [#const.Reshape<[1, 6400, 1, 1]>, #const.Reshape<[6400, 1, 1, 1]>, #const.SubView<[0, 0, 0, 0], [1600, 1, 1, 1]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
        // CHECK-SAME:      shape_value = [1, 1, 19, 6400]
        // CHECK-SAME:  } : tensor<1x19x80x80xf16> -> tensor<1x1x19x6400xf16>

        // CHECK:   [[PERMUTE_INPUT:%.+]] = IE.PermuteCast([[RESHAPE_INPUT]]) {
        // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
        // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x6400x1x19xf16, {order = #NHWC}>

        // CHECK:   [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 1600, 4, 19]} inputs([[PERMUTE_INPUT]] : tensor<1x6400x1x19xf16, {order = #NHWC}>) -> tensor<1x1600x4x19xf16, {order = #NHWC}>

        // CHECK:   [[MUL:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[MUL_WEIGHTS]]) {
        // CHECK-SAME:      dilations = [1, 1], groups = 1600 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1600x4x19xf16, {order = #NHWC}>, tensor<1600x1x1x1xf16, {order = #NHWC}> -> tensor<1x1600x4x19xf16, {order = #NHWC}>

        // CHECK:   [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 6400, 1, 19]} inputs([[MUL]] : tensor<1x1600x4x19xf16, {order = #NHWC}>) -> tensor<1x6400x1x19xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTE_OUT:%.+]] = IE.PermuteCast([[SHAPECAST_OUT]]) {
        // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
        // CHECK-SAME:  } : tensor<1x6400x1x19xf16, {order = #NHWC}> -> tensor<1x1x19x6400xf16>

        // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[PERMUTE_OUT]]) {
        // CHECK-SAME:      shape_value = [1, 19, 80, 80]
        // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x19x80x80xf16>

        // CHECK:   return [[RESHAPE_OUT]] : tensor<1x19x80x80xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HandleFirstPermuteOnNCE
module @HandleFirstPermuteOnNCE {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x384x384xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x384x384xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x384x384xui8>) -> tensor<1x3x384x384xf16> {
    func.func @main(%arg0: tensor<1x3x384x384xui8>) -> tensor<1x3x384x384xf16> {
        %cst = const.Declare tensor<1x3x1x1xf16> = dense<127.5> : tensor<1x3x1x1xf16>
        %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<127.5> : tensor<1x3x1x1xf16>

        %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x384x384xui8> -> tensor<1x3x384x384xf16>
        %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x384x384xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x384x384xf16>
        %2 = IE.Add(%1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x384x384xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x384x384xf16>

        return %2 : tensor<1x3x384x384xf16>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.275000e+02> : tensor<1x3x1x1xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        // CHECK:       [[CST_0:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.275000e+02> : tensor<1x3x1x1xf16>, [#const.Reshape<[3, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>]
        // CHECK:       [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x384x384xui8> -> tensor<1x3x384x384xf16>
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize([[CONVERT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x384x384xf16> -> tensor<1x16x384x384xf16, {order = #NHWC}>
        // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[PERM]], [[CST_0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:          tensor<1x16x384x384xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x384x384xf16>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[GROUP_CONV]] [0, 0, 0, 0] [1, 3, 384, 384] : tensor<1x16x384x384xf16> to tensor<1x3x384x384xf16>
        // CHECK:       return [[SLICE]] : tensor<1x3x384x384xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeGroupConvWithBiasConcatPostClamp
module @OptimizeGroupConvWithBiasConcatPostClamp {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x144x144xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x144x144xf16>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x16x144x144xf16>
    func.func @main(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
        %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
        %cst_1 = const.Declare tensor<16x1x1x1xf16> = dense<1.0> : tensor<16x1x1x1xf16>
        %cst_2 = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
        %0 = IE.Convolution(%arg0, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
        %1 = IE.GroupConvolution(%0, %cst_1, %cst_2) {
            dilations = [1, 1],
            groups = 16 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x144x144xf16>, tensor<16x1x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x144x144xf16>
        %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>
        %3 = IE.Clamp(%2) {
            max = 6.000000e+00,
            min = 0.000000e+00
        } : tensor<1x32x144x144xf16> -> tensor<1x32x144x144xf16>

        return %3 : tensor<1x32x144x144xf16>

        // CHECK-DAG:       [[NEW_WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x32x1x1xf16>

        // CHECK:           [[PERMUTE:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x144x144xf16> -> tensor<1x16x144x144xf16, {order = #NHWC}>
        // CHECK:           [[CONV:%.+]] = IE.Convolution([[PERMUTE]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x144x144xf16, {order = #NHWC}>
        // CHECK:           [[NEW_CONV:%.+]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [1, 1]} : tensor<1x16x144x144xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>, tensor<1x32x1x1xf16> -> tensor<1x32x144x144xf16>

        // CHECK:           return [[NEW_CONV]] : tensor<1x32x144x144xf16>
    }
}

// -----

// CHECK-LABEL: @ReduceMax
module @ReduceMax {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x42840x17xf16>
    }
    outputsInfo : {
        DataInfo "reducemax" : tensor<1x42840x1xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x42840x17xf16>)
    func.func @main(%arg0: tensor<1x42840x17xf16>) -> tensor<1x42840x1xf16> {
         %cst = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
        %0 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x42840x17xf16>, tensor<1xsi64> -> tensor<1x42840x1xf16>
        return %0 : tensor<1x42840x1xf16>
    }

        // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]] [0, 0, 0] [1, 42840, 1] : tensor<1x42840x17xf16> to tensor<1x42840x1xf16>
        // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[ARG0]], [[SLICE1]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 0, 17]]} : tensor<1x42840x17xf16>, tensor<1x42840x1xf16> -> tensor<1x42840x18xf16>
        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CONCAT_1]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]} : tensor<1x42840x18xf16> -> tensor<1x42840x32xf16>
        // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[EXPAND]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 6, 7140, 32]} : tensor<1x42840x32xf16> -> tensor<1x6x7140x32xf16>
        // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[AFFINERESHAPE1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x6x7140x32xf16> -> tensor<1x16x7140x32xf16, {order = #NHWC}>
        // CHECK:       [[SLICE2:%.+]] = IE.Slice [[PERMUTEQUANTIZE]] [0, 0, 0, 0] [1, 16, 7140, 18] : tensor<1x16x7140x32xf16, {order = #NHWC}> to tensor<1x16x7140x18xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL1:%.+]] = IE.MaxPool([[SLICE2]]) {kernel_size = [1, 6], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 6]} : tensor<1x16x7140x18xf16, {order = #NHWC}> -> tensor<1x16x7140x3xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]]) {kernel_size = [1, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x7140x3xf16, {order = #NHWC}> -> tensor<1x16x7140x1xf16>
        // CHECK:       [[SLICE3:%.+]] = IE.Slice [[MAXPOOL2]] [0, 0, 0, 0] [1, 6, 7140, 1] : tensor<1x16x7140x1xf16> to tensor<1x6x7140x1xf16
        // CHECK:       [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[SLICE3]])
        // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 42840, 1, 1]} : tensor<1x6x7140x1xf16> -> tensor<1x42840x1x1xf16>
        // CHECK:       [[AFFINERESHAPE3:%.+]] = IE.AffineReshape([[AFFINERESHAPE2]])
        // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 42840, 1]} : tensor<1x42840x1x1xf16> -> tensor<1x42840x1xf16>
        // CHECK:       return [[AFFINERESHAPE3]] : tensor<1x42840x1xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<i4:f16, 2.000000e+00>
// CHECK: #HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK: #NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @GroupMatMulI4Weights
module @GroupMatMulI4Weights {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x1x4096xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x8192xf32>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x4096xf32>) -> tensor<1x1x8192xf32> {
    func.func @main(%arg0: tensor<1x1x4096xf32>) -> tensor<1x1x8192xf32> {
        %cst_0 = const.Declare tensor<8192x2x2048xf16>
                = dense<1> : tensor<8192x2x2048xsi8>, [#const.CastElemType<si4>, #const.CastElemType<f16>]
        %cst_1 = const.Declare tensor<8192x2x1xf16>
                = dense<2.0> : tensor<8192x2x1xf16>
        %0 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<8192x2x2048xf16>, tensor<8192x2x1xf16> -> tensor<8192x2x2048xf16>
        %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1]], shape_value = [8192, 4096]}
            : tensor<8192x2x2048xf16> -> tensor<8192x4096xf16>
        %2 = IE.Convert(%1) {dstElemType = f32} : tensor<8192x4096xf16> -> tensor<8192x4096xf32>
        %3 = IE.MatMul(%arg0, %2) {transpose_b}
            : tensor<1x1x4096xf32>, tensor<8192x4096xf32> -> tensor<1x1x8192xf32>

        return %3 : tensor<1x1x8192xf32>

        // CHECK-DAG:       [[ACCUM_WEIGHTS0:%.+]] = const.Declare tensor<16x1x2x1xf16, {order = #NHWC}> = dense<5.000000e-01>
        // CHECK-DAG:       [[ACCUM_WEIGHTS1:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<2.000000e+00>

        // CHECK:       [[WEIGHTS_GRP0:%.+]] = const.Declare tensor<8192x2048x1x1x!qElemType, {order = #NHWC}>
        // CHECK-SAME:      tensor<8192x2x2048xsi8>, [#const.SubView<[0, 1, 0], [8192, 1, 2048]>, #const.CastElemType<si4>,
        // CHECK-SAME:                               #const.CastElemType<f16>, #const.Transpose<#HWC>,
        // CHECK-SAME:                               #const.Reshape<[1, 2048, 1, 8192]>, #const.CastElemType<si4>,
        // CHECK-SAME:                               #const.CastElemType<!qElemType>, #const.Transpose<#NWHC>,
        // CHECK-SAME:                               #const.Reshape<[8192, 2048, 1, 1]>, #const.Reorder<#NHWC>]

        // CHECK:       [[WEIGHTS_GRP1:%.+]] = const.Declare tensor<8192x2048x1x1x!qElemType, {order = #NHWC}>
        // CHECK-SAME:      tensor<8192x2x2048xsi8>, [#const.SubView<[0, 0, 0], [8192, 1, 2048]>, #const.CastElemType<si4>,
        // CHECK-SAME:                               #const.CastElemType<f16>, #const.Transpose<#HWC>,
        // CHECK-SAME:                               #const.Reshape<[1, 2048, 1, 8192]>, #const.CastElemType<si4>,
        // CHECK-SAME:                               #const.CastElemType<!qElemType>, #const.Transpose<#NWHC>,
        // CHECK-SAME:                               #const.Reshape<[8192, 2048, 1, 1]>, #const.Reorder<#NHWC>]

        // CHECK:    [[VAL0:%.+]] = IE.AffineReshape([[ARG0]])
        // CHECK-SAME:         : tensor<1x1x4096xf32> -> tensor<1x1x1x4096xf32>
        // CHECK:    [[VAL1:%.+]] = IE.Convert([[VAL0]]) {dstElemType = f16} : tensor<1x1x1x4096xf32> -> tensor<1x1x1x4096xf16>
        // CHECK:    [[VAL2:%.+]] = IE.AffineReshape([[VAL1]])
        // CHECK-SAME:          : tensor<1x1x1x4096xf16> -> tensor<1x4096x1x1xf16>
        // CHECK:    [[VAL3:%.+]] = IE.PermuteCast([[VAL2]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:          : tensor<1x4096x1x1xf16> -> tensor<1x4096x1x1xf16, {order = #NHWC}>
        // CHECK:    [[GRP1:%.+]] = IE.Slice [[VAL3]] [0, 0, 0, 0] [1, 2048, 1, 1]
        // CHECK-SAME:          : tensor<1x4096x1x1xf16, {order = #NHWC}> to tensor<1x2048x1x1xf16, {order = #NHWC}>
        // CHECK:    [[GRP0:%.+]] = IE.Slice [[VAL3]] [0, 2048, 0, 0] [1, 2048, 1, 1]
        // CHECK-SAME:          : tensor<1x4096x1x1xf16, {order = #NHWC}> to tensor<1x2048x1x1xf16, {order = #NHWC}>

        // CHECK:    [[CONV_GRP1:%.+]] = IE.Convolution([[GRP1]], [[WEIGHTS_GRP1]])
        // CHECK:    [[CONV_GRP0:%.+]] = IE.Convolution([[GRP0]], [[WEIGHTS_GRP0]])
        // CHECK:    [[CONCAT:%.+]] = IE.Concat([[CONV_GRP1]], [[CONV_GRP0]])
        // CHECK-SAME:     : tensor<1x8192x1x1xf16, {order = #NHWC}>, tensor<1x8192x1x1xf16, {order = #NHWC}> -> tensor<2x8192x1x1xf16, {order = #NHWC}>

        // CHECK:    [[PERM_CAST:%.+]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NWCH}
        // CHECK-SAME:      : tensor<2x8192x1x1xf16, {order = #NHWC}> -> tensor<2x8192x1x1xf16>
        // CHECK:    [[RESHAPE:%.+]] = IE.Reshape([[PERM_CAST]])
        // CHECK-SAME:      : tensor<2x8192x1x1xf16> -> tensor<1x2x512x16xf16>
        // CHECK:    [[PERM_CAST1:%.+]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW}
        // CHECK-SAME:      : tensor<1x2x512x16xf16> -> tensor<1x16x2x512xf16, {order = #NHWC}>

        // CHECK:    [[ACCUM_DWCONV0:%.+]] = IE.GroupConvolution([[PERM_CAST1]], [[ACCUM_WEIGHTS0]])
        // CHECK:    [[RESHAPE1:%.+]] = IE.AffineReshape([[ACCUM_DWCONV0]])
        // CHECK-SAME:      : tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x4x128xf16, {order = #NHWC}>

        // CHECK:    [[ACCUM_DWCONV1:%.+]] = IE.GroupConvolution([[RESHAPE1]], [[ACCUM_WEIGHTS1]])
        // CHECK:    [[RESHAPE2:%.+]] = IE.AffineReshape([[ACCUM_DWCONV1]])
        // CHECK-SAME:      : tensor<1x16x4x128xf16, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>

        // CHECK:    [[PERM_CAST_OUT:%.+]] = IE.PermuteCast([[RESHAPE2]]) {dst_order = #NCHW, mem_perm = #NCHW}
        // CHECK-SAME:      : tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x1x512x16xf16>
        // CHECK:    [[RESHAPE_OUT0:%.+]] = IE.AffineReshape([[PERM_CAST_OUT]])
        // CHECK-SAME:      : tensor<1x1x512x16xf16> -> tensor<1x1x1x8192xf16>

        // CHECK:    [[CONVERT:%.+]] = IE.Convert([[RESHAPE_OUT0]])
        // CHECK-SAME:      : tensor<1x1x1x8192xf16> -> tensor<1x1x1x8192xf32>
        // CHECK:    [[RESHAPE_OUT1:%.+]] = IE.AffineReshape([[CONVERT]])
        // CHECK-SAME:      : tensor<1x1x1x8192xf32> -> tensor<1x1x8192xf32>
    }
}
