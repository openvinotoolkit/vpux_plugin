//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> :
        // CHECK-SAME:       tensor<48x3x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x16x62x62xf16>
        // CHECK:       [[PERM:%.+]] = IE.MemPermute([[EXPAND]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x62x62xf16> -> tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[PERM]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        // CHECK-SAME:       tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[OUT:%.+]] = IE.MemPermute([[CONV]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>

        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
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
        // CHECK:               [[RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE]]) {axisInd = 3 : i64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[OUT:%.+]] = IE.AffineReshape([[SOFTMAX]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        // CHECK:               return [[OUT]] : tensor<1x1000xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> :
        // CHECK-SAME:       tensor<48x3x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x16x62x62xf16>
        // CHECK:       [[PERM:%.+]] = IE.MemPermute([[EXPAND]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x62x62xf16> -> tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[PERM]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        // CHECK-SAME:       tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[OUT:%.+]] = IE.MemPermute([[CONV]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>

        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %0 = IE.SoftMax(%arg) {axisInd = 3} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>

        // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        // CHECK: return [[SOFTMAX]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %0 = call @foo1(%arg) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
        %1 = call @foo2(%0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        // CHECK: [[FOO1_RES:%.+]] = call @foo1([[CONVERT]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: return [[FOO2_RES]] : tensor<1x48x60x60xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

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

        // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<64x2x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<64x2x3x3xf16>, [#const.Reorder<#NHWC>]
        // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x2x1x96xf16> = dense<0.000000e+00> : tensor<1x2x1x96xf16>
        // CHECK:           [[CONCAT:%.*]] = IE.Concat([[CST_0]], %arg0) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x2x1x96xf16>, tensor<1x2x2x96xf16> -> tensor<1x2x3x96xf16>
        // CHECK:           [[CONV:%.*]] = IE.Convolution([[CONCAT]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x2x3x96xf16>, tensor<64x2x3x3xf16, {order = #NHWC}> -> tensor<1x64x2x96xf16, {order = #NHWC}>
        // CHECK:           [[MEMPERMUTE:%.*]] = IE.MemPermute([[CONV]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x64x2x96xf16, {order = #NHWC}> -> tensor<1x64x2x96xf16>
        // CHECK:        return [[MEMPERMUTE]] : tensor<1x64x2x96xf16>
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
        %cst = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<f16>]
        %0 = IE.Add(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>

        return %0 : tensor<1x16x16x32xf16>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
        // CHECK:       [[CST_1:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<f16>]
        // CHECK:       [[PERM_CAST:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x32xf16> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 32]} inputs([[PERM_CAST]] : tensor<1x32x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x32xf16, {order = #NHWC}>
        // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPE_CAST]], [[CST]], [[CST_1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x32xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16, {order = #NHWC}>
        // CHECK:       [[SHAPE_CAST2:%.+]] = IE.ShapeCast {shape = [1, 32, 16, 16]} inputs([[GROUP_CONV]] : tensor<1x16x16x32xf16, {order = #NHWC}>) -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[PERM_CAST2:%.+]] = IE.PermuteCast([[SHAPE_CAST2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16>
        // CHECK:       return [[PERM_CAST2]] : tensor<1x16x16x32xf16>
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

        // CHECK:       [[TILE1:%.+]] = IE.PerAxisTile(%arg1) {axis = 2 : i64, tiles = 16 : i64} : tensor<1x16x1x1xf16> -> tensor<1x16x16x1xf16>
        // CHECK:       [[TILE2:%.+]] = IE.PerAxisTile([[TILE1]]) {axis = 3 : i64, tiles = 32 : i64} : tensor<1x16x16x1xf16> -> tensor<1x16x16x32xf16>
        // CHECK:       [[PERM_CAST:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x32xf16> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[PERM_CAST2:%.+]] = IE.PermuteCast([[TILE2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x32xf16> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[ADD:%.+]] = IE.Add([[PERM_CAST]], [[PERM_CAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[PERM_CAST3:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16>
        // CHECK:       return [[PERM_CAST3:%.+]] : tensor<1x16x16x32xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Test the dependency relationship between ConvertTransposedConv2DToConv2D and HandleLargeKernels
// It can convert TransposedConv with large kernel to Upsampling and Convolution
// CHECK-LABEL: @HandleTransposedConvWithLargeKernels
module @HandleTransposedConvWithLargeKernels {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x64x1x256xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x1x1036xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x64x1x256xf16>) -> tensor<1x1x1x1036xf16> {
    func.func @main(%arg0: tensor<1x64x1x256xf16>) -> tensor<1x1x1x1036xf16> {
        %weights = const.Declare tensor<1x64x1x16xf16> = dense<1.000000e+00> : tensor<1x64x1x16xf16>
        %trans_conv = IE.TransposedConvolution(%arg0, %weights) {
                        dilations = [1, 1],
                        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
                        output_padding = [0, 0],
                        pads_begin = [0, 0],
                        pads_end = [0, 0],
                        strides = [1, 4]
                    } : tensor<1x64x1x256xf16>, tensor<1x64x1x16xf16> -> tensor<1x1x1x1036xf16>

        return %trans_conv : tensor<1x1x1x1036xf16>

        // CHECK-DAG:           [[WEIGHTS1:%.+]] = const.Declare tensor<16x64x1x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x64x1x16xf16>, [#const.SubView<[0, 0, 0, 11], [1, 64, 1, 5]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
        // CHECK-DAG:           [[WEIGHTS0:%.+]] = const.Declare tensor<16x64x1x11xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x64x1x16xf16>, [#const.SubView<[0, 0, 0, 0], [1, 64, 1, 11]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
        // CHECK-DAG:           [[CST1:%.+]] = const.Declare tensor<1x64x1x15xf16> = dense<0.000000e+00> : tensor<1x64x1x15xf32>, [#const.ConvertElemType<f16>]
        // CHECK-DAG:           [[CST2:%.+]] = const.Declare tensor<1x64x1x12xf16> = dense<0.000000e+00> : tensor<1x64x1x12xf32>, [#const.ConvertElemType<f16>]
        // CHECK:               [[UPSAMPLE:%.+]] = IE.Upsampling([[ARG0]]) {
        // CHECK-SAME:                  pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 3]>, upsampling_factor = [4, 1, 1]
        // CHECK-SAME:              } : tensor<1x64x1x256xf16> -> tensor<1x64x1x1024xf16>
        // CHECK:               [[CONCAT:%.+]] = IE.Concat([[CST1]], [[UPSAMPLE]], [[CST2]]) {
        // CHECK-SAME{LITERAL}:         static_offsets = [[0, 0, 0, 0], [0, 0, 0, 15], [0, 0, 0, 1039]]
        // CHECK-SAME:              } : tensor<1x64x1x15xf16>, tensor<1x64x1x1024xf16>, tensor<1x64x1x12xf16> -> tensor<1x64x1x1051xf16>
        // CHECK:               [[SLICE0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 64, 1, 1046] : tensor<1x64x1x1051xf16> to tensor<1x64x1x1046xf16>
        // CHECK:               [[PERMUTE0:%.+]] = IE.MemPermute([[SLICE0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x1x1046xf16> -> tensor<1x64x1x1046xf16, {order = #NHWC}>
        // CHECK:               [[CONV0:%.+]] = IE.Convolution([[PERMUTE0]], [[WEIGHTS0]]) {
        // CHECK-SAME:                  dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        // CHECK-SAME:                  } : tensor<1x64x1x1046xf16, {order = #NHWC}>, tensor<16x64x1x11xf16, {order = #NHWC}> -> tensor<1x16x1x1036xf16, {order = #NHWC}>

        // CHECK:               [[SLICE1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 64, 1, 1040] : tensor<1x64x1x1051xf16> to tensor<1x64x1x1040xf16>
        // CHECK:               [[PERMUTE1:%.+]] = IE.MemPermute([[SLICE1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x1x1040xf16> -> tensor<1x64x1x1040xf16, {order = #NHWC}>
        // CHECK:               [[CONV1:%.+]] = IE.Convolution([[PERMUTE1]], [[WEIGHTS1]]) {
        // CHECK-SAME:                  dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        // CHECK-SAME:              } : tensor<1x64x1x1040xf16, {order = #NHWC}>, tensor<16x64x1x5xf16, {order = #NHWC}> -> tensor<1x16x1x1036xf16, {order = #NHWC}>

        // CHECK:               [[ADD:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
        // CHECK-SAME:              } : tensor<1x16x1x1036xf16, {order = #NHWC}>, tensor<1x16x1x1036xf16, {order = #NHWC}> -> tensor<1x16x1x1036xf16, {order = #NHWC}>

        // CHECK:               [[SLICE_OUT:%.+]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 1, 1, 1036] : tensor<1x16x1x1036xf16, {order = #NHWC}> to tensor<1x1x1x1036xf16, {order = #NHWC}>
        // CHECK:               [[PERMUTE_OUT:%.+]] = IE.PermuteCast([[SLICE_OUT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x1036xf16, {order = #NHWC}> -> tensor<1x1x1x1036xf16>
        // CHECK:               return [[PERMUTE_OUT]] : tensor<1x1x1x1036xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Test the dependency relationship between ConvertGroupConvToConv and HandleLargeKernels
// It can convert GroupConv with large kernel to NCEConvolution
// CHECK-LABEL: @HandleGroupConvWithLargeKernels
module @HandleGroupConvWithLargeKernels {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x128x1x112xf16>
        DataInfo "input1" : tensor<128x64x1x22xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x128x1x91xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x128x1x112xf16>, [[ARG1:%.+]]: tensor<128x64x1x22xf16>) -> tensor<1x128x1x91xf16> {
    func.func @main(%arg0: tensor<1x128x1x112xf16>, %arg1: tensor<128x64x1x22xf16>) -> tensor<1x128x1x91xf16> {
        %group_conv = IE.GroupConvolution(%arg0, %arg1) {
                        dilations = [1, 1],
                        groups = 2,
                        pads_begin = [0, 0],
                        pads_end = [0, 0],
                        strides = [1, 1]
                    } : tensor<1x128x1x112xf16>, tensor<128x64x1x22xf16> -> tensor<1x128x1x91xf16>

        return %group_conv : tensor<1x128x1x91xf16>

        // CHECK:   [[PERMUTE_IN:%.+]] = IE.MemPermute([[ARG0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x1x112xf16> -> tensor<1x128x1x112xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_0:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 0] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_1:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 11] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_2:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 0] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_3:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 11] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTE_WEIGHT:%.+]] = IE.MemPermute([[ARG1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<128x64x1x22xf16> -> tensor<128x64x1x22xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_WEIGHT_0:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 0] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_WEIGHT_1:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 11] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_WEIGHT_2:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 0] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_WEIGHT_3:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 11] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>

        // CHECK:   [[CONV_0:%.+]] = IE.Convolution([[SLICE_IN_0]], [[SLICE_WEIGHT_0]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.+]] = IE.Convolution([[SLICE_IN_1]], [[SLICE_WEIGHT_1]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[GROUP_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x91xf16, {order = #NHWC}>, tensor<1x64x1x91xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>

        // CHECK:   [[CONV_2:%.+]] = IE.Convolution([[SLICE_IN_2]], [[SLICE_WEIGHT_2]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[CONV_3:%.+]] = IE.Convolution([[SLICE_IN_3]], [[SLICE_WEIGHT_3]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[GROUP_1:%.+]] = IE.Add([[CONV_2]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x91xf16, {order = #NHWC}>, tensor<1x64x1x91xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>

        // CHECK:   [[CONCAT:%.+]] = IE.Concat([[GROUP_0]], [[GROUP_1]]) {
        // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1x91xf16, {order = #NHWC}>, tensor<1x64x1x91xf16, {order = #NHWC}> -> tensor<1x128x1x91xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTE_OUT:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x128x1x91xf16, {order = #NHWC}> -> tensor<1x128x1x91xf16>
        // CHECK:   return [[PERMUTE_OUT]] : tensor<1x128x1x91xf16>
    }
}
