//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX37XX

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

        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
        // CHECK-SAME:       tensor<1x3x62x64xf16> -> tensor<1x16x62x64xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERM]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>

        // CHECK:       [[OUT:%.+]] = IE.Convolution([[SLICE]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        // CHECK-SAME:       tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8<0:1>:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.027773432638130938:101>
!qElemType2 = !quant.uniform<u8:f16, 0.013886716319065469:101>
!qElemType3 = !quant.uniform<u8:f16, 0.0069433581595327344:101>

// CHECK-LABEL: @Depth2SpaceToTransConv
module @Depth2SpaceToTransConv {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x512x16x16xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x128x32x32xf16>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x512x16x16xf16>)
    func.func @main(%input: tensor<1x512x16x16xf16>) -> tensor<1x128x32x32xf16> {
        %lowFq = const.Declare tensor<1x1x1x1xf16> = dense<-1.39694476> : tensor<1x1x1x1xf32>, [ #const.ConvertElemType<f16> ]
        %highFq = const.Declare tensor<1x1x1x1xf16> = dense<2.1441679> : tensor<1x1x1x1xf32>, [ #const.ConvertElemType<f16> ]

        %fq1 = IE.FakeQuantize(%input, %lowFq, %highFq, %lowFq, %highFq) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 256 : i64
        } :
            tensor<1x512x16x16xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16> ->
            tensor<1x512x16x16xf16>

        %d2s = IE.DepthToSpace(%fq1) {
            block_size = 2 : i64,
            mode = #IE.depth_to_space_mode<DEPTH_FIRST>
        } : tensor<1x512x16x16xf16> -> tensor<1x128x32x32xf16>

        %fq2 = IE.FakeQuantize(%d2s, %lowFq, %highFq, %lowFq, %highFq) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 256 : i64
        } :
            tensor<1x128x32x32xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16> ->
            tensor<1x128x32x32xf16>

        return %fq2 : tensor<1x128x32x32xf16>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<128x512x2x2x!qElemType, {order = #NHWC}> =
        // CHECK-SAME:      dense_resource<__elided__> : tensor<128x512x2x2xui8, {order = #NHWC}>,
        // CHECK-SAME:      [#const.ConvertElemType<f16>, #const.Reorder<#NCHW>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]

        // CHECK:       [[PERMQUANT:%.+]] = IE.PermuteQuantize([[INPUT]]) {
        // CHECK-SAME:      dstElemType = f16,
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #NHWC,
        // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
        // CHECK-SAME:      pads_end = [0, 0, 0, 0]
        // CHECK-SAME:  } : tensor<1x512x16x16xf16> -> tensor<1x512x16x16xf16, {order = #NHWC}>

        // CHECK:       [[ADD1:%.+]] = IE.Add([[PERMQUANT]], [[PERMQUANT]]) {
        // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
        // CHECK-SAME:  } :
        // CHECK-SAME:      tensor<1x512x16x16xf16, {order = #NHWC}>,
        // CHECK-SAME:      tensor<1x512x16x16xf16, {order = #NHWC}> ->
        // CHECK-SAME:      tensor<1x512x16x16x!qElemType1, {order = #NHWC}>

        // CHECK:       [[QUANTCAST1:%.+]] = IE.QuantizeCast([[ADD1]]) {
        // CHECK-SAME:      dstElemType = !qElemType2
        // CHECK-SAME:  } : tensor<1x512x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x512x16x16x!qElemType2, {order = #NHWC}>

        // CHECK:       [[UPSAMPLE:%.+]] = IE.Upsampling([[QUANTCAST1]]) {
        // CHECK-SAME:      pad = #IE.UpsamplingPad<
        // CHECK-SAME:          pads_channel = [0, 0],
        // CHECK-SAME:          pads_height = [0, 1],
        // CHECK-SAME:          pads_width = [0, 1]
        // CHECK-SAME:      >,
        // CHECK-SAME:      upsampling_factor = [2, 2, 1]
        // CHECK-SAME:  } : tensor<1x512x16x16x!qElemType2, {order = #NHWC}> -> tensor<1x512x32x32x!qElemType2, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = IE.Convolution([[UPSAMPLE]], [[WEIGHTS]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [1, 1],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } :
        // CHECK-SAME:      tensor<1x512x32x32x!qElemType2, {order = #NHWC}>,
        // CHECK-SAME:      tensor<128x512x2x2x!qElemType, {order = #NHWC}> ->
        // CHECK-SAME:      tensor<1x128x32x32x!qElemType2, {order = #NHWC}>

        // CHECK:       [[QUANTCAST2:%.+]] = IE.QuantizeCast([[CONV]]) {
        // CHECK-SAME:      dstElemType = !qElemType3
        // CHECK-SAME:  } : tensor<1x128x32x32x!qElemType2, {order = #NHWC}> -> tensor<1x128x32x32x!qElemType3, {order = #NHWC}>

        // CHECK:       [[ADD2:%.+]] = IE.Add([[QUANTCAST2]], [[QUANTCAST2]]) {
        // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
        // CHECK-SAME:  } : tensor<1x128x32x32x!qElemType3, {order = #NHWC}>, tensor<1x128x32x32x!qElemType3, {order = #NHWC}> -> tensor<1x128x32x32xf16>

        // CHECK: return [[ADD2]]
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

// CHECK-LABEL: @ReduceMax
module @ReduceMax {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x8x24x24xf16>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1x24x24xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x8x24x24xf16>) -> tensor<1x1x24x24xf16> {
    func.func @main(%arg0: tensor<1x8x24x24xf16>) -> tensor<1x1x24x24xf16> {
        %0 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x8x24x24xf16> -> tensor<1x1x24x24xf16>
        return %0 : tensor<1x1x24x24xf16>

        // CHECK:               [[RESHAPE0:%.+]] = IE.Reshape({{[^:]+}}) {shape_value = [1, 8, 36, 16]} : tensor<1x8x24x24xf16> -> tensor<1x8x36x16xf16>
        // CHECK:               [[PERMUTECAST0:%.+]] = IE.PermuteCast([[RESHAPE0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x8x36x16xf16> -> tensor<1x16x8x36xf16, {order = #NHWC}>
        // CHECK:               [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST0]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0],
        // CHECK-SAME:            rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x8x36xf16, {order = #NHWC}> -> tensor<1x16x1x36xf16, {order = #NHWC}>
        // CHECK:               [[PERMUTECAST1:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x1x36xf16, {order = #NHWC}> -> tensor<1x1x36x16xf16>
        // CHECK:               [[RESHAPE1:%.+]] = IE.Reshape([[PERMUTECAST1]]) {shape_value = [1, 1, 24, 24]} : tensor<1x1x36x16xf16> -> tensor<1x1x24x24xf16>
        // CHECK:               return [[RESHAPE1]] : tensor<1x1x24x24xf16>
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

        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
        // CHECK-SAME:       tensor<1x3x62x64xf16> -> tensor<1x16x62x64xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERM]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>

        // CHECK:       [[OUT:%.+]] = IE.Convolution([[SLICE]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        // CHECK-SAME:       tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
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

        // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<64x2x3x3xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 14, 0, 0]>]
        // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x2x1x96xf16> = dense<0.000000e+00> : tensor<1x2x1x96xf16>
        // CHECK:           [[CONCAT:%.*]] = IE.Concat([[CST_0]], %arg0) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x2x1x96xf16>, tensor<1x2x2x96xf16> -> tensor<1x2x3x96xf16>
        // CHECK:           [[PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize([[CONCAT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 14, 0, 0]} : tensor<1x2x3x96xf16> -> tensor<1x16x3x96xf16, {order = #NHWC}>
        // CHECK:           [[CONV:%.*]] = IE.Convolution([[PERMUTEQUANTIZE]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x3x96xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x2x96xf16>
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
        %cst = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<f16>]
        %0 = IE.Add(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>

        return %0 : tensor<1x16x16x32xf16>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
        // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<f16>]
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
        // CHECK [[CST_1:%.+]] = const.Declare tensor<1x3x640x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x640x1xf32>, [#const.Reorder<#NHWC>, #const.ConvertElemType<f16>]
        // CHECK [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 1] [1, 3, 640, 639] : tensor<1x3x640x640xf16, {order = #NHWC}> to tensor<1x3x640x639xf16, {order = #NHWC}>
        // CHECK [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST_0]]) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 639]]} : tensor<1x3x640x639xf16, {order = #NHWC}>, tensor<1x3x640x1xf16, {order = #NHWC}> -> tensor<1x3x640x640xf16, {order = #NHWC}>
        // CHECK [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 96, 640, 20]} inputs([[CONCAT]] : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x96x640x20xf16, {order = #NHWC}>
        // CHECK [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_IN]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x640x20xf16, {order = #NHWC}>, tensor<48x96x1x1xf16, {order = #NHWC}> -> tensor<1x48x640x20xf16, {order = #NHWC}>
        // CHECK [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 640, 320]} inputs([[CONV]] : tensor<1x48x640x20xf16, {order = #NHWC}>) -> tensor<1x3x640x320xf16, {order = #NHWC}>
        // CHECK return [[SHAPE_CAST_OUT]] : tensor<1x3x640x320xf16, {order = #NHWC}>
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

        // CHECK-DAG:       [[WEIGHTS1:%.+]] = const.Declare tensor<16x64x1x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x64x1x16xf16>, [#const.SubView<[0, 0, 0, 11], [1, 64, 1, 5]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
        // CHECK-DAG:       [[WEIGHTS0:%.+]] = const.Declare tensor<16x64x1x11xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x64x1x16xf16>, [#const.SubView<[0, 0, 0, 0], [1, 64, 1, 11]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
        // CHECK-DAG:       [[PAD_VAL0:%.+]] = const.Declare tensor<1x64x1x15xf16> = dense<0.000000e+00> : tensor<1x64x1x15xf32>, [#const.ConvertElemType<f16>]
        // CHECK-DAG:       [[PAD_VAL1:%.+]] = const.Declare tensor<1x64x1x12xf16> = dense<0.000000e+00> : tensor<1x64x1x12xf32>, [#const.ConvertElemType<f16>]
        // CHECK:           [[UPSAMPLE:%.+]] = IE.Upsampling([[ARG0]]) {
        // CHECK-SAME:              pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 3]>, upsampling_factor = [4, 1, 1]
        // CHECK-SAME:          } : tensor<1x64x1x256xf16> -> tensor<1x64x1x1024xf16>
        // CHECK:           [[CONCAT:%.+]] = IE.Concat([[PAD_VAL0]], [[UPSAMPLE]], [[PAD_VAL1]]) {
        // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 0, 15], [0, 0, 0, 1039]]
        // CHECK-SAME:          } : tensor<1x64x1x15xf16>, tensor<1x64x1x1024xf16>, tensor<1x64x1x12xf16> -> tensor<1x64x1x1051xf16>

        // CHECK:           [[EXPAND:%.+]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 5]} : tensor<1x64x1x1051xf16> -> tensor<1x64x1x1056xf16>
        // CHECK:           [[PERMUTE0:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x64x1x1056xf16> -> tensor<1x64x1x1056xf16, {order = #NHWC}>
        // CHECK:           [[SLICE0:%.+]] = IE.Slice [[PERMUTE0]] [0, 0, 0, 0] [1, 64, 1, 1046] : tensor<1x64x1x1056xf16, {order = #NHWC}> to tensor<1x64x1x1046xf16, {order = #NHWC}>
        // CHECK:           [[CONV0:%.+]] = IE.Convolution([[SLICE0]], [[WEIGHTS0]]) {
        // CHECK-SAME:              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        // CHECK-SAME:          } : tensor<1x64x1x1046xf16, {order = #NHWC}>, tensor<16x64x1x11xf16, {order = #NHWC}> -> tensor<1x16x1x1036xf16, {order = #NHWC}>

        // CHECK:           [[SLICE1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 64, 1, 1040] : tensor<1x64x1x1051xf16> to tensor<1x64x1x1040xf16>
        // CHECK:           [[PERMUTE1:%.+]] = IE.PermuteQuantize([[SLICE1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x64x1x1040xf16> -> tensor<1x64x1x1040xf16, {order = #NHWC}>
        // CHECK:           [[CONV1:%.+]] = IE.Convolution([[PERMUTE1]], [[WEIGHTS1]]) {
        // CHECK-SAME:              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        // CHECK-SAME:          } : tensor<1x64x1x1040xf16, {order = #NHWC}>, tensor<16x64x1x5xf16, {order = #NHWC}> -> tensor<1x16x1x1036xf16, {order = #NHWC}>

        // CHECK:           [[ADD:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
        // CHECK-SAME:          } : tensor<1x16x1x1036xf16, {order = #NHWC}>, tensor<1x16x1x1036xf16, {order = #NHWC}> -> tensor<1x16x1x1036xf16, {order = #NHWC}>

        // CHECK:           [[SLICE_OUT:%.+]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 1, 1, 1036] : tensor<1x16x1x1036xf16, {order = #NHWC}> to tensor<1x1x1x1036xf16, {order = #NHWC}>
        // CHECK:           [[PERMUTE_OUT:%.+]] = IE.PermuteCast([[SLICE_OUT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x1036xf16, {order = #NHWC}> -> tensor<1x1x1x1036xf16>
        // CHECK:           return [[PERMUTE_OUT]] : tensor<1x1x1x1036xf16>
    }
}

// -----

// CHECK-LABEL-DAG: @MatMulWithGroupQuant
// CHECK-DAG: [[Q_TYPE:!.*]] = !quant.uniform<i8<-127:127>:f16, 0.0078740157480314959>
module @MatMulWithGroupQuant {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<16x96xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<16x64xf32>
    }

    func.func @main(%arg0: tensor<16x96xf32>) -> tensor<16x64xf32> {
        %WEIGHTS = const.Declare tensor<3x32x64xf32> = dense<1.0> : tensor<3x32x64xf32>
        // CHECK-DAG:   [[WEIGHTS_0:%.*]] = const.Declare tensor<64x32x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x32x64xf32>, [#const.SubView<[0, 0, 0], [1, 32, 64]>, #const.ConvertElemType<f16>, #const.Reshape<[1, 32, 1, 64]>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Reshape<[1, 1, 32, 64]>, #const.Transpose<#NCWH>, #const.Reshape<[64, 32, 1, 1]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_1:%.*]] = const.Declare tensor<64x32x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x32x64xf32>, [#const.SubView<[1, 0, 0], [1, 32, 64]>, #const.ConvertElemType<f16>, #const.Reshape<[1, 32, 1, 64]>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Reshape<[1, 1, 32, 64]>, #const.Transpose<#NCWH>, #const.Reshape<[64, 32, 1, 1]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_2:%.*]] = const.Declare tensor<64x32x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x32x64xf32>, [#const.SubView<[2, 0, 0], [1, 32, 64]>, #const.ConvertElemType<f16>, #const.Reshape<[1, 32, 1, 64]>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Reshape<[1, 1, 32, 64]>, #const.Transpose<#NCWH>, #const.Reshape<[64, 32, 1, 1]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_LHS:%.*]] = IE.AffineReshape(%arg0) {
        // CHECK-SAME:      shape_value = [1, 1, 16, 96]
        // CHECK-SAME:  } : tensor<16x96xf32> -> tensor<1x1x16x96xf32>
        // CHECK:   [[CONVERT_LHS:%.*]] = IE.Convert([[RESHAPE_LHS]]) {
        // CHECK-SAME:      dstElemType = f16
        // CHECK-SAME:  } : tensor<1x1x16x96xf32> -> tensor<1x1x16x96xf16>

        %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
        %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
        %OUT_LOW = const.Declare tensor<3x1x64xf32> = dense<-1.0> : tensor<3x1x64xf32>
        %OUT_HIGH = const.Declare tensor<3x1x64xf32> = dense<1.0> : tensor<3x1x64xf32>

        %FQ = IE.FakeQuantize(%WEIGHTS, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 255 : i64
        } : tensor<3x32x64xf32>,
            tensor<1x1x1xf32>,
            tensor<1x1x1xf32>,
            tensor<3x1x64xf32>,
            tensor<3x1x64xf32>
                -> tensor<3x32x64xf32>

        %SHAPE_CST = const.Declare tensor<2xsi64> = dense<[96, 64]> : tensor<2xsi64>
        %RESHAPE = IE.Reshape(%FQ, %SHAPE_CST) : tensor<3x32x64xf32>, tensor<2xsi64> -> tensor<96x64xf32>
        %GEMM = IE.MatMul(%arg0, %RESHAPE) : tensor<16x96xf32>, tensor<96x64xf32> -> tensor<16x64xf32>
        // CHECK:   [[SLICE_0:%.*]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x1x16x96xf16> to tensor<1x1x16x32xf16>
        // CHECK:   [[SLICE_1:%.*]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 32] [1, 1, 16, 32] : tensor<1x1x16x96xf16> to tensor<1x1x16x32xf16>
        // CHECK:   [[SLICE_2:%.*]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 64] [1, 1, 16, 32] : tensor<1x1x16x96xf16> to tensor<1x1x16x32xf16>

        // CHECK:   [[RESHAPE_SLICE_0:%.*]] = IE.AffineReshape([[SLICE_0]]) {
        // CHECK-SAME:      shape_value = [16, 32, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x32xf16> -> tensor<16x32x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_0:%.*]] = IE.PermuteCast([[RESHAPE_SLICE_0]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x32x1x1xf16> -> tensor<1x32x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_0:%.*]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_0]]) {
        // CHECK-SAME:      shape_value = [1, 32, 4, 4]
        // CHECK-SAME:  } : tensor<1x32x16x1xf16, {order = #NHWC}> -> tensor<1x32x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_0:%.*]] = IE.Convolution([[CONV_INPUT_0]], [[WEIGHTS_0]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<64x32x1x1x!qElemType, {order = #NHWC}> -> tensor<1x64x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_0:%.*]] = IE.AffineReshape([[CONV_0]]) {
        // CHECK-SAME:      shape_value = [1, 64, 16, 1]
        // CHECK-SAME:  } : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_SLICE_1:%.*]] = IE.AffineReshape([[SLICE_1]]) {
        // CHECK-SAME:      shape_value = [16, 32, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x32xf16> -> tensor<16x32x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_1:%.*]] = IE.PermuteCast([[RESHAPE_SLICE_1]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x32x1x1xf16> -> tensor<1x32x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_1:%.*]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_1]]) {
        // CHECK-SAME:      shape_value = [1, 32, 4, 4]
        // CHECK-SAME:  } : tensor<1x32x16x1xf16, {order = #NHWC}> -> tensor<1x32x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.*]] = IE.Convolution([[CONV_INPUT_1]], [[WEIGHTS_1]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<64x32x1x1x!qElemType, {order = #NHWC}> -> tensor<1x64x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_1:%.*]] = IE.AffineReshape([[CONV_1]]) {
        // CHECK-SAME:      shape_value = [1, 64, 16, 1]
        // CHECK-SAME:  } : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_SLICE_2:%.*]] = IE.AffineReshape([[SLICE_2]]) {
        // CHECK-SAME:      shape_value = [16, 32, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x32xf16> -> tensor<16x32x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_2:%.*]] = IE.PermuteCast([[RESHAPE_SLICE_2]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x32x1x1xf16> -> tensor<1x32x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_2:%.*]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_2]]) {
        // CHECK-SAME:      shape_value = [1, 32, 4, 4]
        // CHECK-SAME:  } : tensor<1x32x16x1xf16, {order = #NHWC}> -> tensor<1x32x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_2:%.*]] = IE.Convolution([[CONV_INPUT_2]], [[WEIGHTS_2]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<64x32x1x1x!qElemType, {order = #NHWC}> -> tensor<1x64x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_2:%.*]] = IE.AffineReshape([[CONV_2]]) {
        // CHECK-SAME:      shape_value = [1, 64, 16, 1]
        // CHECK-SAME:  } : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

        // CHECK:   [[ADD_0:%.*]] = IE.Accumulate([[RESHAPE_CONV_0]], [[RESHAPE_CONV_1]])
        // CHECK:   [[ADD_1:%.*]] = IE.Accumulate([[ADD_0]], [[RESHAPE_CONV_2]])

        // CHECK:   [[PERMUTE_CAST_OUT:%.*]] = IE.PermuteCast([[ADD_1]]) {
        // CHECK-SAME:      dst_order = #NCHW,
        // CHECK-SAME:      mem_perm = #NHCW
        // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}> -> tensor<1x1x16x64xf16>

        // CHECK:   [[CONVERT_OUT:%.*]] = IE.Convert([[PERMUTE_CAST_OUT]]) {
        // CHECK-SAME:      dstElemType = f32
        // CHECK-SAME:  } : tensor<1x1x16x64xf16> -> tensor<1x1x16x64xf32>

        // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[CONVERT_OUT]]) {
        // CHECK-SAME:      shape_value = [16, 64]
        // CHECK-SAME:  } : tensor<1x1x16x64xf32> -> tensor<16x64xf32>

        return %GEMM : tensor<16x64xf32>
        // CHECK:   return [[RESHAPE_OUT]] : tensor<16x64xf32>
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

        // CHECK:   [[PERMUTE_WEIGHT:%.+]] = IE.MemPermute([[ARG1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<128x64x1x22xf16> -> tensor<128x64x1x22xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_0:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 0] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_1:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 11] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_2:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 0] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_3:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 11] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTE_IN:%.+]] = IE.PermuteQuantize([[ARG0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x128x1x112xf16> -> tensor<1x128x1x112xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_3:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 11] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_2:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 0] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_1:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 11] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_0:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 0] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>

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
        // CHECK:   [[PERMUTE_OUT:%.+]] = IE.MaxPool([[CONCAT]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x1x91xf16, {order = #NHWC}> -> tensor<1x128x1x91xf16>
        // CHECK:   return [[PERMUTE_OUT]] : tensor<1x128x1x91xf16>
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

        // CHECK-DAG:       [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1600x1x1x1xf16, {order = #NHWC}> = dense<2.000000e+00>
        // CHECK-SAME           : tensor<1x1x80x80xf16>, [#const.Reshape<[1, 6400, 1, 1]>, #const.Reshape<[6400, 1, 1, 1]>, #const.SubView<[0, 0, 0, 0], [1600, 1, 1, 1]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
        // CHECK-SAME:      shape_value = [1, 1, 19, 6400]
        // CHECK-SAME:  } : tensor<1x19x80x80xf16> -> tensor<1x1x19x6400xf16>

        // CHECK:   [[PERMUTE_INPUT:%.*]] = IE.PermuteCast([[RESHAPE_INPUT]]) {
        // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
        // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x6400x1x19xf16, {order = #NHWC}>

        // CHECK:   [[SHAPECAST_IN:%.*]] = IE.ShapeCast {shape = [1, 1600, 4, 19]} inputs([[PERMUTE_INPUT]] : tensor<1x6400x1x19xf16, {order = #NHWC}>) -> tensor<1x1600x4x19xf16, {order = #NHWC}>

        // CHECK:   [[MUL:%.*]] = IE.GroupConvolution([[SHAPECAST_IN]], [[MUL_WEIGHTS]]) {
        // CHECK-SAME:      dilations = [1, 1], groups = 1600 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1600x4x19xf16, {order = #NHWC}>, tensor<1600x1x1x1xf16, {order = #NHWC}> -> tensor<1x1600x4x19xf16, {order = #NHWC}>

        // CHECK:   [[SHAPECAST_OUT:%.*]] = IE.ShapeCast {shape = [1, 6400, 1, 19]} inputs([[MUL]] : tensor<1x1600x4x19xf16, {order = #NHWC}>) -> tensor<1x6400x1x19xf16, {order = #NHWC}>

        // CHECK:   [[PERMUTE_OUT:%.*]] = IE.PermuteCast([[SHAPECAST_OUT]]) {
        // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
        // CHECK-SAME:  } : tensor<1x6400x1x19xf16, {order = #NHWC}> -> tensor<1x1x19x6400xf16>

        // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[PERMUTE_OUT]]) {
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
