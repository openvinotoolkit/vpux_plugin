//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCE
func.func @DepthConvToNCE(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [1, 1],
            groups = 16,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.1}>
        } : tensor<1x16x40x80xf16, {order = #NHWC}>, tensor<16x1x4x8xf16, {order = #NHWC}>
            -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK-DAG:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]] )
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 102 : i64, lrelu_shift = 10 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 1, 4, 8], strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddWithReluRewriter
func.func @EltwiseAddWithReluRewriter(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1)
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <ADD>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16384], quant_shift = [14], quant_post_shift = 0 : i64>}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAndSameInputsToNCE
func.func @EltwiseAndSameInputsToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.And(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise(%arg0, %arg0)
    // CHECK-SAME:      op_type = #VPU.eltwise_type<AND>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeIncompatibleArch
func.func @PermuteQuantizeIncompatibleArch(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Permute
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 1, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @CMConvToNCE
func.func @CMConvToNCE(%arg0: tensor<1x4x96x160xf16>)
                        -> tensor<1x16x48x80xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x4x7x7xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x4x7x7xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %0 = IE.Convolution(%arg0, %weights, %bias) {
        dilations = [1, 1],
        pads_begin = [2, 2], pads_end = [3, 3],
        strides = [2, 2]
    } : tensor<1x4x96x160xf16>,
        tensor<16x4x7x7xf16, {order = #NHWC}>,
        tensor<1x16x1x1xf16>
        -> tensor<1x16x48x80xf16, {order = #NHWC}>

    return %0 : tensor<1x16x48x80xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.CompressConv

    // CHECK:       [[CM_CONV:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:      activation_window_channel_length = 364 : i64,
    // CHECK-SAME:      pad = #VPU.Padding<left = 2 : i64, right = 3 : i64, top = 2 : i64, bottom = 3 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 4, 7, 7], strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x16x48x80xf16, {order = #NHWC}>

    // CHECK:       return [[CM_CONV]] : tensor<1x16x48x80xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCE
func.func @MaxPoolToNCE(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.0, min = 0.0}>
        } : tensor<1x16x1x4xf16, {order = #NHWC}> -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.MaxPool(%arg0, [[WEIGHTS_TABLE]] , [[ACTIVATION_WINDOW]] )
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 393216 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}
