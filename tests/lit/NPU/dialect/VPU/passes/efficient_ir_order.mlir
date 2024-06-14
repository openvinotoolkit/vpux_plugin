//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --efficient-ir-order %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @EfficientEltwiseOrder(%arg0: tensor<1x96x24x40xf16, {order = #NHWC}>) -> tensor<1x96x24x40xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    
    %0 = VPU.NCE.Convolution(%arg0, %cst_1, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} -> tensor<1x96x24x40xf16, {order = #NHWC}> 
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} -> tensor<1x96x24x40xf16, {order = #NHWC}> 
    %2 = VPU.NCE.Convolution(%1, %cst_1, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} -> tensor<1x96x24x40xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %cst_1, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 96, 3, 3], strides = [2, 2]} -> tensor<1x96x12x20xf16, {order = #NHWC}>
    %4 = VPU.Concat(%3, %3) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x96x12x20xf16, {order = #NHWC}>, tensor<1x96x12x20xf16, {order = #NHWC}> -> tensor<1x96x24x20xf16, {order = #NHWC}>
    %5 = VPU.Concat(%4, %4) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x96x24x20xf16, {order = #NHWC}>, tensor<1x96x24x20xf16, {order = #NHWC}> -> tensor<1x96x24x40xf16, {order = #NHWC}>

    %6 = VPU.NCE.Eltwise(%2, %5) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [23564], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = [19364], in2_quant_mult = [3693], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x96x24x40xf16, {order = #NHWC}>
    return %6: tensor<1x96x24x40xf16, {order = #NHWC}>


    //CHECK:  [[CONV_0:%.+]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst)  
    //CHECK:  [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_0]], %cst_0, %cst) 
    //CHECK:  [[CONCAT_0:%.+]] = VPU.Concat([[CONV_1]], [[CONV_1]])
    //CHECK:  [[CONCAT_1:%.+]] = VPU.Concat([[CONCAT_0]], [[CONCAT_0]])
    //CHECK:  [[CONV_2:%.+]] = VPU.NCE.Convolution([[CONV_0]], %cst_0, %cst)
    //CHECK:  [[CONV_3:%.+]] = VPU.NCE.Convolution([[CONV_2]], %cst_0, %cst) 
    //CHECK:  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV_3]], [[CONCAT_1]]) 
    //CHECK:  return [[ELTWISE]] : tensor<1x96x24x40xf16, {order = #NHWC}>
}
