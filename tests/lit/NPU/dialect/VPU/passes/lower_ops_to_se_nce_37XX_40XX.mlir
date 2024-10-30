//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0257227579752604:128>
!qElemType2 = !quant.uniform<u8:f16, 2.500000e-01>

// CHECK:   !qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
// CHECK:   !qElemType1 = !quant.uniform<u8:f16, 0.025722757975260399:128>
// CHECK:   !qElemType2 = !quant.uniform<u8:f16, 2.500000e-01>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearQuantized([[INPUT_DATA:%.+]]: tensor<1x16x3x3x!qElemType, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType1, {order = #NHWC}> {
func.func @InterpolateBilinearQuantized(%arg0: tensor<1x16x3x3x!qElemType, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType1, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<1x16x3x3x!qElemType, {order = #NHWC}> -> tensor<1x16x6x6x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x16x6x6x!qElemType1, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x7x7xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x7x7xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x7x7xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:       scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x7x7xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x7x7xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2x!qElemType2, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x2x2xf32>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1267142912, 0]]], [[[64, 0, 1267142912, 0]]], [[[128, 0, 1267142912, 0]]], [[[192, 0, 1267142912, 0]]],
    // CHECK-SAME{LITERAL}:          [[[256, 0, 1267142912, 0]]], [[[320, 0, 1267142912, 0]]], [[[384, 0, 1267142912, 0]]], [[[448, 0, 1267142912, 0]]],
    // CHECK-SAME{LITERAL}:          [[[512, 0, 1267142912, 0]]], [[[576, 0, 1267142912, 0]]], [[[640, 0, 1267142912, 0]]], [[[704, 0, 1267142912, 0]]],
    // CHECK-SAME{LITERAL}:          [[[768, 0, 1267142912, 0]]], [[[832, 0, 1267142912, 0]]], [[[896, 0, 1267142912, 0]]], [[[960, 0, 1267142912, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       opaque_ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
    // CHECK-SAME:       rawFilterShape = [16, 16, 2, 2],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x6x6x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolution([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x46x60xf16, {order = #NHWC}> {
func.func @TransposedConvolution(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x46x60xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x46x60xf16, {order = #NHWC}>
    return %output : tensor<1x16x46x60xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 32 : i64
    // CHECK-SAME:  } -> tensor<1x1x47x61xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x47x61xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x47x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x47x61xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x47x61xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      opaque_ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x46x60xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0257227579752604:128>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.025722757975260399:128>

// CHECK: func.func @TransposedConvolutionQuantized([[INPUT_DATA:%.+]]: tensor<1x32x23x30x!qElemType, {order = #NHWC}>) -> tensor<1x16x46x60x!qElemType, {order = #NHWC}> {
func.func @TransposedConvolutionQuantized(%input: tensor<1x32x23x30x!qElemType, {order = #NHWC}>) -> tensor<1x16x46x60x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2x!qElemType1, {order = #NHWC}> = dense<1> : tensor<16x32x2x2xui8, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30x!qElemType, {order = #NHWC}>, tensor<16x32x2x2x!qElemType1, {order = #NHWC}> -> tensor<1x16x46x60x!qElemType, {order = #NHWC}>
    return %output : tensor<1x16x46x60x!qElemType, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2x!qElemType1, {order = #NHWC}> = dense<1> : tensor<16x32x2x2xui8, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 32 : i64
    // CHECK-SAME:  } -> tensor<1x1x47x61xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x47x61xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x47x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x47x61xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x47x61xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1767642112, 0]]], [[[128, 0, 1767642112, 0]]], [[[256, 0, 1767642112, 0]]], [[[384, 0, 1767642112, 0]]],
    // CHECK-SAME{LITERAL}:         [[[512, 0, 1767642112, 0]]], [[[640, 0, 1767642112, 0]]], [[[768, 0, 1767642112, 0]]], [[[896, 0, 1767642112, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1767642112, 0]]], [[[1152, 0, 1767642112, 0]]], [[[1280, 0, 1767642112, 0]]], [[[1408, 0, 1767642112, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1536, 0, 1767642112, 0]]], [[[1664, 0, 1767642112, 0]]], [[[1792, 0, 1767642112, 0]]], [[[1920, 0, 1767642112, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      opaque_ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x46x60x!qElemType, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionWithPostOp([[INPUT_DATA:%.+]]: tensor<1x32x64x1xf16, {order = #NHWC}>) -> tensor<1x16x128x2xf16, {order = #NHWC}> {
func.func @TransposedConvolutionWithPostOp(%input: tensor<1x32x64x1xf16, {order = #NHWC}>) -> tensor<1x16x128x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x3x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [1, 0], pads_begin = [1, 0], pads_end = [1, 0],
            post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 2.500000e-01 : f64}>, strides = [2, 1]
        } : tensor<1x32x64x1xf16, {order = #NHWC}>, tensor<16x32x3x2xf16, {order = #NHWC}> -> tensor<1x16x128x2xf16, {order = #NHWC}>
    return %output : tensor<1x16x128x2xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x3x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 64, 1],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 0], padding = [1, 1, 1, 2]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 32 : i64
    // CHECK-SAME:  } -> tensor<1x1x130x3xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x130x3xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x130x3xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 0], padding = [1, 1, 1, 2]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x64x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x130x3xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x130x3xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 0], padding = [1, 1, 1, 2]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[384, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]], [[[1152, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1536, 0, 1065353216, 0]]], [[[1920, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2688, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3456, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]], [[[4224, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[4608, 0, 1065353216, 0]]], [[[4992, 0, 1065353216, 0]]], [[[5376, 0, 1065353216, 0]]], [[[5760, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      opaque_ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:      lrelu_mult = 1024 : i64, lrelu_shift = 12 : i64, fp_prelu_alpha = 2.500000e-01 : f64>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 3, 2], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x128x2xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionWithBias([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x46x60xf16, {order = #NHWC}> {
func.func @TransposedConvolutionWithBias(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x46x60xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x1x1xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights, %bias) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x46x60xf16, {order = #NHWC}>
    return %output : tensor<1x16x46x60xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    // CHECK:       [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 32 : i64
    // CHECK-SAME:  } -> tensor<1x1x47x61xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x47x61xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x47x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x47x61xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x47x61xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 1065353216]]], [[[256, 0, 1065353216, 1065353216]]], [[[512, 0, 1065353216, 1065353216]]], [[[768, 0, 1065353216, 1065353216]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 1065353216]]], [[[1280, 0, 1065353216, 1065353216]]], [[[1536, 0, 1065353216, 1065353216]]], [[[1792, 0, 1065353216, 1065353216]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 1065353216]]], [[[2304, 0, 1065353216, 1065353216]]], [[[2560, 0, 1065353216, 1065353216]]], [[[2816, 0, 1065353216, 1065353216]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 1065353216]]], [[[3328, 0, 1065353216, 1065353216]]], [[[3584, 0, 1065353216, 1065353216]]], [[[3840, 0, 1065353216, 1065353216]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      opaque_ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x46x60xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}
