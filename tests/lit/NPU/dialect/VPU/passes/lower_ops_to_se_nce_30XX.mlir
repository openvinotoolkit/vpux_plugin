//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateNearestScaleCalcModeAsymmetric([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @InterpolateNearestScaleCalcModeAsymmetric(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <NEAREST>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x6x6xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x6x6xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x6x6xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x6x6xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x6x6xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                            nearest_mode = <FLOOR>>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073761793, 0]]], [[[32, 0, 1073761793, 0]]], [[[64, 0, 1073761793, 0]]], [[[96, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[128, 0, 1073761793, 0]]], [[[160, 0, 1073761793, 0]]], [[[192, 0, 1073761793, 0]]], [[[224, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[256, 0, 1073761793, 0]]], [[[288, 0, 1073761793, 0]]], [[[320, 0, 1073761793, 0]]], [[[352, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[384, 0, 1073761793, 0]]], [[[416, 0, 1073761793, 0]]], [[[448, 0, 1073761793, 0]]], [[[480, 0, 1073761793, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:       rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateNearestSizesCalcModeAsymmetric([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @InterpolateNearestSizesCalcModeAsymmetric(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <NEAREST>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3],
            scales_attr = [2.300000e+00, 1.500000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x6x6xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x6x6xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x6x6xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x6x6xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x6x6xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                            nearest_mode = <FLOOR>>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073761793, 0]]], [[[32, 0, 1073761793, 0]]], [[[64, 0, 1073761793, 0]]], [[[96, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[128, 0, 1073761793, 0]]], [[[160, 0, 1073761793, 0]]], [[[192, 0, 1073761793, 0]]], [[[224, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[256, 0, 1073761793, 0]]], [[[288, 0, 1073761793, 0]]], [[[320, 0, 1073761793, 0]]], [[[352, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[384, 0, 1073761793, 0]]], [[[416, 0, 1073761793, 0]]], [[[448, 0, 1073761793, 0]]], [[[480, 0, 1073761793, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:       rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @InterpolateNearestAsymmetricLargeChannels
// CHECK-SAME:     ([[INPUT_DATA:%.+]]: tensor<1x144x3x3xf16, {order = #NHWC}>)
func.func @InterpolateNearestAsymmetricLargeChannels(%input: tensor<1x144x3x3xf16, {order = #NHWC}>) -> tensor<1x144x6x6xf16, {order = #NHWC}> {
    %output = VPU.Interpolate(%input) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <NEAREST>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<1x144x3x3xf16, {order = #NHWC}> -> tensor<1x144x6x6xf16, {order = #NHWC}>

    return %output : tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 144, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>,
    // CHECK-SAME:      seDepth = 9 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x9x6x6xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x144x6x6xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x144x6x6xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x144x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x144x6x6xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x9x6x6xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                            nearest_mode = <FLOOR>>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<144x144x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<144x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:       rawFilterShape = [144, 144, 1, 1],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x144x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearAsymmetric([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @InterpolateBilinearAsymmetric(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
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
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x7x7xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x7x7xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x7x7xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x7x7xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x7x7xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x2x2xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073761793, 0]]], [[[128, 0, 1073761793, 0]]], [[[256, 0, 1073761793, 0]]], [[[384, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[512, 0, 1073761793, 0]]], [[[640, 0, 1073761793, 0]]], [[[768, 0, 1073761793, 0]]], [[[896, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[1024, 0, 1073761793, 0]]], [[[1152, 0, 1073761793, 0]]], [[[1280, 0, 1073761793, 0]]], [[[1408, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[1536, 0, 1073761793, 0]]], [[[1664, 0, 1073761793, 0]]], [[[1792, 0, 1073761793, 0]]], [[[1920, 0, 1073761793, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [16, 16, 2, 2],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearHalfPixelWithEvenScale([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @InterpolateBilinearHalfPixelWithEvenScale(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <HALF_PIXEL>,
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
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x14x14xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x14x14xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x14x14xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x14x14xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x14x14xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x4x4xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073761793, 0]]], [[[512, 0, 1073761793, 0]]], [[[1024, 0, 1073761793, 0]]], [[[1536, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[2048, 0, 1073761793, 0]]], [[[2560, 0, 1073761793, 0]]], [[[3072, 0, 1073761793, 0]]], [[[3584, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[4096, 0, 1073761793, 0]]], [[[4608, 0, 1073761793, 0]]], [[[5120, 0, 1073761793, 0]]], [[[5632, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[6144, 0, 1073761793, 0]]], [[[6656, 0, 1073761793, 0]]], [[[7168, 0, 1073761793, 0]]], [[[7680, 0, 1073761793, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [16, 16, 4, 4],
    // CHECK-SAME:       strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearHalfPixelWithOddScale([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x9x9xf16, {order = #NHWC}> {
func.func @InterpolateBilinearHalfPixelWithOddScale(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x9x9xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <HALF_PIXEL>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [3.000000e+00, 3.000000e+00],
            sizes_attr = [9, 9],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x9x9xf16, {order = #NHWC}>

    return %0 : tensor<1x16x9x9xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x11x11xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x11x11xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x11x11xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x11x11xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x11x11xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073761793, 0]]], [[[288, 0, 1073761793, 0]]], [[[576, 0, 1073761793, 0]]], [[[864, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[1152, 0, 1073761793, 0]]], [[[1440, 0, 1073761793, 0]]], [[[1728, 0, 1073761793, 0]]], [[[2016, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[2304, 0, 1073761793, 0]]], [[[2592, 0, 1073761793, 0]]], [[[2880, 0, 1073761793, 0]]], [[[3168, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[3456, 0, 1073761793, 0]]], [[[3744, 0, 1073761793, 0]]], [[[4032, 0, 1073761793, 0]]], [[[4320, 0, 1073761793, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [16, 16, 3, 3],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x9x9xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearPytorchHalfPixelWithEvenScale([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @InterpolateBilinearPytorchHalfPixelWithEvenScale(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <PYTORCH_HALF_PIXEL>,
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
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x14x14xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x14x14xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x14x14xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x14x14xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x14x14xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x4x4xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073761793, 0]]], [[[512, 0, 1073761793, 0]]], [[[1024, 0, 1073761793, 0]]], [[[1536, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[2048, 0, 1073761793, 0]]], [[[2560, 0, 1073761793, 0]]], [[[3072, 0, 1073761793, 0]]], [[[3584, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[4096, 0, 1073761793, 0]]], [[[4608, 0, 1073761793, 0]]], [[[5120, 0, 1073761793, 0]]], [[[5632, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[6144, 0, 1073761793, 0]]], [[[6656, 0, 1073761793, 0]]], [[[7168, 0, 1073761793, 0]]], [[[7680, 0, 1073761793, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [16, 16, 4, 4],
    // CHECK-SAME:       strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearPytorchHalfPixelWithOddScale([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x9x9xf16, {order = #NHWC}> {
func.func @InterpolateBilinearPytorchHalfPixelWithOddScale(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x9x9xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <PYTORCH_HALF_PIXEL>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [3.000000e+00, 3.000000e+00],
            sizes_attr = [9, 9],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x9x9xf16, {order = #NHWC}>

    return %0 : tensor<1x16x9x9xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x11x11xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x11x11xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x11x11xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x11x11xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x11x11xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073761793, 0]]], [[[288, 0, 1073761793, 0]]], [[[576, 0, 1073761793, 0]]], [[[864, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[1152, 0, 1073761793, 0]]], [[[1440, 0, 1073761793, 0]]], [[[1728, 0, 1073761793, 0]]], [[[2016, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[2304, 0, 1073761793, 0]]], [[[2592, 0, 1073761793, 0]]], [[[2880, 0, 1073761793, 0]]], [[[3168, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[3456, 0, 1073761793, 0]]], [[[3744, 0, 1073761793, 0]]], [[[4032, 0, 1073761793, 0]]], [[[4320, 0, 1073761793, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [16, 16, 3, 3],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x9x9xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>
// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateNearestQuantized([[INPUT_DATA:%.+]]: tensor<1x16x3x3x!qElemType, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType, {order = #NHWC}> {
func.func @InterpolateNearestQuantized(%arg0: tensor<1x16x3x3x!qElemType, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <NEAREST>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<1x16x3x3x!qElemType, {order = #NHWC}> -> tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x6x6xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x6x6xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x6x6xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x6x6xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x6x6xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073761793, 0]]], [[[16, 0, 1073761793, 0]]], [[[32, 0, 1073761793, 0]]], [[[48, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[64, 0, 1073761793, 0]]], [[[80, 0, 1073761793, 0]]], [[[96, 0, 1073761793, 0]]], [[[112, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[128, 0, 1073761793, 0]]], [[[144, 0, 1073761793, 0]]], [[[160, 0, 1073761793, 0]]], [[[176, 0, 1073761793, 0]]],
    // CHECK-SAME{LITERAL}:          [[[192, 0, 1073761793, 0]]], [[[208, 0, 1073761793, 0]]], [[[224, 0, 1073761793, 0]]], [[[240, 0, 1073761793, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:       rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>
// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 2.500000e-01>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearQuantized([[INPUT_DATA:%.+]]: tensor<1x16x3x3x!qElemType, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType, {order = #NHWC}> {
func.func @InterpolateBilinearQuantized(%arg0: tensor<1x16x3x3x!qElemType, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType, {order = #NHWC}> {
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
        } : tensor<1x16x3x3x!qElemType, {order = #NHWC}> -> tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x7x7xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x7x7xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x7x7xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x7x7xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x7x7xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2x!qElemType1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x2x2xf32>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073762305, 0]]], [[[64, 0, 1073762305, 0]]], [[[128, 0, 1073762305, 0]]], [[[192, 0, 1073762305, 0]]],
    // CHECK-SAME{LITERAL}:          [[[256, 0, 1073762305, 0]]], [[[320, 0, 1073762305, 0]]], [[[384, 0, 1073762305, 0]]], [[[448, 0, 1073762305, 0]]],
    // CHECK-SAME{LITERAL}:          [[[512, 0, 1073762305, 0]]], [[[576, 0, 1073762305, 0]]], [[[640, 0, 1073762305, 0]]], [[[704, 0, 1073762305, 0]]],
    // CHECK-SAME{LITERAL}:          [[[768, 0, 1073762305, 0]]], [[[832, 0, 1073762305, 0]]], [[[896, 0, 1073762305, 0]]], [[[960, 0, 1073762305, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [16, 16, 2, 2],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}
