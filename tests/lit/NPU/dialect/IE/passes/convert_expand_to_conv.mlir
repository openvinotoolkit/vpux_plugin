//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-expand-to-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertExpandToConv16Channels
func.func @ConvertExpandToConv16Channels(%arg0: tensor<1x3x64x224xf16, {order = #NHWC}>)
    -> tensor<1x16x64x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x16x64x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<256x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      003C00000000{{([0]{180})}}
    // CHECK-SAME:      0000003C0000{{([0]{180})}}
    // CHECK-SAME:      00000000003C{{([0]{180})}}
    // CHECK-SAME:      000000000000{{([0]{180})}}

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 64, 14]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x48x64x14xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x64x14xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x64x14xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 64, 224]
    // CHECK-SAME:  } : tensor<1x256x64x14xf16, {order = #NHWC}> -> tensor<1x16x64x224xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x64x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertExpandToConv4Channels
func.func @ConvertExpandToConv4Channels(%arg0: tensor<1x3x64x224xf16, {order = #NHWC}>)
    -> tensor<1x4x64x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x4x64x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x4x64x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      003C00000000000000000000{{([0]{168})}}
    // CHECK-SAME:      0000003C0000000000000000{{([0]{168})}}
    // CHECK-SAME:      00000000003C000000000000{{([0]{168})}}
    // CHECK-SAME:      000000000000000000000000{{([0]{168})}}
    // CHECK-SAME:      000000000000003C00000000{{([0]{168})}}
    // CHECK-SAME:      0000000000000000003C0000{{([0]{168})}}
    // CHECK-SAME:      00000000000000000000003C{{([0]{168})}}
    // CHECK-SAME:      000000000000000000000000{{([0]{168})}}

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 64, 14]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x48x64x14xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x64x14xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x64x14xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 64, 224]
    // CHECK-SAME:  } : tensor<1x64x64x14xf16, {order = #NHWC}> -> tensor<1x4x64x224xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x64x224xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SkipExpandNCHW
func.func @SkipExpandNCHW(%arg0: tensor<1x3x64x224xf16>) -> tensor<1x16x64x224xf16> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x224xf16> -> tensor<1x16x64x224xf16>

    return %EXPAND : tensor<1x16x64x224xf16>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16> -> tensor<1x16x64x224xf16>

    // CHECK:   return [[EXPAND]] : tensor<1x16x64x224xf16>
}

// -----

// CHECK-LABEL: @SkipExpand3d
func.func @SkipExpand3d(%arg0: tensor<1x3x64xf16>) -> tensor<1x16x64xf16> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0],
        pads_end = [0, 13, 0]
    } : tensor<1x3x64xf16> -> tensor<1x16x64xf16>

    return %EXPAND : tensor<1x16x64xf16>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0]
    // CHECK-SAME:  } : tensor<1x3x64xf16> -> tensor<1x16x64xf16>

    // CHECK:   return [[EXPAND]] : tensor<1x16x64xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipNonZeroPadsBegin
func.func @SkipNonZeroPadsBegin(%arg0: tensor<1x3x64x224xf16, {order = #NHWC}>)
    -> tensor<1x16x64x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 1, 0, 0],
        pads_end = [0, 12, 0, 0]
    } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x16x64x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 1, 0, 0],
    // CHECK-SAME:      pads_end = [0, 12, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x16x64x224xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x16x64x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipPaddingOverHeight
func.func @SkipPaddingOverHeight(%arg0: tensor<1x3x64x224xf16, {order = #NHWC}>)
    -> tensor<1x3x66x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 2, 0]
    } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x3x66x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x3x66x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 2, 0]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x3x66x224xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x3x66x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipBatch
func.func @SkipBatch(%arg0: tensor<2x3x64x224xf16, {order = #NHWC}>)
    -> tensor<2x16x64x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<2x3x64x224xf16, {order = #NHWC}> -> tensor<2x16x64x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<2x16x64x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0, 0]
    // CHECK-SAME:  } : tensor<2x3x64x224xf16, {order = #NHWC}> -> tensor<2x16x64x224xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<2x16x64x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipUnalignedWidth
func.func @SkipUnalignedWidth(%arg0: tensor<1x3x64x225xf16, {order = #NHWC}>)
    -> tensor<1x16x64x225xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x225xf16, {order = #NHWC}> -> tensor<1x16x64x225xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x225xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x64x225xf16, {order = #NHWC}> -> tensor<1x16x64x225xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x16x64x225xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipFloat32Expand
func.func @SkipFloat32Expand(%arg0: tensor<1x3x64x224xf32, {order = #NHWC}>)
    -> tensor<1x16x64x224xf32, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x224xf32, {order = #NHWC}> -> tensor<1x16x64x224xf32, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x224xf32, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x64x224xf32, {order = #NHWC}> -> tensor<1x16x64x224xf32, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x16x64x224xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

// CHECK-DAG: [[Q_ACT:!.*]] = !quant.uniform<u8:f16, 2.000000e+00>
// CHECK-DAG: [[Q_WEIGHTS:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_ACT and Q_WEIGHTS
func.func @ConvertQuantizedExpand(%arg0: tensor<1x3x64x224x!qElemType, {order = #NHWC}>)
    -> tensor<1x16x64x224x!qElemType, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x224x!qElemType, {order = #NHWC}> -> tensor<1x16x64x224x!qElemType, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.Expand

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<256x48x1x1x[[Q_WEIGHTS]], {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      003C00000000{{([0]{180})}}
    // CHECK-SAME:      0000003C0000{{([0]{180})}}
    // CHECK-SAME:      00000000003C{{([0]{180})}}
    // CHECK-SAME:      000000000000{{([0]{180})}}

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 64, 14]
    // CHECK-SAME:  } : tensor<1x3x64x224x[[Q_ACT]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x64x14x[[Q_ACT]], {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x64x14x[[Q_ACT]], {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1x[[Q_WEIGHTS]], {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x64x14x[[Q_ACT]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 64, 224]
    // CHECK-SAME:  } : tensor<1x256x64x14x[[Q_ACT]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x64x224x[[Q_ACT]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x64x224x[[Q_ACT]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertLargeExpand
func.func @ConvertLargeExpand(%arg0: tensor<1x3x512x896xf16, {order = #NHWC}>)
    -> tensor<1x16x512x896xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x512x896xf16, {order = #NHWC}> -> tensor<1x16x512x896xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x16x512x896xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Expand
    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 512, 56]
    // CHECK-SAME:  } : tensor<1x3x512x896xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x512x56xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x512x56xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x512x56xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 512, 896]
    // CHECK-SAME:  } : tensor<1x256x512x56xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x512x896xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x512x896xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-DAG: [[Q_TYPE:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_TYPE
func.func @FuseQuantizeProducer(%arg0: tensor<1x1x19x80xf16, {order = #NHWC}>)
    -> tensor<1x4x19x80x!qElemType1, {order = #NHWC}> {
    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 5]
    } inputs(%arg0 : tensor<1x1x19x80xf16, {order = #NHWC}>) -> tensor<1x16x19x5xf16, {order = #NHWC}>

    %ADD = IE.Add(%IN_SHAPE_CAST, %IN_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x16x19x5xf16, {order = #NHWC}>,
        tensor<1x16x19x5xf16, {order = #NHWC}>
            -> tensor<1x16x19x5x!qElemType, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 1, 19, 80]
    } inputs(%ADD : tensor<1x16x19x5x!qElemType, {order = #NHWC}>) -> tensor<1x1x19x80x!qElemType, {order = #NHWC}>

    %QUANT_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {
        dstElemType = !qElemType1
    } : tensor<1x1x19x80x!qElemType, {order = #NHWC}> -> tensor<1x1x19x80x!qElemType1, {order = #NHWC}>

    %EXPAND = IE.Expand(%QUANT_CAST) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x1x19x80x!qElemType1, {order = #NHWC}> -> tensor<1x4x19x80x!qElemType1, {order = #NHWC}>

    return %EXPAND : tensor<1x4x19x80x!qElemType1, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 19, 5]
    // CHECK-SAME:  } : tensor<1x1x19x80xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x19x5xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x19x5xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x19x5x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 19, 80]
    // CHECK-SAME:  } : tensor<1x64x19x5x[[Q_TYPE]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-DAG: [[Q_ACT_TYPE:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_ACT_TYPE
func.func @FuseQuantizeWithoutShapeCast(%arg0: tensor<1x1x19x80xf16, {order = #NHWC}>)
    -> tensor<1x4x19x80x!qElemType1, {order = #NHWC}> {
    %ADD = IE.Add(%arg0, %arg0) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x1x19x80xf16, {order = #NHWC}>,
        tensor<1x1x19x80xf16, {order = #NHWC}>
            -> tensor<1x1x19x80x!qElemType, {order = #NHWC}>

    %QUANT_CAST = IE.QuantizeCast(%ADD) {
        dstElemType = !qElemType1
    } : tensor<1x1x19x80x!qElemType, {order = #NHWC}> -> tensor<1x1x19x80x!qElemType1, {order = #NHWC}>

    %EXPAND = IE.Expand(%QUANT_CAST) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x1x19x80x!qElemType1, {order = #NHWC}> -> tensor<1x4x19x80x!qElemType1, {order = #NHWC}>

    return %EXPAND : tensor<1x4x19x80x!qElemType1, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 19, 5]
    // CHECK-SAME:  } : tensor<1x1x19x80xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x19x5xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x19x5xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x19x5x[[Q_ACT_TYPE]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 19, 80]
    // CHECK-SAME:  } : tensor<1x64x19x5x[[Q_ACT_TYPE]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x19x80x[[Q_ACT_TYPE]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x19x80x[[Q_ACT_TYPE]], {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-DAG: [[Q_ACT_TYPE:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_ACT_TYPE
func.func @FuseQuantizeWithAvgPool(%arg0: tensor<1x1x19x80xf16, {order = #NHWC}>)
    -> tensor<1x4x19x80x!qElemType, {order = #NHWC}> {
    %AvgPool = IE.AvgPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    } : tensor<1x1x19x80xf16, {order = #NHWC}> -> tensor<1x1x19x80x!qElemType, {order = #NHWC}>

    %EXPAND = IE.Expand(%AvgPool) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x1x19x80x!qElemType, {order = #NHWC}> -> tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    return %EXPAND : tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 19, 5]
    // CHECK-SAME:  } : tensor<1x1x19x80xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x19x5xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x19x5xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x19x5x[[Q_ACT_TYPE]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 19, 80]
    // CHECK-SAME:  } : tensor<1x64x19x5x[[Q_ACT_TYPE]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x19x80x[[Q_ACT_TYPE]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x19x80x[[Q_ACT_TYPE]], {order = #NHWC}>
}



// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-DAG: [[Q_TYPE:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_TYPE
func.func @FuseQuantizeWithShapeCastAvgPool(%arg0: tensor<1x1x19x80xf16, {order = #NHWC}>)
    -> tensor<1x4x19x80x!qElemType, {order = #NHWC}> {
    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 5]
    } inputs(%arg0 : tensor<1x1x19x80xf16, {order = #NHWC}>) -> tensor<1x16x19x5xf16, {order = #NHWC}>

    %AvgPool = IE.AvgPool(%IN_SHAPE_CAST) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    } : tensor<1x16x19x5xf16, {order = #NHWC}> -> tensor<1x16x19x5x!qElemType, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 1, 19, 80]
    } inputs(%AvgPool : tensor<1x16x19x5x!qElemType, {order = #NHWC}>) -> tensor<1x1x19x80x!qElemType, {order = #NHWC}>

    %EXPAND = IE.Expand(%OUT_SHAPE_CAST) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x1x19x80x!qElemType, {order = #NHWC}> -> tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    return %EXPAND : tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 19, 5]
    // CHECK-SAME:  } : tensor<1x1x19x80xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x19x5xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x19x5xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x19x5x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 19, 80]
    // CHECK-SAME:  } : tensor<1x64x19x5x[[Q_TYPE]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipLargeKernelAndSmallSpatialSize
func.func @SkipLargeKernelAndSmallSpatialSize(%arg0: tensor<1x387x45x80xf16, {order = #NHWC}>)
    -> tensor<1x400x45x80xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x387x45x80xf16, {order = #NHWC}> -> tensor<1x400x45x80xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x400x45x80xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0, 0]
    // CHECK-SAME:  } : tensor<1x387x45x80xf16, {order = #NHWC}> -> tensor<1x400x45x80xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x400x45x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertWithBigChannelAndHeightRatio
func.func @NotConvertWithBigChannelAndHeightRatio(%arg0: tensor<1x60x80x80xf16, {order = #NHWC}>)
    -> tensor<1x64x80x80xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 4, 0, 0]
    } : tensor<1x60x80x80xf16, {order = #NHWC}> -> tensor<1x64x80x80xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x64x80x80xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand({{[^:]+}}) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 4, 0, 0]
    // CHECK-SAME:  } : tensor<1x60x80x80xf16, {order = #NHWC}> -> tensor<1x64x80x80xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x64x80x80xf16, {order = #NHWC}>
}
