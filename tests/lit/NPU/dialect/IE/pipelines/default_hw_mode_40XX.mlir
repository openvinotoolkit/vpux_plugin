//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU40XX

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

        // CHECK:       [[CONV:%.+]] = IE.TransposedConvolution([[QUANTCAST1]], [[WEIGHTS]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        // CHECK-SAME:      output_padding = [0, 0],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [2, 2]
        // CHECK-SAME:  } :
        // CHECK-SAME:      tensor<1x512x16x16x!qElemType2, {order = #NHWC}>,
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

// CHECK-LABEL-DAG: @MatMulWithGroupQuant
// CHECK-DAG: [[Q_TYPE:!.+]] = !quant.uniform<i8<-127:127>:f16, 0.0078740157480314959>
module @MatMulWithGroupQuant {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<16x96xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<16x64xf32>
    }

    // CHECK: [[ARG0:%.+]]: tensor<16x96xf32>
    func.func @main(%arg0: tensor<16x96xf32>) -> tensor<16x64xf32> {
        %WEIGHTS = const.Declare tensor<3x32x64xf32> = dense<1.0> : tensor<3x32x64xf32>
        // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<64x32x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x32x64xf32>, [#const.SubView<[2, 0, 0], [1, 32, 64]>, #const.Reshape<[1, 32, 1, 64]>, #const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Transpose<#NWHC>, #const.Reshape<[64, 32, 1, 1]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<64x32x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x32x64xf32>, [#const.SubView<[1, 0, 0], [1, 32, 64]>, #const.Reshape<[1, 32, 1, 64]>, #const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Transpose<#NWHC>, #const.Reshape<[64, 32, 1, 1]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_2:%.+]] = const.Declare tensor<64x32x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x32x64xf32>, [#const.SubView<[0, 0, 0], [1, 32, 64]>, #const.Reshape<[1, 32, 1, 64]>, #const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Transpose<#NWHC>, #const.Reshape<[64, 32, 1, 1]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_LHS:%.+]] = IE.AffineReshape([[ARG0]]) {
        // CHECK-SAME:      shape_value = [1, 1, 16, 96]
        // CHECK-SAME:  } : tensor<16x96xf32> -> tensor<1x1x16x96xf32>
        // CHECK:   [[CONVERT_LHS:%.+]] = IE.Convert([[RESHAPE_LHS]]) {
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
        // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x1x16x96xf16> to tensor<1x1x16x32xf16>
        // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 32] [1, 1, 16, 32] : tensor<1x1x16x96xf16> to tensor<1x1x16x32xf16>
        // CHECK:   [[SLICE_2:%.+]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 64] [1, 1, 16, 32] : tensor<1x1x16x96xf16> to tensor<1x1x16x32xf16>

        // CHECK:   [[RESHAPE_SLICE_0:%.+]] = IE.AffineReshape([[SLICE_0]]) {
        // CHECK-SAME:      shape_value = [16, 32, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x32xf16> -> tensor<16x32x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_0:%.+]] = IE.PermuteCast([[RESHAPE_SLICE_0]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x32x1x1xf16> -> tensor<1x32x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_0:%.+]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_0]]) {
        // CHECK-SAME:      shape_value = [1, 32, 4, 4]
        // CHECK-SAME:  } : tensor<1x32x16x1xf16, {order = #NHWC}> -> tensor<1x32x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_0:%.+]] = IE.Convolution([[CONV_INPUT_0]], [[WEIGHTS_2]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<64x32x1x1x!qElemType, {order = #NHWC}> -> tensor<1x64x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_0:%.+]] = IE.AffineReshape([[CONV_0]]) {
        // CHECK-SAME:      shape_value = [1, 64, 16, 1]
        // CHECK-SAME:  } : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_SLICE_1:%.+]] = IE.AffineReshape([[SLICE_1]]) {
        // CHECK-SAME:      shape_value = [16, 32, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x32xf16> -> tensor<16x32x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_1:%.+]] = IE.PermuteCast([[RESHAPE_SLICE_1]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x32x1x1xf16> -> tensor<1x32x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_1:%.+]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_1]]) {
        // CHECK-SAME:      shape_value = [1, 32, 4, 4]
        // CHECK-SAME:  } : tensor<1x32x16x1xf16, {order = #NHWC}> -> tensor<1x32x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.+]] = IE.Convolution([[CONV_INPUT_1]], [[WEIGHTS_1]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<64x32x1x1x!qElemType, {order = #NHWC}> -> tensor<1x64x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_1:%.+]] = IE.AffineReshape([[CONV_1]]) {
        // CHECK-SAME:      shape_value = [1, 64, 16, 1]
        // CHECK-SAME:  } : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_SLICE_2:%.+]] = IE.AffineReshape([[SLICE_2]]) {
        // CHECK-SAME:      shape_value = [16, 32, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x32xf16> -> tensor<16x32x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_2:%.+]] = IE.PermuteCast([[RESHAPE_SLICE_2]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x32x1x1xf16> -> tensor<1x32x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_2:%.+]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_2]]) {
        // CHECK-SAME:      shape_value = [1, 32, 4, 4]
        // CHECK-SAME:  } : tensor<1x32x16x1xf16, {order = #NHWC}> -> tensor<1x32x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_2:%.+]] = IE.Convolution([[CONV_INPUT_2]], [[WEIGHTS_0]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x32x4x4xf16, {order = #NHWC}>, tensor<64x32x1x1x!qElemType, {order = #NHWC}> -> tensor<1x64x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_2:%.+]] = IE.AffineReshape([[CONV_2]]) {
        // CHECK-SAME:      shape_value = [1, 64, 16, 1]
        // CHECK-SAME:  } : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

        // CHECK:   [[ADD_0:%.+]] = IE.Accumulate([[RESHAPE_CONV_0]], [[RESHAPE_CONV_1]])
        // CHECK:   [[ADD_1:%.+]] = IE.Accumulate([[ADD_0]], [[RESHAPE_CONV_2]])

        // CHECK:   [[PERMUTE_CAST_OUT:%.+]] = IE.PermuteCast([[ADD_1]]) {
        // CHECK-SAME:      dst_order = #NCHW,
        // CHECK-SAME:      mem_perm = #NHCW
        // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}> -> tensor<1x1x16x64xf16>

        // CHECK:   [[CONVERT_OUT:%.+]] = IE.Convert([[PERMUTE_CAST_OUT]]) {
        // CHECK-SAME:      dstElemType = f32
        // CHECK-SAME:  } : tensor<1x1x16x64xf16> -> tensor<1x1x16x64xf32>

        // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[CONVERT_OUT]]) {
        // CHECK-SAME:      shape_value = [16, 64]
        // CHECK-SAME:  } : tensor<1x1x16x64xf32> -> tensor<16x64xf32>

        return %GEMM : tensor<16x64xf32>
        // CHECK:   return [[RESHAPE_OUT]] : tensor<16x64xf32>
    }
}
