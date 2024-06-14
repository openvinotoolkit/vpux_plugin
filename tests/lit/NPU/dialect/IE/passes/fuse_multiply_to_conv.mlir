//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-multiply-to-conv --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @FuseMultiplyToConvolutionSimple
// CHECK-SAME: ([[ARG0:%.+]]: tensor<512x96x1x1xf16>, [[ARG1:%.+]]: tensor<512x96x1x1xf16>)
// CHECK-SAME:  -> tensor<512x512x1x1xf16>
func.func @FuseMultiplyToConvolutionSimple(%arg0: tensor<512x96x1x1xf16>, %arg1: tensor<512x96x1x1xf16>)
        -> tensor<512x512x1x1xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>, [#const.Add<1.0 : f16>]

    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<512x96x1x1xf16>, tensor<512x96x1x1xf16> -> tensor<512x512x1x1xf16>

    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<512x512x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<512x512x1x1xf16>

    return %1 : tensor<512x512x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]])
    // CHECK-SAME:  static_scale = 3.000000e+00

    // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @FuseMultiplyToConvolution
// CHECK-SAME: ([[ARG00:%.+]]: tensor<512x96x1x1xf16>, [[ARG01:%.+]]: tensor<512x96x1x1xf16>, [[ARG10:%.+]]: tensor<512x96x1x1xf16>, [[ARG11:%.+]]: tensor<512x96x1x1xf16>)
// CHECK-SAME:  -> tensor<512x512x2x1xf16>
func.func @FuseMultiplyToConvolution(%arg00: tensor<512x96x1x1xf16>, %arg01: tensor<512x96x1x1xf16>,
                                     %arg10: tensor<512x96x1x1xf16>, %arg11: tensor<512x96x1x1xf16>)
        -> tensor<512x512x2x1xf16> {
    %scale = const.Declare tensor<512x512x2x1xf16> = dense<0.2> : tensor<512x512x2x1xf16>

    %0 = IE.Convolution(%arg00, %arg01) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<512x96x1x1xf16>, tensor<512x96x1x1xf16> -> tensor<512x512x1x1xf16>
    %1 = IE.Convolution(%arg10, %arg11) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<512x96x1x1xf16>, tensor<512x96x1x1xf16> -> tensor<512x512x1x1xf16>

    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} :
        tensor<512x512x1x1xf16>, tensor<512x512x1x1xf16> -> tensor<512x512x2x1xf16>

    %3 = IE.Multiply(%2, %scale) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<512x512x2x1xf16>, tensor<512x512x2x1xf16> -> tensor<512x512x2x1xf16>

    return %3 : tensor<512x512x2x1xf16>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[ARG00]], [[ARG01]])
    // CHECK-SAME:  static_scale = 0.199951172 : f32

    // CHECK: [[CONV1:%.+]] = IE.Convolution([[ARG10]], [[ARG11]])
    // CHECK-SAME:  static_scale = 0.199951172 : f32

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]])
    // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @FuseMultiplyToConvolutionFull
// CHECK-SAME: ([[ARG00:%.+]]: tensor<512x96x1x1xf16>, [[ARG01:%.+]]: tensor<512x96x1x1xf16>, [[ARG10:%.+]]: tensor<512x96x1x1xf16>, [[ARG11:%.+]]: tensor<512x96x1x1xf16>)
// CHECK-SAME:  -> tensor<1x2x512x512xf16>
func.func @FuseMultiplyToConvolutionFull(%arg00: tensor<512x96x1x1xf16>, %arg01: tensor<512x96x1x1xf16>,
                                         %arg10: tensor<512x96x1x1xf16>, %arg11: tensor<512x96x1x1xf16>)
        -> tensor<1x2x512x512xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.135345653> : tensor<1x1x1x1xf16>

    %47 = IE.Convolution(%arg00, %arg01) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<512x96x1x1xf16>, tensor<512x96x1x1xf16> -> tensor<512x512x1x1xf16>
    %50 = IE.Convolution(%arg10, %arg11) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<512x96x1x1xf16>, tensor<512x96x1x1xf16> -> tensor<512x512x1x1xf16>

    %81 = IE.AffineReshape(%47) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 512, 512]}
        : tensor<512x512x1x1xf16> -> tensor<1x1x512x512xf16>
    %82 = IE.AffineReshape(%50) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 512, 512]}
        : tensor<512x512x1x1xf16> -> tensor<1x1x512x512xf16>

    %93 = IE.Concat(%81, %82) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} :
        tensor<1x1x512x512xf16>, tensor<1x1x512x512xf16> -> tensor<1x2x512x512xf16>

    %94 = IE.Multiply(%93, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x2x512x512xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x512xf16>

    return %94 : tensor<1x2x512x512xf16>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[ARG00]], [[ARG01]])
    // CHECK-SAME:  static_scale = 0.135375977 : f32

    // CHECK: [[CONV1:%.+]] = IE.Convolution([[ARG10]], [[ARG11]])
    // CHECK-SAME:  static_scale = 0.135375977 : f32

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[CONV0]])
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV1]])

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]])
    // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @DoNotReorderMultiply
// CHECK-SAME: ([[ARG00:%.+]]: tensor<512x96x1x1xf16>, [[ARG01:%.+]]: tensor<512x96x1x1xf16>,
// CHECK-SAME:  [[UNKNOWN:%.+]]: tensor<1x1x512x512xf16>)
// CHECK-SAME:  -> tensor<1x2x512x512xf16>
func.func @DoNotReorderMultiply(%arg0: tensor<512x96x1x1xf16>, %arg1: tensor<512x96x1x1xf16>,
                                %unknown: tensor<1x1x512x512xf16>)
        -> tensor<1x2x512x512xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<1.25> : tensor<1x1x1x1xf16>

    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<512x96x1x1xf16>, tensor<512x96x1x1xf16> -> tensor<512x512x1x1xf16>

    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 512, 512]}
        : tensor<512x512x1x1xf16> -> tensor<1x1x512x512xf16>

    %2 = IE.Concat(%1, %unknown) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} :
        tensor<1x1x512x512xf16>, tensor<1x1x512x512xf16> -> tensor<1x2x512x512xf16>

    %3 = IE.Multiply(%2, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x2x512x512xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x512xf16>

    return %3 : tensor<1x2x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[ARG00]], [[ARG01]])
    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[CONV0]])
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[RESHAPE0]], [[UNKNOWN]])

    // CHECK: [[MULT:%.+]] = IE.Multiply([[CONCAT]], [[CONST]])
    // CHECK: return [[MULT]]
}

// -----

// CHECK-LABEL: @DoNotFuseMultiply_NonConst
// CHECK-SAME: ([[ARG0:%.+]]: tensor<512x96x1x1xf16>, [[ARG1:%.+]]: tensor<512x96x1x1xf16>,
// CHECK-SAME:  [[SCALE:%.+]]: tensor<512x512x1x1xf16>)
// CHECK-SAME:  -> tensor<512x512x1x1xf16>
func.func @DoNotFuseMultiply_NonConst(%arg0: tensor<512x96x1x1xf16>, %arg1: tensor<512x96x1x1xf16>,
                                      %scale: tensor<512x512x1x1xf16>)
        -> tensor<512x512x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<512x96x1x1xf16>, tensor<512x96x1x1xf16> -> tensor<512x512x1x1xf16>

    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<512x512x1x1xf16>, tensor<512x512x1x1xf16> -> tensor<512x512x1x1xf16>

    return %1 : tensor<512x512x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]])
    // CHECK: [[MULT:%.+]] = IE.Multiply([[CONV]], [[SCALE]])
    // CHECK: return [[MULT]]
}

// -----

// CHECK-LABEL: @FuseMultiply_PresentPpe
// CHECK-SAME: ([[ARG0:%.+]]: tensor<512x96x1x1xf16>, [[ARG1:%.+]]: tensor<512x96x1x1xf16>)
// CHECK-SAME:  -> tensor<512x512x1x1xf16>
func.func @FuseMultiply_PresentPpe(%arg0: tensor<512x96x1x1xf16>, %arg1: tensor<512x96x1x1xf16>)
        -> tensor<512x512x1x1xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

    %0 = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1],
        post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.1}>
    } : tensor<512x96x1x1xf16>, tensor<512x96x1x1xf16> -> tensor<512x512x1x1xf16>

    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<512x512x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<512x512x1x1xf16>

    return %1 : tensor<512x512x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]])
    // CHECK-SAME:  post_op = #IE.PostOp<name = "IE.LeakyRelu"
    // CHECK-SAME:  static_scale = 2.000000e+00

    // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @FuseMultiply_TwoScales
// CHECK-SAME: ([[ARG0:%.+]]: tensor<512x96x1x1xf16>, [[ARG1:%.+]]: tensor<512x96x1x1xf16>)
// CHECK-SAME:  -> tensor<512x512x1x1xf16>
func.func @FuseMultiply_TwoScales(%arg0: tensor<512x96x1x1xf16>, %arg1: tensor<512x96x1x1xf16>)
        -> tensor<512x512x1x1xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

    %0 = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1],
        static_scale = 42.0 : f32
    } : tensor<512x96x1x1xf16>, tensor<512x96x1x1xf16> -> tensor<512x512x1x1xf16>

    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<512x512x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<512x512x1x1xf16>

    return %1 : tensor<512x512x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]])
    // CHECK-SAME:  static_scale = 8.400000e+01 : f32

    // CHECK: return [[CONV]]
}
