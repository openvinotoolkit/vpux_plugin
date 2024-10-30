//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: arch-NPU37XX

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: @Conv2dLeakyReluWithQuantize
func.func @Conv2dLeakyReluWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x3x3x3x!qElemType> {
    %cst = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> : tensor<3x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.PostOp<
            name = "IE.LeakyRelu",
            attrs = {
                negative_slope = 2.500000e-01 : f64
            }
        >,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x3x3x3xf16> -> tensor<1x3x3x3x!qElemType>

    return %1 : tensor<1x3x3x3x!qElemType>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<3x16x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.PostOp<
    // CHECK-SAME:          name = "IE.LeakyRelu",
    // CHECK-SAME:          attrs = {
    // CHECK-SAME:              negative_slope = 2.500000e-01 : f64
    // CHECK-SAME:          }
    // CHECK-SAME:      >,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x3x3x3x!qElemType>
}
