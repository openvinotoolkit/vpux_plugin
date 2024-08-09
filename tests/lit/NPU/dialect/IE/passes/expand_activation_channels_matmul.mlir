//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --expand-activation-channels %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ExpandICMatMul([[ARG0:%.+]]: tensor<64x4x50x49xf16>) ->  tensor<64x4x50x32xf16> {
func.func @ExpandICMatMul(%arg0: tensor<64x4x50x49xf16>) -> tensor<64x4x50x32xf16> {
    %0 = const.Declare tensor<64x4x49x32xf16> =
        dense<1.0> : tensor<64x4x49x32xf16>
    %1 = IE.MatMul(%arg0, %0) : tensor<64x4x50x49xf16>, tensor<64x4x49x32xf16> -> tensor<64x4x50x32xf16>

    return %1 : tensor<64x4x50x32xf16>
    // CHECK:       [[VAL0:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]}
    // CHECK-SAME:  : tensor<64x4x50x49xf16> -> tensor<64x4x50x64xf16>
    // CHECK:       [[CST:%.+]] = const.Declare tensor<64x4x64x32xf16> = dense<1.000000e+00> :
    // CHECK-SAME:  tensor<64x4x49x32xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 15, 0]>]
    // CHECK:       [[VAL1:%.+]] = IE.MatMul([[VAL0]], [[CST]])
    // CHECK-SAME:  : tensor<64x4x50x64xf16>, tensor<64x4x64x32xf16> -> tensor<64x4x50x32xf16>
    // CHECK:       [[VAL2:%.+]] = IE.Slice [[VAL1]] [0, 0, 0, 0] [64, 4, 50, 32]
    // CHECK-SAME:  : tensor<64x4x50x32xf16> to tensor<64x4x50x32xf16>
    // CHECK:       return [[VAL2]] : tensor<64x4x50x32xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK:       func.func @ExpandICMatMulWithTranspose([[ARG0:%.+]]: tensor<64x4x50x49xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<64x4x49x32xf16>) ->  tensor<64x4x50x32xf16> {
func.func @ExpandICMatMulWithTranspose(%arg0: tensor<64x4x50x49xf16>, %arg1: tensor<64x4x49x32xf16>) -> tensor<64x4x50x32xf16> {
    %0 = IE.Transpose(%arg1) {order_value = #NCWH} : tensor<64x4x49x32xf16> -> tensor<64x4x32x49xf16>
    %1 = IE.MatMul(%arg0, %0) {transpose_b} : tensor<64x4x50x49xf16>, tensor<64x4x32x49xf16> -> tensor<64x4x50x32xf16>
    
    return %1 : tensor<64x4x50x32xf16>
    // CHECK:       [[VAL0:%.+]] = IE.Transpose([[ARG1]]) {order_value = #NCWH}
    // CHECK-SAME:  : tensor<64x4x49x32xf16> -> tensor<64x4x32x49xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]}
    // CHECK-SAME:  : tensor<64x4x50x49xf16> -> tensor<64x4x50x64xf16>
    // CHECK:       [[VAL2:%.+]] = IE.Expand([[VAL0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]}
    // CHECK-SAME:  : tensor<64x4x32x49xf16> -> tensor<64x4x32x64xf16>
    // CHECK:       [[VAL3:%.+]] = IE.MatMul([[VAL1]], [[VAL2]]) {transpose_b}
    // CHECK-SAME:  : tensor<64x4x50x64xf16>, tensor<64x4x32x64xf16> -> tensor<64x4x50x32xf16>
    // CHECK:       [[VAL4:%.+]] = IE.Slice [[VAL3]] [0, 0, 0, 0] [64, 4, 50, 32]
    // CHECK-SAME:  : tensor<64x4x50x32xf16> to tensor<64x4x50x32xf16>
    // CHECK:       return [[VAL4]] : tensor<64x4x50x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK:       func.func @ExpandOCMatMulWithTranspose([[ARG0:%.+]]: tensor<16x8x49x32xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x8x32x50xf16>) ->  tensor<16x8x49x50xf16> {
func.func @ExpandOCMatMulWithTranspose(%arg0: tensor<16x8x49x32xf16>, %arg1: tensor<16x8x32x50xf16>) -> tensor<16x8x49x50xf16>  {
    %0 = IE.Transpose(%arg1) {order_value = #NCWH} : tensor<16x8x32x50xf16> -> tensor<16x8x50x32xf16>
    %1 = IE.MatMul(%arg0, %0) {transpose_b} : tensor<16x8x49x32xf16>, tensor<16x8x50x32xf16> -> tensor<16x8x49x50xf16> 

    return %1 : tensor<16x8x49x50xf16> 
    // CHECK:       [[VAL0:%.+]] = IE.Transpose([[ARG1]]) {order_value = #NCWH}
    // CHECK-SAME:  : tensor<16x8x32x50xf16> -> tensor<16x8x50x32xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Expand([[VAL0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 14, 0]}
    // CHECK-SAME:  : tensor<16x8x50x32xf16> -> tensor<16x8x64x32xf16>
    // CHECK:       [[VAL2:%.+]] = IE.MatMul([[ARG0]], [[VAL1]]) {transpose_b}
    // CHECK-SAME:  : tensor<16x8x49x32xf16>, tensor<16x8x64x32xf16> -> tensor<16x8x49x64xf16>
    // CHECK:       [[VAL3:%.+]] = IE.Slice [[VAL2]] [0, 0, 0, 0] [16, 8, 49, 50]
    // CHECK-SAME:  : tensor<16x8x49x64xf16> to tensor<16x8x49x50xf16>
    // CHECK:       return [[VAL3]] : tensor<16x8x49x50xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK:       func.func @ExpandICOCMatMulWithTranspose([[ARG0:%.+]]:  tensor<16x8x49x25xf16>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x8x25x49xf16>) ->  tensor<16x8x49x49xf16> {
func.func @ExpandICOCMatMulWithTranspose(%arg0: tensor<16x8x49x25xf16>, %arg1: tensor<16x8x25x49xf16>) -> tensor<16x8x49x49xf16>  {
    %0 = IE.Transpose(%arg1) {order_value = #NCWH} : tensor<16x8x25x49xf16> -> tensor<16x8x49x25xf16>
    %1 = IE.MatMul(%arg0, %0) {transpose_b} : tensor<16x8x49x25xf16>, tensor<16x8x49x25xf16> -> tensor<16x8x49x49xf16> 

    return %1 : tensor<16x8x49x49xf16> 
    // CHECK:       [[VAL0:%.+]] = IE.Transpose([[ARG1]]) {order_value = #NCWH}
    // CHECK-SAME:  : tensor<16x8x25x49xf16> -> tensor<16x8x49x25xf16>
    // CHECK:       [[VAL1:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]}
    // CHECK-SAME:  : tensor<16x8x49x25xf16> -> tensor<16x8x49x32xf16>
    // CHECK:       [[VAL2:%.+]] = IE.Expand([[VAL0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]}
    // CHECK-SAME:  : tensor<16x8x49x25xf16> -> tensor<16x8x49x32xf16>
    // CHECK:       [[VAL3:%.+]] = IE.Expand([[VAL2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 15, 0]}
    // CHECK-SAME:  : tensor<16x8x49x32xf16> -> tensor<16x8x64x32xf16>
    // CHECK:       [[VAL4:%.+]] = IE.MatMul([[VAL1]], [[VAL3]]) {transpose_b}
    // CHECK-SAME:  : tensor<16x8x49x32xf16>, tensor<16x8x64x32xf16> -> tensor<16x8x49x64xf16>
    // CHECK:       [[VAL5:%.+]] = IE.Slice [[VAL4]] [0, 0, 0, 0] [16, 8, 49, 49]
    // CHECK-SAME:  : tensor<16x8x49x64xf16> to tensor<16x8x49x49xf16>
    // CHECK:       return [[VAL5]] : tensor<16x8x49x49xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK:       func.func @ExpandICOCMatMul([[ARG0:%.+]]:  tensor<16x8x49x25xf16>
// CHECK-SAME:  ->  tensor<16x8x49x49xf16> {
func.func @ExpandICOCMatMul(%arg0: tensor<16x8x49x25xf16>) -> tensor<16x8x49x49xf16>  {
    %0 = const.Declare tensor<16x8x49x25xf16> =
        dense<1.0> : tensor<16x8x49x25xf16>
    %1 = IE.MatMul(%arg0, %0) {transpose_b} : tensor<16x8x49x25xf16>, tensor<16x8x49x25xf16> -> tensor<16x8x49x49xf16> 

    return %1 : tensor<16x8x49x49xf16> 
    // CHECK:           [[VAL0:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 7]}
    // CHECK-SAME: :    tensor<16x8x49x25xf16> -> tensor<16x8x49x32xf16>
    // CHECK:           [[CST1:%.+]] = const.Declare tensor<16x8x64x32xf16> = dense<1.000000e+00> : tensor<16x8x49x25xf16>,
    // CHECK-SAME:      [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 7]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 15, 0]>]
    // CHECK:           [[VAL1:%.+]] = IE.MatMul([[VAL0]], [[CST1]]) {transpose_b}
    // CHECK-SAME:      : tensor<16x8x49x32xf16>, tensor<16x8x64x32xf16> -> tensor<16x8x49x64xf16>
    // CHECK:           [[VAL2:%.+]] = IE.Slice [[VAL1]] [0, 0, 0, 0] [16, 8, 49, 49]
    // CHECK-SAME:      : tensor<16x8x49x64xf16> to tensor<16x8x49x49xf16>
    // CHECK:           return [[VAL2]] : tensor<16x8x49x49xf16>
}
