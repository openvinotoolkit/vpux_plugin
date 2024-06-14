//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --operation-conversion %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


// CHECK-LABEL: @OperationConversionAllOpsSubset
func.func @OperationConversionAllOpsSubset(%arg0: tensor<1x16x8x12xf16>) -> tensor<1x12x8x1xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x16x8x12xf16> -> tensor<1x1x8x12xf16>
  %1 = IE.ExtractImagePatches(%0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [1, 12], strides = [1, 1]} : tensor<1x1x8x12xf16> -> tensor<1x12x8x1xf16>
  %cst_exponent = const.Declare tensor<1xf16> = dense<2.0> : tensor<1xf16>
  %sqd = IE.SquaredDiff(%1, %cst_exponent) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x8x1xf16>, tensor<1xf16> -> tensor<1x12x8x1xf16>
  return %sqd : tensor<1x12x8x1xf16>


  //CHECK: [[CST:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NCHW>]
  //CHECK: [[CST0:%.*]] = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<1xf16>
  //CHECK: [[VAL0:%.*]] = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x8x12xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x8x12xf16>
  //CHECK: [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #NWHC} : tensor<1x1x8x12xf16> -> tensor<1x12x8x1xf16>
  //CHECK: [[VAL2:%.*]] = IE.SquaredDiff([[VAL1]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x8x1xf16>, tensor<1xf16> -> tensor<1x12x8x1xf16>

}
