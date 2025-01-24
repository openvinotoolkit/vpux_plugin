//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --consolidate-weights-dequantize --mlir-print-elementsattrs-with-hex-if-larger -1 %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00:8>

// CHECK-LABEL: @DynamicDequantization
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1xi4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x16x1x1xf16>
func.func @DynamicDequantization(%input: tensor<1x16x16x16xf16>, %weights: tensor<16x16x1x1xi4>, %scale: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %zp = const.Declare tensor<1x16x1x1xi4> = dense<8.0> : tensor<1x16x1x1xf16>,
              [#const.CastElemType<i4>]
    %zp_f16 = IE.Convert(%zp) { dstElemType = f16 } : tensor<1x16x1x1xi4> -> tensor<1x16x1x1xf16>
        
    %weights_f16 = IE.Convert(%weights) { dstElemType = f16 } : tensor<16x16x1x1xi4> -> tensor<16x16x1x1xf16>

    %subtract = IE.Subtract(%weights_f16, %zp_f16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
    %multiply = IE.Multiply(%subtract, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    %conv = IE.Convolution(%input, %multiply) {
              dilations = [1, 1],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              strides = [1, 1]
          } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>
              -> tensor<1x16x16x16xf16>

    return %conv : tensor<1x16x16x16xf16>
  
    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<16x16x1x1xi4> -> tensor<16x16x1x1x!qElemType>

    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT_CAST]], [[SCALE]]) {dstElemType = f16}
    // CHECK-SAME:     : tensor<16x16x1x1x!qElemType>, tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>

    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DYN_DEQUANT]])
    // CHECK-SAME:     {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:     : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    // CHECK:  return [[CONV]] : tensor<1x16x16x16xf16>
}


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @DynamicDequantizationWithTranspose
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<28x512x128xi4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<28x1x512xf32>
func.func @DynamicDequantizationWithTranspose(%weights: tensor<28x512x128xi4>, %scale: tensor<28x1x512xf32>) -> tensor<28x128x512xf32> {
    %weights_f32 = IE.Convert(%weights) {dstElemType = f32} : tensor<28x512x128xi4> -> tensor<28x512x128xf32>
    %transpose = IE.Transpose(%weights_f32) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<28x512x128xf32> -> tensor<28x128x512xf32>
    %multiply = IE.Multiply(%transpose, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<28x128x512xf32>, tensor<28x1x512xf32> -> tensor<28x128x512xf32>

    return %multiply : tensor<28x128x512xf32>

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<28x512x128xi4> -> tensor<28x512x128x!qElemType>
    // CHECK:  [[TRANSPOSE:%.+]] = IE.Transpose([[QUANT_CAST]]) {order_value = #map} : tensor<28x512x128x!qElemType> -> tensor<28x128x512x!qElemType>
    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[TRANSPOSE]], [[SCALE]]) {dstElemType = f32} : tensor<28x128x512x!qElemType>, tensor<28x1x512xf32> -> tensor<28x128x512xf32>
    // CHECK:  return [[DYN_DEQUANT]] : tensor<28x128x512xf32>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @DynamicDequantizationForINT8Weights
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<73440x1536xsi8>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<73440x1xf16>,
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x1536xf32>
func.func @DynamicDequantizationForINT8Weights(%weights: tensor<73440x1536xsi8>, %scale: tensor<73440x1xf16>, %input: tensor<1x1536xf32>) -> tensor<1x73440xf32> {
    %weights_f16 = IE.Convert(%weights) {dstElemType = f16} : tensor<73440x1536xsi8> -> tensor<73440x1536xf16>
    %multiply = IE.Multiply(%weights_f16, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<73440x1536xf16>, tensor<73440x1xf16> -> tensor<73440x1536xf16>
    %weights_f32 = IE.Convert(%multiply) {dstElemType = f32} : tensor<73440x1536xf16> -> tensor<73440x1536xf32>
    %fc = IE.FullyConnected(%input, %weights_f32) : tensor<1x1536xf32>, tensor<73440x1536xf32> -> tensor<1x73440xf32>

    return %fc: tensor<1x73440xf32>

    // CHECK:  [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<73440x1536xsi8> -> tensor<73440x1536x!qElemType>
    // CHECK:  [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT_CAST]], [[SCALE]]) {dstElemType = f16} : tensor<73440x1536x!qElemType>, tensor<73440x1xf16> -> tensor<73440x1536xf16>
    // CHECK:  [[CONVERT:%.+]] = IE.Convert([[DYN_DEQUANT]]) {dstElemType = f32} : tensor<73440x1536xf16> -> tensor<73440x1536xf32>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[CONVERT]]) : tensor<1x1536xf32>, tensor<73440x1536xf32> -> tensor<1x73440xf32>

    // CHECK:  return [[FC]] : tensor<1x73440xf32>
}
