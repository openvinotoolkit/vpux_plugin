//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0064682904411764702:128>
!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
!qElemType1 = !quant.uniform<u8:f16, 0.01293658088235294:64>
!qElemType2 = !quant.uniform<u8:f16, 0.0064682904411764702:128>

// CHECK-LABEL: @FuseQuantizeCasts
func.func @FuseQuantizeCasts(%arg0: tensor<1x16x32x64x!qElemType>) -> tensor<1x16x32x64x!qElemType2> {
    %FIRST_QUANT_CAST = IE.QuantizeCast(%arg0) {
        dstElemType = !qElemType1
    } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType1>

    %SECOND_QUANT_CAST = IE.QuantizeCast(%FIRST_QUANT_CAST) {
        dstElemType = !qElemType2
    } : tensor<1x16x32x64x!qElemType1> -> tensor<1x16x32x64x!qElemType2>

    return %SECOND_QUANT_CAST : tensor<1x16x32x64x!qElemType2>

    // CHECK:       [[QUANT_CAST:%.*]] = IE.QuantizeCast(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType1
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType1>

    // CHECK:       return [[QUANT_CAST]] : tensor<1x16x32x64x!qElemType1>
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0064682904411764702:128>
// CHECK: !qElemType2 = !quant.uniform<u8:f16, 0.01293658088235294:64>
!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
!qElemType1 = !quant.uniform<u8:f16, 0.01293658088235294:64>
!qElemType2 = !quant.uniform<u8:f16, 0.0064682904411764702:128>

// CHECK-LABEL: @FuseQuantCastsMultipleConsumers
func.func @FuseQuantCastsMultipleConsumers(%arg0: tensor<1x16x32x64x!qElemType>) -> tensor<1x16x32x64x!qElemType2> {
    %FIRST_QUANT_CAST = IE.QuantizeCast(%arg0) {
        dstElemType = !qElemType1
    } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType1>

    %ADD = IE.Add(%FIRST_QUANT_CAST, %FIRST_QUANT_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x32x64x!qElemType1>, tensor<1x16x32x64x!qElemType1> -> tensor<1x16x32x64x!qElemType1>

    %ADD_QUANT_CAST = IE.QuantizeCast(%ADD) {
        dstElemType = !qElemType2
    } : tensor<1x16x32x64x!qElemType1> -> tensor<1x16x32x64x!qElemType2>

    %SECOND_QUANT_CAST = IE.QuantizeCast(%FIRST_QUANT_CAST) {
        dstElemType = !qElemType2
    } : tensor<1x16x32x64x!qElemType1> -> tensor<1x16x32x64x!qElemType2>

    %MUL = IE.Multiply(%SECOND_QUANT_CAST, %ADD_QUANT_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x32x64x!qElemType2>, tensor<1x16x32x64x!qElemType2> -> tensor<1x16x32x64x!qElemType2>

    return %MUL : tensor<1x16x32x64x!qElemType2>

    // CHECK:       [[FIRST_QUANT_CAST:%.*]] = IE.QuantizeCast(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType2
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType2>

    // CHECK:       [[ADD:%.*]] = IE.Add([[FIRST_QUANT_CAST]], [[FIRST_QUANT_CAST]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType2>, tensor<1x16x32x64x!qElemType2>
    // CHECK-SAME:  -> tensor<1x16x32x64x!qElemType2>

    // CHECK:       [[ADD_QUANT_CAST:%.*]] = IE.QuantizeCast([[ADD]]) {
    // CHECK-SAME:      dstElemType = !qElemType1
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType2> -> tensor<1x16x32x64x!qElemType1>

    // Note that the second IE.QuantizeCast accepts arg0, not FIRST_QUANT_CAST
    // CHECK:       [[SECOND_QUANT_CAST:%.*]] = IE.QuantizeCast(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType1
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType> -> tensor<1x16x32x64x!qElemType1>

    // CHECK:       [[MUL:%.*]] = IE.Multiply([[SECOND_QUANT_CAST]], [[ADD_QUANT_CAST]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x16x32x64x!qElemType1>, tensor<1x16x32x64x!qElemType1>
    // CHECK-SAME:  -> tensor<1x16x32x64x!qElemType1>

    // CHECK:       return [[MUL]] : tensor<1x16x32x64x!qElemType1>
}
