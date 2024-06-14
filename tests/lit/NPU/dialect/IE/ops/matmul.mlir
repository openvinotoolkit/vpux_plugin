//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @MatMulTransposeInputsTransposeAllTrue
// CHECK-SAME:      [[INPUT:%.+]]: tensor<256x166xf32>
func.func @MatMulTransposeInputsTransposeAllTrue(%arg0: tensor<256x166xf32>, %arg1: tensor<256x256xf32>) -> tensor<166x256xf32> {
    %cst = const.Declare tensor<256x256xf32> = dense<1.0> : tensor<256x256xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_a, transpose_b} : tensor<256x166xf32>, tensor<256x256xf32> -> tensor<166x256xf32>

    return %0 : tensor<166x256xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<256x256xf32> = dense<1.000000e+00> : tensor<256x256xf32>
    // CHECK:   IE.Transpose([[INPUT]]) {order_value = #CN} : tensor<256x166xf32> -> tensor<166x256xf32>
    // CHECK:   IE.FullyConnected(%0, [[CST]]) : tensor<166x256xf32>, tensor<256x256xf32> -> tensor<166x256xf32>
    // CHECK:   return %1 : tensor<166x256xf32>
}

// -----

// CHECK-LABEL: @MatMulTransposeInputsTransposeAFalse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<196x128xf32>
func.func @MatMulTransposeInputsTransposeAFalse(%arg0: tensor<196x128xf32>, %arg1: tensor<640x128xf32>) -> tensor<196x640xf32> {
    %cst = const.Declare tensor<640x128xf32> = dense<1.0> : tensor<640x128xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<196x128xf32>, tensor<640x128xf32> -> tensor<196x640xf32>

    return %0 : tensor<196x640xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<640x128xf32> = dense<1.000000e+00> : tensor<640x128xf32>
    // CHECK:   %0 = IE.FullyConnected([[INPUT]], [[CST]]) : tensor<196x128xf32>, tensor<640x128xf32> -> tensor<196x640xf32>
    // CHECK:   return %0 : tensor<196x640xf32>
}

// // -----

// CHECK-LABEL: @MatMulTransposeInputsTransposeBFalse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<40x131072xf32>
func.func @MatMulTransposeInputsTransposeBFalse(%arg0: tensor<40x131072xf32>, %arg1: tensor<40x19xf32>) -> tensor<131072x19xf32> {
    %cst = const.Declare tensor<40x19xf32> = dense<1.0> : tensor<40x19xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_a} : tensor<40x131072xf32>, tensor<40x19xf32> -> tensor<131072x19xf32>

    return %0 : tensor<131072x19xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<19x40xf32> = dense<1.000000e+00> : tensor<40x19xf32>, [#const.Transpose<#CN>]
    // CHECK:   %0 = IE.Transpose([[INPUT]]) {order_value = #CN} : tensor<40x131072xf32> -> tensor<131072x40xf32>
    // CHECK:   %1 = IE.FullyConnected(%0, [[CST]]) : tensor<131072x40xf32>, tensor<19x40xf32> -> tensor<131072x19xf32>
    // CHECK:   return %1 : tensor<131072x19xf32>
}

// -----

// CHECK-LABEL: @MatMulTransposeInputsTransposeAllFalse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x784xf32>
func.func @MatMulTransposeInputsTransposeAllFalse(%arg0: tensor<1x784xf32>, %arg1: tensor<784x256xf32>) -> tensor<1x256xf32> {
    %cst = const.Declare tensor<784x256xf32> = dense<1.0> : tensor<784x256xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x784xf32>, tensor<784x256xf32> -> tensor<1x256xf32>

    return %0 : tensor<1x256xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<256x784xf32> = dense<1.000000e+00> : tensor<784x256xf32>, [#const.Transpose<#CN>]
    // CHECK:   %0 = IE.FullyConnected([[INPUT]], [[CST]]) : tensor<1x784xf32>, tensor<256x784xf32> -> tensor<1x256xf32>
    // CHECK:   return %0 : tensor<1x256xf32>
}

// -----

// CHECK-LABEL: @MatMul_SI32toFP16
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x20x560xsi32>
func.func @MatMul_SI32toFP16(%arg0: tensor<1x20x560xsi32>) -> tensor<1x20x1536xsi32> {
    %cst = const.Declare tensor<1536x560xsi32> = dense<1> : tensor<1536x560xsi32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x20x560xsi32>, tensor<1536x560xsi32> -> tensor<1x20x1536xsi32>
    return %0 : tensor<1x20x1536xsi32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1536x560xf16> = dense<1> : tensor<1536x560xsi32>, [#const.ConvertElemType<f16>]
    // CHECK: [[INPUT_CVT:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x20x560xsi32> -> tensor<1x20x560xf16>
    // CHECK: [[MAT_MUL:%.*]] = IE.MatMul([[INPUT_CVT]], [[CST]]) {transpose_b} : tensor<1x20x560xf16>, tensor<1536x560xf16> -> tensor<1x20x1536xf16>
    // CHECK: [[OUT_CVT:%.*]] = IE.Convert([[MAT_MUL]]) {dstElemType = si32} : tensor<1x20x1536xf16> -> tensor<1x20x1536xsi32>
    // CHECK: return [[OUT_CVT]] : tensor<1x20x1536xsi32>
}


// -----

// CHECK-LABEL: @PropagateTransposeToConst
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2560xf32>
func.func @PropagateTransposeToConst(%arg0: tensor<1x2560xf32>) -> tensor<1x1280xf32> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst_1 = const.Declare tensor<1280x20x1xf32> = dense<2.500000e+01>  : tensor<1280x20x1xf32>
    %cst_2 = const.Declare tensor<1280x20x1xf32> = dense<3.500000e+01>  : tensor<1280x20x1xf32>
    %cst_3 = const.Declare tensor<1280x20x128xf32> = dense<4.500000e+01>  : tensor<1280x20x128xf32>
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1280x20x128xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1280x20x1xf32>, tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1]], shape_value = [1280, 2560]} : tensor<1280x20x128xf32> -> tensor<1280x2560xf32>
    %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x2560xf32>, tensor<1280x2560xf32> -> tensor<1x1280xf32>

    return %2 : tensor<1x1280xf32>

    // CHECK-DAG: [[CST:%.*]]  = const.Declare tensor<20x128x1280xf32> = dense<4.500000e+01> : tensor<1280x20x128xf32>, [#const.Transpose<#HWC>]
    // CHECK-DAG: [[CST0:%.*]]  = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Transpose<#HWC>]
    // CHECK-DAG: [[CST1:%.*]]  = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>, [#const.Transpose<#HWC>]
    // CHECK-DAG: [[CST2:%.*]]  = const.Declare tensor<20x1x1280xf32> = dense<2.500000e+01> : tensor<1280x20x1xf32>, [#const.Transpose<#HWC>]
    // CHECK-DAG: [[CST3:%.*]]  = const.Declare tensor<20x1x1280xf32> = dense<3.500000e+01> : tensor<1280x20x1xf32>, [#const.Transpose<#HWC>]
    // CHECK:     [[FQ:%.*]]  = IE.FakeQuantize([[CST]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<20x128x1280xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<20x1x1280xf32>, tensor<20x1x1280xf32> -> tensor<20x128x1280xf32>
    // CHECK:     [[RESHAPE:%.*]] = IE.AffineReshape([[FQ]])
    // CHECK-SAME{LITERAL}:                {dim_mapping = [[0], [0], [1]], shape_value = [2560, 1280]} : tensor<20x128x1280xf32> -> tensor<2560x1280xf32>
    // CHECK:     [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE]]) {order_value = #CN} : tensor<2560x1280xf32> -> tensor<1280x2560xf32>
    // CHECK:     [[FULLCONNECTED:%.*]] = IE.FullyConnected([[INPUT]], [[TRANSPOSE]]) : tensor<1x2560xf32>, tensor<1280x2560xf32> -> tensor<1x1280xf32>
    // CHECK:     return [[FULLCONNECTED]] : tensor<1x1280xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @PropagateTransposeToConstForDifferentDimMapping
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024xf32>
func.func @PropagateTransposeToConstForDifferentDimMapping(%arg0: tensor<1x1024xf32>) -> tensor<1x1023xf32> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<2.500000e+01>  : tensor<1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1xf32> = dense<3.500000e+01>  : tensor<1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1023x1024xf32> = dense<4.500000e+01>  : tensor<1x1023x1024xf32>
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x1023x1024xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1023x1024xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [1023, 1024]} : tensor<1x1023x1024xf32> -> tensor<1023x1024xf32>
    %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x1024xf32>, tensor<1023x1024xf32> -> tensor<1x1023xf32>

    return %2 : tensor<1x1023xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1024x1x1023xf32> = dense<4.500000e+01> : tensor<1x1023x1024xf32>, [#const.Transpose<#map>]
    // CHECK-DAG: [[CST0:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Transpose<#map>]
    // CHECK-DAG: [[CST1:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>, [#const.Transpose<#map>]
    // CHECK-DAG: [[CST2:%.*]] = const.Declare tensor<1x1x1xf32> = dense<2.500000e+01> : tensor<1x1x1xf32>, [#const.Transpose<#map>]
    // CHECK-DAG: [[CST3:%.*]] = const.Declare tensor<1x1x1xf32> = dense<3.500000e+01> : tensor<1x1x1xf32>, [#const.Transpose<#map>]
    // CHECK:     [[FQ:%.*]] = IE.FakeQuantize([[CST]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1024x1x1023xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1024x1x1023xf32>
    // CHECK:     [[RESHAPE:%.*]] = IE.AffineReshape([[FQ]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [1024, 1023]} : tensor<1024x1x1023xf32> -> tensor<1024x1023xf32>
    // CHECK:     [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE]]) {order_value = #CN} : tensor<1024x1023xf32> -> tensor<1023x1024xf32>
    // CHECK:     [[FULLCONNECTED:%.*]] = IE.FullyConnected([[INPUT]], [[TRANSPOSE]]) : tensor<1x1024xf32>, tensor<1023x1024xf32> -> tensor<1x1023xf32>
    // CHECK:       return [[FULLCONNECTED]] : tensor<1x1023xf32>

}

// -----

// CHECK-LABEL: @NotBeneficialToPropagateTransposeToConst
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x25600xf32>
func.func @NotBeneficialToPropagateTransposeToConst(%arg0: tensor<1x25600xf32>) -> tensor<1x64xf32> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    %cst_1 = const.Declare tensor<64x200x1xf32> = dense<2.500000e+01>  : tensor<64x200x1xf32>
    %cst_2 = const.Declare tensor<64x200x1xf32> = dense<3.500000e+01>  : tensor<64x200x1xf32>
    %cst_3 = const.Declare tensor<64x200x128xf32> = dense<4.500000e+01>  : tensor<64x200x128xf32>
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<64x200x128xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<64x200x1xf32>, tensor<64x200x1xf32> -> tensor<64x200x128xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1]], shape_value = [64, 25600]} : tensor<64x200x128xf32> -> tensor<64x25600xf32>
    %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x25600xf32>, tensor<64x25600xf32> -> tensor<1x64xf32>

    return %2 : tensor<1x64xf32>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG: [[CST0:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    // CHECK-DAG: [[CST1:%.*]] = const.Declare tensor<64x200x1xf32> = dense<2.500000e+01> : tensor<64x200x1xf32>
    // CHECK-DAG: [[CST2:%.*]] = const.Declare tensor<64x200x1xf32> = dense<3.500000e+01> : tensor<64x200x1xf32>
    // CHECK-DAG: [[CST3:%.*]] = const.Declare tensor<64x200x128xf32> = dense<4.500000e+01> : tensor<64x200x128xf32>

    // CHECK:     [[FQ:%.*]]  = IE.FakeQuantize([[CST3]], [[CST]], [[CST0]], [[CST1]], [[CST2]])
    // CHECK:     [[RESHAPE:%.*]] = IE.AffineReshape([[FQ]])
    // CHECK:     [[FULLCONNECTED:%.*]] = IE.FullyConnected([[INPUT]], [[RESHAPE]])
    // CHECK:     return [[FULLCONNECTED]] : tensor<1x64xf32>
}
