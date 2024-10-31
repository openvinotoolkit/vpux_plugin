//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-power-to-mult %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ConvertPowerWithExponent2ToMult
func.func @ConvertPowerWithExponent2ToMult(%arg0: tensor<1x16xf16>) -> tensor<1x16xf16> {
    %cst_exponent = const.Declare tensor<1x1xf16> = dense<2.0> : tensor<1x1xf16>
    %power = IE.Power(%arg0, %cst_exponent) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16xf16>, tensor<1x1xf16> -> tensor<1x16xf16>
    return %power : tensor<1x16xf16>

    // CHECK-NOT:   IE.Power
    // CHECK:       %[[VAL0:.*]] = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @RemoveSqrtPowerWithExponent2Pattern
func.func @RemoveSqrtPowerWithExponent2Pattern(%arg0: tensor<1x900x161xf16>) -> tensor<1x900x161xf16> {
    %cst = const.Declare tensor<1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %sqrt = IE.Sqrt(%arg0) : tensor<1x900x161xf16> -> tensor<1x900x161xf16>
    %power = IE.Power(%sqrt, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x900x161xf16>, tensor<1x1x1xf16> -> tensor<1x900x161xf16>
    return %power : tensor<1x900x161xf16>

    // CHECK-NOT:   IE.Power
    // CHECK:       return {{[^:]+}} : tensor<1x900x161xf16>
}
