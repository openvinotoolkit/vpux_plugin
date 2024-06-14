//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

func.func @FuseFQ(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    %1 = IE.FakeQuantize(%0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>
    // CHECK-DAG:   %[[ILOW:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG:   %[[IHIGH:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%arg0, %[[ILOW]], %[[IHIGH]], %[[ILOW]], %[[IHIGH]])

    // CHECK-NOT:   IE.FakeQuantize

    // CHECK:       return %[[FQ]]
}

// -----

func.func @DoNotFuseFQ(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high_1 = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %input_high_2 = const.Declare tensor<f32> = dense<128.0> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high_1, %input_low, %input_high_1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    %1 = IE.FakeQuantize(%0, %input_low, %input_high_1, %input_low, %input_high_2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>
    // CHECK-DAG:   %[[ILOW:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG:   %[[IHIGH1:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG:   %[[IHIGH2:.*]] = const.Declare tensor<f32> = dense<1.280000e+02> : tensor<f32>

    // CHECK:   %[[FQ1:.*]] = IE.FakeQuantize(%arg0, %[[ILOW]], %[[IHIGH1]], %[[ILOW]], %[[IHIGH1]])

    // CHECK:   %[[FQ2:.*]] = IE.FakeQuantize(%[[FQ1]], %[[ILOW]], %[[IHIGH1]], %[[ILOW]], %[[IHIGH2]])

    // CHECK:       return %[[FQ2]]
}
