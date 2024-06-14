//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-fake-quant-has-negative-scales %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @HandleFakeQuantHasNegativeScales
func.func @HandleFakeQuantHasNegativeScales() -> tensor<4x3x1x1xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_low = const.Declare tensor<4x1x1x1xf16> =
                    dense<[[[[-1.536700e+00]]], [[[2.70509982]]], [[[1.71450007]]], [[[-4.025900e+00]]]]>
                    : tensor<4x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_high = const.Declare tensor<4x1x1x1xf16> =
                    dense<[[[[1.536700e+00]]], [[[-2.70509982]]], [[[-1.71450007]]], [[[4.025900e+00]]]]>
                    : tensor<4x1x1x1xf32>, [#const.ConvertElemType<f16>]

    %const_input = const.Declare tensor<4x3x1x1xf16> =
                    dense<[[[[-5.600000e+01]], [[9.600000e+01]], [[-3.200000e+01]]],
                           [[[8.000000e+00]], [[4.000000e+00]], [[-5.400000e+01]]],
                           [[[-8.800000e+01]], [[6.800000e+01]], [[-7.400000e+01]]],
                           [[[5.400000e+01]], [[-2.600000e+01]], [[3.400000e+01]]]]>
                    : tensor<4x3x1x1xf32>, [#const.ConvertElemType<f16>]

    %0 = IE.FakeQuantize(%const_input, %input_low, %input_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} :
            tensor<4x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<4x1x1x1xf16>, tensor<4x1x1x1xf16> -> tensor<4x3x1x1xf16>

    return %0 : tensor<4x3x1x1xf16>

    // CHECK:               [[INLOW:%.+]] = const.Declare tensor<1x1x1x1xf16> =
    // CHECK-SAME:              dense<-1.270000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK:               [[INHIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> =
    // CHECK-SAME:              dense<1.270000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]

    // CHECK:               [[CONSTIN:%.+]] = const.Declare tensor<4x3x1x1xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[-5.600000e+01]], [[9.600000e+01]], [[-3.200000e+01]]],
    // CHECK-SAME{LITERAL}:             [[[-8.000000e+00]], [[-4.000000e+00]], [[5.400000e+01]]],
    // CHECK-SAME{LITERAL}:             [[[8.800000e+01]], [[-6.800000e+01]], [[7.400000e+01]]],
    // CHECK-SAME{LITERAL}:             [[[5.400000e+01]], [[-2.600000e+01]], [[3.400000e+01]]]]>
    // CHECK-SAME:              : tensor<4x3x1x1xf32>, [#const.ConvertElemType<f16>]

    // CHECK:               [[OUTLOW:%.+]] = const.Declare tensor<4x1x1x1xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[-1.536700e+00]]], [[[-2.70509982]]], [[[-1.71450007]]], [[[-4.025900e+00]]]]>
    // CHECK-SAME:              : tensor<4x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK:               [[OUTHIGH:%.+]] = const.Declare tensor<4x1x1x1xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[1.536700e+00]]], [[[2.70509982]]], [[[1.71450007]]], [[[4.025900e+00]]]]>
    // CHECK-SAME:              : tensor<4x1x1x1xf32>, [#const.ConvertElemType<f16>]


    // CHECK:               [[FAKEQUANT:%.+]] = IE.FakeQuantize([[CONSTIN]], [[INLOW]], [[INHIGH]], [[OUTLOW]], [[OUTHIGH]])
    // CHECK-SAME:              {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64}
    // CHECK-SAME:              tensor<4x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<4x1x1x1xf16>, tensor<4x1x1x1xf16> -> tensor<4x3x1x1xf16>

    // CHECK:               return [[FAKEQUANT]] : tensor<4x3x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleFakeQuantHasNegativeScalesWithBroadCastConst
func.func @HandleFakeQuantHasNegativeScalesWithBroadCastConst() -> tensor<4x3x1x1xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_low = const.Declare tensor<4x1x1x1xf16> = dense<1.320000e-02>
                    : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.Broadcast<0 : i64, 4 : i64>, #const.ConvertElemType<f16>]
    %output_high = const.Declare tensor<4x1x1x1xf16> = dense<-1.320000e-02>
                    : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.Broadcast<0 : i64, 4 : i64>, #const.ConvertElemType<f16>]

    %const_input = const.Declare tensor<4x3x1x1xf16> =
                    dense<[[[[-5.600000e+01]], [[9.600000e+01]], [[-3.200000e+01]]],
                           [[[8.000000e+00]], [[4.000000e+00]], [[-5.400000e+01]]],
                           [[[-8.800000e+01]], [[6.800000e+01]], [[-7.400000e+01]]],
                           [[[5.400000e+01]], [[-2.600000e+01]], [[3.400000e+01]]]]>
                    : tensor<4x3x1x1xf32>, [#const.ConvertElemType<f16>]

    %0 = IE.FakeQuantize(%const_input, %input_low, %input_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} :
            tensor<4x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<4x1x1x1xf16>, tensor<4x1x1x1xf16> -> tensor<4x3x1x1xf16>

    return %0 : tensor<4x3x1x1xf16>

    // CHECK:               [[INLOW:%.+]] = const.Declare tensor<1x1x1x1xf16> =
    // CHECK-SAME:              dense<-1.270000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK:               [[INHIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> =
    // CHECK-SAME:              dense<1.270000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]

    // CHECK:               [[CONSTIN:%.+]] = const.Declare tensor<4x3x1x1xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[5.600000e+01]], [[-9.600000e+01]], [[3.200000e+01]]],
    // CHECK-SAME{LITERAL}:             [[[-8.000000e+00]], [[-4.000000e+00]], [[5.400000e+01]]],
    // CHECK-SAME{LITERAL}:             [[[8.800000e+01]], [[-6.800000e+01]], [[7.400000e+01]]],
    // CHECK-SAME{LITERAL}:             [[[-5.400000e+01]], [[2.600000e+01]], [[-3.400000e+01]]]]>
    // CHECK-SAME:              : tensor<4x3x1x1xf32>, [#const.ConvertElemType<f16>]

    // CHECK:               [[OUTLOW:%.+]] = const.Declare tensor<4x1x1x1xf16> = dense<-1.320000e-02>
    // CHECK-SAME:              : tensor<4x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK:               [[OUTHIGH:%.+]] = const.Declare tensor<4x1x1x1xf16> = dense<1.320000e-02>
    // CHECK-SAME:              : tensor<4x1x1x1xf32>, [#const.ConvertElemType<f16>]

    // CHECK:               [[FAKEQUANT:%.+]] = IE.FakeQuantize([[CONSTIN]], [[INLOW]], [[INHIGH]], [[OUTLOW]], [[OUTHIGH]])
    // CHECK-SAME:              {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64}
    // CHECK-SAME:              tensor<4x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<4x1x1x1xf16>, tensor<4x1x1x1xf16> -> tensor<4x3x1x1xf16>

    // CHECK:               return [[FAKEQUANT]] : tensor<4x3x1x1xf16>
}
