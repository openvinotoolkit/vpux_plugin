//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=512 --init-compiler="vpu-arch=%arch%" --handle-u16-fake-quantize="enable-handle-u16-fake-quantize=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// CHECK-LABEL: func.func @NoReplaceFQU16WithReLU
// CHECK-SAME:        [[INPUT:%arg0]]: tensor<1x4x640x640xf16>
func.func @NoReplaceFQU16WithReLU(%arg0: tensor<1x4x640x640xf16>) -> tensor<1x4x640x640xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<57.1374702> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x4x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x640x640xf16>
    %1 = IE.Sigmoid(%0) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>
    return %1 : tensor<1x4x640x640xf16>

    // CHECK: [[SIGMOID:%.*]] = IE.Sigmoid([[INPUT]]) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>

    // CHECK: return [[SIGMOID]]
}
