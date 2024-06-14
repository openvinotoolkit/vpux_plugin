//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dequantize-const %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

!qElemType = !quant.uniform<u8:f32:0, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @PerAxis
func.func @PerAxis() -> tensor<4x1x1x1xf32> {
    %0 = const.Declare tensor<4x1x1x1x!qElemType> =
        dense<129> : tensor<4x1x1x1xui8>, [#const.QuantCast<!qElemType>]
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<4x1x1x1x!qElemType> -> tensor<4x1x1x1xf32>
    return %1 : tensor<4x1x1x1xf32>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare
    // CHECK-SAME:      dense<129> : tensor<4x1x1x1xui8
    // CHECK-SAME:      #const.QuantCast<!qElemType>
    // CHECK-SAME:      #const.Dequantize

    // CHECK:       return [[CST]]
}
