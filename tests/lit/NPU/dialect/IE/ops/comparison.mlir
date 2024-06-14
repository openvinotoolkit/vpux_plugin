//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @LessBroadcastable
func.func @LessBroadcastable(%arg0: tensor<10x1xf16>, %arg1: tensor<1x50xf16>) -> tensor<10x50xf16> {
    %0 = IE.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10x1xf16>, tensor<1x50xf16> -> tensor<10x50xf16>
    return %0 : tensor<10x50xf16>

    // CHECK:       %[[VAL0:.*]] =   IE.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10x1xf16>, tensor<1x50xf16> -> tensor<10x50xf16>
    // CHECK-NOT:   IE.Less
    // CHECK:       return %[[VAL0]]
}
