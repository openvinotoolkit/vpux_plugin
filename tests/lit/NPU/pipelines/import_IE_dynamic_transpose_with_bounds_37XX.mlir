//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=VPUX37XX --import-IE --set-upper-bounds="1 3 192 192" ./dynamic_transpose.xml -o %t
// RUN: FileCheck %s --input-file %t

// CHECK: module @dynamic_transpose {
// CHECK:   IE.CNNNetwork entryPoint : @main inputsInfo : {
// CHECK:       DataInfo "Parameter_18" : tensor<1x3x192x192xf32>
// CHECK:   } outputsInfo : {
// CHECK:       DataInfo "Transpose_21" : tensor<1x192x3x192xf32>
// CHECK:   }
// CHECK:   func.func @main([[ARG0:[^:]+]]: tensor<1x3x?x?xf32, {bounds = [1, 3, 192, 192], order = #NCHW}>)
// CHECK-SAME:      -> tensor<1x?x3x?xf32, {bounds = [1, 192, 3, 192], order = #NCHW}> {

// CHECK:       [[CONVERT_IN:%.*]] = IE.Convert([[ARG0]]) {
// CHECK-SAME:      dstElemType = f16
// CHECK-SAME:  } : tensor<1x3x?x?xf32, {bounds = [1, 3, 192, 192], order = #NCHW}>
// CHECK-SAME:      -> tensor<1x3x?x?xf16, {bounds = [1, 3, 192, 192], order = #NCHW}>

// CHECK:       [[ADD:%.*]] = IE.Add([[CONVERT_IN]], [[CONVERT_IN]]) {
// CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
// CHECK-SAME:  } : tensor<1x3x?x?xf16, {bounds = [1, 3, 192, 192], order = #NCHW}>,
// CHECK-SAME:      tensor<1x3x?x?xf16, {bounds = [1, 3, 192, 192], order = #NCHW}>
// CHECK-SAME:          -> tensor<1x3x?x?xf16, {bounds = [1, 3, 192, 192], order = #NCHW}>

// CHECK:       [[CONVERT_OUT:%.*]] = IE.Convert([[ADD]]) {
// CHECK-SAME:      dstElemType = f32
// CHECK-SAME:  } : tensor<1x3x?x?xf16, {bounds = [1, 3, 192, 192], order = #NCHW}>
// CHECK-SAME:      -> tensor<1x3x?x?xf32, {bounds = [1, 3, 192, 192], order = #NCHW}>

// CHECK:       [[CST:%.*]] = const.Declare tensor<4xsi64> = dense<[0, 2, 1, 3]> : tensor<4xsi64>

// CHECK:       [[TRANSPOSE:%.*]] = IE.Transpose([[CONVERT_OUT]], [[CST]]) :
// CHECK-SAME:      tensor<1x3x?x?xf32, {bounds = [1, 3, 192, 192], order = #NCHW}>,
// CHECK-SAME:      tensor<4xsi64>
// CHECK-SAME:          -> tensor<1x?x3x?xf32, {bounds = [1, 192, 3, 192], order = #NCHW}>

// CHECK:       return [[TRANSPOSE]] : tensor<1x?x3x?xf32, {bounds = [1, 192, 3, 192], order = #NCHW}>
// CHECK:   }
// CHECK: }
