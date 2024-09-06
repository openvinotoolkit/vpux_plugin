//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-mvn %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @NHWCTilingDecomposeMVN
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<3x1x1x2420001xf16, {order = #NHWC}>
func.func @NHWCTilingDecomposeMVN(%arg0: tensor<3x1x1x2420001xf16, {order = #NHWC}>) -> (tensor<3x1x1x2420001xf16, {order = #NHWC}>) {
      %0 = VPU.MVN(%arg0) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<3x1x1x2420001xf16, {order = #NHWC}> -> tensor<3x1x1x2420001xf16, {order = #NHWC}>
      return %0 : tensor<3x1x1x2420001xf16, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [3, 1, 1, 1210001] : tensor<3x1x1x2420001xf16, {order = #NHWC}> to tensor<3x1x1x1210001xf16, {order = #NHWC}>
    // CHECK:            [[RESHAPE_TILE_1:%.+]] = VPU.AffineReshape([[INPUT_TILE_1]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2], [3]], shape_value = [3, 1, 1210001, 1]} : tensor<3x1x1x1210001xf16, {order = #NHWC}> -> tensor<3x1x1210001x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[RESHAPE_TILE_1]])
    // CHECK-SAME:           :  tensor<3x1x1210001x1xf16, {order = #NHWC}> -> tensor<3x1x6x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1210001] [3, 1, 1, 1210000] : tensor<3x1x1x2420001xf16, {order = #NHWC}> to tensor<3x1x1x1210000xf16, {order = #NHWC}>
    // CHECK:            [[RESHAPE_TILE_2:%.+]] = VPU.AffineReshape([[INPUT_TILE_2]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2], [3]], shape_value = [3, 1, 1210000, 1]} : tensor<3x1x1x1210000xf16, {order = #NHWC}> -> tensor<3x1x1210000x1xf16, {order = #NHWC}>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[RESHAPE_TILE_2]])
    // CHECK-SAME:           :  tensor<3x1x1210000x1xf16, {order = #NHWC}> -> tensor<3x1x6x2xf32, {order = #NHWC}>

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]])
    // CHECK-SAME:           -> tensor<3x1x6x4xf32, {order = #NHWC}>

    // CHECK:            [[VAL3:%.+]] = VPU.MVN1MeanVar([[CONCAT]])
    // CHECK-SAME:           : tensor<3x1x6x4xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            [[VAL4:%.+]] = VPU.MVN1Normalize(%arg0, [[VAL3]])
    // CHECK-SAME:           :  tensor<3x1x1x2420001xf16, {order = #NHWC}>, tensor<3x1x1x2xf16, {order = #NHWC}> -> tensor<3x1x1x2420001xf16, {order = #NHWC}>

    // CHECK:            return [[VAL4]]
}

// -----

// CHECK-LABEL: func.func @NCHWTilingDecomposeMVN
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<3x1x1x420001xf16>
func.func @NCHWTilingDecomposeMVN(%arg0: tensor<3x1x1x420001xf16>) -> (tensor<3x1x1x420001xf16>) {
      %0 = VPU.MVN(%arg0) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<3x1x1x420001xf16> -> tensor<3x1x1x420001xf16>
      return %0 : tensor<3x1x1x420001xf16>

    // CHECK:            [[INPUT_TILE_1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [3, 1, 1, 210001] : tensor<3x1x1x420001xf16> to tensor<3x1x1x210001xf16>
    // CHECK:            [[RESHAPE_TILE_1:%.+]] = VPU.AffineReshape([[INPUT_TILE_1]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2], [3]], shape_value = [3, 1, 210001, 1]} : tensor<3x1x1x210001xf16> -> tensor<3x1x210001x1xf16>
    // CHECK:            [[TILE_1:%.+]] = VPU.MVN1SumOp([[RESHAPE_TILE_1]])
    // CHECK-SAME:           :  tensor<3x1x210001x1xf16> -> tensor<3x1x1x2xf32, {order = #NHWC}>

    // CHECK:            [[INPUT_TILE_2:%.+]] = VPU.Slice %arg0 [0, 0, 0, 210001] [3, 1, 1, 210000] : tensor<3x1x1x420001xf16> to tensor<3x1x1x210000xf16>
    // CHECK:            [[RESHAPE_TILE_2:%.+]] = VPU.AffineReshape([[INPUT_TILE_2]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2], [3]], shape_value = [3, 1, 210000, 1]} : tensor<3x1x1x210000xf16> -> tensor<3x1x210000x1xf16>
    // CHECK:            [[TILE_2:%.+]] = VPU.MVN1SumOp([[RESHAPE_TILE_2]])
    // CHECK-SAME:           :  tensor<3x1x210000x1xf16> -> tensor<3x1x1x2xf32, {order = #NHWC}>

    // CHECK:            [[CONCAT:%.+]] = VPU.Concat([[TILE_1]], [[TILE_2]])
    // CHECK-SAME:           -> tensor<3x1x1x4xf32, {order = #NHWC}>

    // CHECK:            [[VAL3:%.+]] = VPU.MVN1MeanVar([[CONCAT]])
    // CHECK-SAME:           : tensor<3x1x1x4xf32, {order = #NHWC}> -> tensor<3x1x1x2xf16, {order = #NHWC}>

    // CHECK:            [[VAL4:%.+]] = VPU.MVN1Normalize(%arg0, [[VAL3]])
    // CHECK-SAME:           :  tensor<3x1x1x420001xf16>, tensor<3x1x1x2xf16, {order = #NHWC}> -> tensor<3x1x1x420001xf16>

    // CHECK:            return [[VAL4]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @NoTilingSOHDecomposeMVN
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x515971xf16, {order = #NHWC}>
func.func @NoTilingSOHDecomposeMVN(%arg0: tensor<1x1x1x515971xf16, {order = #NHWC}>) -> (tensor<1x1x1x515971xf16, {order = #NHWC}>) {
      %0 = VPU.MVN(%arg0) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x1x1x515971xf16, {order = #NHWC}> -> tensor<1x1x1x515971xf16, {order = #NHWC}>
      return %0 : tensor<1x1x1x515971xf16, {order = #NHWC}>

    // CHECK:            [[RESHAPE:%.+]] = VPU.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 515971, 1]} : tensor<1x1x1x515971xf16, {order = #NHWC}> -> tensor<1x1x515971x1xf16, {order = #NHWC}>
    // CHECK:            [[VAL0:%.+]] = VPU.MVN1SumOp([[RESHAPE]])
    // CHECK-SAME:          output_height = 6 : i64
    // CHECK-SAME:          tensor<1x1x515971x1xf16, {order = #NHWC}> -> tensor<1x1x6x2xf32, {order = #NHWC}>
    // CHECK:            [[VAL1:%.+]] = VPU.MVN1MeanVar([[VAL0]])
    // CHECK:            [[VAL2:%.+]] = VPU.MVN1Normalize(%arg0, [[VAL1]])
    // CHECK:            return [[VAL2]]
}
