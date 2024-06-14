//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-operations %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @DoNotOptimizeSigmoidReorder
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16, {order = #NHCW}>
func.func @DoNotOptimizeSigmoidReorder(%arg0: tensor<1x16x32x32xf16, {order = #NHCW}>) -> tensor<1x16x32x32xf16> {
   %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16>
   %1 = IE.Sigmoid(%0) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
   return %1 : tensor<1x16x32x32xf16>

   // CHECK:        [[REORDER:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16>
   // CHECK:        [[SIGMOID:%.+]] = IE.Sigmoid([[REORDER]]) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
}
