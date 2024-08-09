//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --lower-IE-to-VPU %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @Function_0 {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter_68" : tensor<1x18x3x3xf32>
  } outputsInfo : {
    DataInfo "Relu_70" : tensor<1x18x3x3xf32>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}>) -> tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}>
  func.func @main(%arg0: tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}>) -> tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}> {
    %0 = IE.ReLU(%arg0) : tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}> -> tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}>
    return %0 : tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}>

    // CHECK:  [[ReLU:%.*]] = VPU.ReLU([[ARG0]]) : tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}> -> tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}>
    // CHECK:  return [[ReLU]] : tensor<1x?x3x3xf32, {bounds = [1, 18, 3, 3], order = #NCHW}>
  }
}
