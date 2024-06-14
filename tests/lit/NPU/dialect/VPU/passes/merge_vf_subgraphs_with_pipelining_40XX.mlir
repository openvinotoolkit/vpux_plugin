//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true" %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @VerticalFuseConvSoftmaxConv(
    %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
    %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %cst_1 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

    %0 = VPU.VerticalFusion (
      %arg0 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %cst_0 as %arg6: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 8, 1, 1]}
            -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg4, %arg5, %arg6) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
          rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    %1 = VPU.VerticalFusion (
      %0 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 8, 1]}
            -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.SoftMax(%arg4) {
          axisInd = 1 : i64,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
            -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    %2 = VPU.VerticalFusion (
      %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
      %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>,
      %cst_1 as %arg6: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 15, 1]}
            -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg4, %arg5, %arg6) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
          rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %2 : tensor<1x48x1024x4xf16, {order = #NHWC}>

    //CHECK: [[VF:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 1, 19, 1]
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution
    //CHECK:    [[SOFTMAX:%.+]] = VPU.SoftMax
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution

    //CHECK: return [[VF]]
}
