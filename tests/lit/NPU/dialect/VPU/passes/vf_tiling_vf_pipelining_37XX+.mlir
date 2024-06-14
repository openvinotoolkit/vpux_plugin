//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --vertical-fusion-tiling="enable-vertical-fusion-pipelining=true" %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ReorderVFPipelinePattern(
    %arg0: tensor<1x48x256x16xf16, {order = #NHWC}>,
    %arg1: tensor<256x48x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<48x256x1x1xf16, {order = #NHWC}>) -> tensor<1x48x256x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.VerticalFusion (
            %arg0 as %arg3: tensor<1x48x256x16xf16, {order = #NHWC}>,
            %arg1 as %arg4: tensor<256x48x1x1xf16, {order = #NHWC}>,
            %cst_0 as %arg5: tensor<256x1x1x4xsi32>,
            %arg2 as %arg6: tensor<48x256x1x1xf16, {order = #NHWC}>,
            %cst as %arg7: tensor<48x1x1x4xsi32>
            ) attributes {tilingStrategy = [1, 1, 2, 1]}
                -> tensor<1x48x256x16xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
      rawFilterShape = [256, 48, 1, 1], strides = [1, 1]} -> tensor<1x256x256x16xf16, {order = #NHWC}>
      %2 = VPU.SoftMax(%1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x256x256x16xf16, {order = #NHWC}> -> tensor<1x256x256x16xf16, {order = #NHWC}>
      %3 = VPU.NCE.Convolution(%2, %arg6, %arg7)
            {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [48, 256, 1, 1], strides = [1, 1]} -> tensor<1x48x256x16xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   return %0: tensor<1x48x256x16xf16, {order = #NHWC}>

   //CHECK: [[SLICE_TILE0:%.+]] = VPU.Slice %arg0
   //CHECK-SAME{LITERAL}: [0, 0, 0, 0] [1, 48, 128, 16]
   //CHECK: [[CONV_0_TILE0:%.+]] = VPU.NCE.Convolution([[SLICE_TILE0]], %arg1, %cst_0)
   //CHECK: [[SOFTMAX_TILE0:%.+]] = VPU.SoftMax([[CONV_0_TILE0]])

   //CHECK: [[SLICE_TILE1:%.+]] = VPU.Slice %arg0
   //CHECK-SAME{LITERAL}: [0, 0, 128, 0] [1, 48, 128, 16]
   //CHECK: [[CONV_0_TILE1:%.+]] = VPU.NCE.Convolution([[SLICE_TILE1]], %arg1, %cst_0)
   //CHECK: [[SOFTMAX_TILE1:%.+]] = VPU.SoftMax([[CONV_0_TILE1]])

   //CHECK: [[CONV_1_TILE0:%.+]] = VPU.NCE.Convolution([[SOFTMAX_TILE0]], %arg2, %cst)
   //CHECK: [[CONV_1_TILE1:%.+]] = VPU.NCE.Convolution([[SOFTMAX_TILE1]], %arg2, %cst)
   //CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV_1_TILE0]], [[CONV_1_TILE1]])

   //CHECK: return [[CONCAT]] : tensor<1x48x256x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:254>:f16, 0.003937007874015748>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 5.000000e-01>

func.func @VfTilingWithSwish(%arg0: tensor<1x16x176x176x!quant.uniform<u8:f16, 0.14376571505677466:128>, {order = #NHWC}>, %cst_0: tensor<1x1x1x16xui8>, %cst_1: tensor<96x16x1x1x!qElemType1, {order = #NHWC}>, %cst_2: tensor<96x1x1x4xsi32>, %cst_3: tensor<96x16x1x1xf16, {order = #NHWC}>, %cst_4: tensor<96x1x1x4xsi32>) -> tensor<1x96x176x176x!qElemType2, {order = #NHWC}>  {
   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x176x176x!quant.uniform<u8:f16, 0.14376571505677466:128>, {order = #NHWC}>, %cst_0 as %arg2: tensor<1x1x1x16xui8>, %cst_1 as %arg3: tensor<96x16x1x1x!qElemType1, {order = #NHWC}>, %cst_2 as %arg4: tensor<96x1x1x4xsi32>, %cst_3 as %arg5: tensor<96x16x1x1xf16, {order = #NHWC}>, %cst_4 as %arg6: tensor<96x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x96x176x176x!qElemType2, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg3, %arg4)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
         fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}>

      %2 = VPU.Swish(%1)
         {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x176x176xf16, {order = #NHWC}> -> tensor<1x96x176x176xf16, {order = #NHWC}>

      %3 = VPU.NCE.DepthConvolution(%2, %arg5, %arg6, %arg2)
         {activation_window_channel_length = 4 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
         fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x176x176x!qElemType2, {order = #NHWC}>

      VPU.Yield %3
   }

   return %0 : tensor<1x96x176x176x!qElemType2, {order = #NHWC}>

   // CHECK: [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 44, 176] : tensor<1x16x176x176x!qElemType, {order = #NHWC}> to tensor<1x16x44x176x!qElemType, {order = #NHWC}>
   // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]], %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]} -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[SWISH0:%.+]] = VPU.Swish([[CONV0]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x44x176xf16, {order = #NHWC}> -> tensor<1x96x44x176xf16, {order = #NHWC}>

   // CHECK: [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 0, 44, 0] [1, 16, 44, 176] : tensor<1x16x176x176x!qElemType, {order = #NHWC}> to tensor<1x16x44x176x!qElemType, {order = #NHWC}>

   // CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]} -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[SWISH1:%.+]] = VPU.Swish([[CONV1]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x44x176xf16, {order = #NHWC}> -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[DEPTHCONV0:%.+]] = VPU.NCE.DepthConvolution([[SWISH0]], %arg4, %arg5, %arg1 ) {activation_window_channel_length = 4 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x44x176x!qElemType2, {order = #NHWC}>

   // CHECK: [[SLICE2:%.+]] = VPU.Slice %arg0 [0, 0, 88, 0] [1, 16, 44, 176] : tensor<1x16x176x176x!qElemType, {order = #NHWC}> to tensor<1x16x44x176x!qElemType, {order = #NHWC}>

   // CHECK: [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]} -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[SWISH2:%.+]] = VPU.Swish([[CONV2]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x44x176xf16, {order = #NHWC}> -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[DEPTHCONV1:%.+]] = VPU.NCE.DepthConvolution([[SWISH1]], %arg4, %arg5, %arg1 ) {activation_window_channel_length = 4 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x44x176x!qElemType2, {order = #NHWC}>


   // CHECK: [[SLICE3:%.+]] = VPU.Slice %arg0 [0, 0, 132, 0] [1, 16, 44, 176] : tensor<1x16x176x176x!qElemType, {order = #NHWC}> to tensor<1x16x44x176x!qElemType, {order = #NHWC}>
   // CHECK: [[CONV3:%.+]] = VPU.NCE.Convolution([[SLICE3]], %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]} -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[SWISH3:%.+]] = VPU.Swish([[CONV3]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x44x176xf16, {order = #NHWC}> -> tensor<1x96x44x176xf16, {order = #NHWC}>
   // CHECK: [[DEPTHCONV2:%.+]] = VPU.NCE.DepthConvolution([[SWISH2]], %arg4, %arg5, %arg1 ) {activation_window_channel_length = 4 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x44x176x!qElemType2, {order = #NHWC}>

   // CHECK: [[DEPTHCONV3:%.+]] = VPU.NCE.DepthConvolution([[SWISH3]], %arg4, %arg5, %arg1 ) {activation_window_channel_length = 4 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x44x176x!qElemType2, {order = #NHWC}>

   // CHECK: [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV0]], [[DEPTHCONV1]], [[DEPTHCONV2]], [[DEPTHCONV3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 44, 0], [0, 0, 88, 0], [0, 0, 132, 0]]} : tensor<1x96x44x176x!qElemType2, {order = #NHWC}>, tensor<1x96x44x176x!qElemType2, {order = #NHWC}>, tensor<1x96x44x176x!qElemType2, {order = #NHWC}>, tensor<1x96x44x176x!qElemType2, {order = #NHWC}> -> tensor<1x96x176x176x!qElemType2, {order = #NHWC}>
   // CHECK: return [[CONCAT]] : tensor<1x96x176x176x!qElemType2, {order = #NHWC}>
}
