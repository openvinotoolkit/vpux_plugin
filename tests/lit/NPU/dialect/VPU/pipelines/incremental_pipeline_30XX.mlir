//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --incremental-pipeline %s | FileCheck %s
// REQUIRES: arch-VPUX30XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @MaxpoolIncrementalPipelineCheck(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst1 = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    %0 = VPU.NCE.MaxPool(%arg0, %cst0, %cst1) { kernel_size = [1, 1],  pad = #VPU.Padding<bottom = 0, left = 0, right = 0, top = 0>, strides = [1, 1], activation_window_channel_length = 4} -> tensor<1x16x1x4xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>
}

    //CHECK: [[CONST:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    //CHECK: [[CONST_0:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    //CHECK: [[OP0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x16x1x4xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:   [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x16x1x4xf16, {order = #NHWC}> -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES0]]
    //CHECK: }
    //CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[CONST]] as %arg1: tensor<16x1x1x4xsi32>) -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:   [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:   VPU.Yield [[RES1]]
    //CHECK: }
    //CHECK: [[OP2:%.*]] = VPU.NCE.ClusterTiling ([[CONST_0]] as %arg1: tensor<1x1x1x16xui8>) -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:   [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:   VPU.Yield [[RES2]]
    //CHECK: }
    //CHECK: [[OP3:%.*]] = VPU.NCE.ClusterTiling ([[OP0]] as %arg1: tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[OP1]] as %arg2: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>, [[OP2]] as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:   [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1, %arg2 , %arg3 ) {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES3]]
    //CHECK: }
    //CHECK: [[OP4:%.*]] = VPU.NCE.ClusterTiling ([[OP3]] as %arg1: tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    //CHECK:   [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x1x4xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES4]]
    //CHECK: }
    //CHECK: return [[OP4]] : tensor<1x16x1x4xf16, {order = #NHWC}>

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @ConvIncrementalPipeline(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) { pad = #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>
}

    //CHECK: [[CONST:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK: [[CONST_0:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[OP0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:   [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES0]]
    //CHECK: }
    //CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[CONST_0]] as %arg1: tensor<80x64x3x3xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:   [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES1]]
    //CHECK: }
    //CHECK: [[OP2:%.*]] = VPU.NCE.ClusterTiling ([[CONST]] as %arg1: tensor<80x1x1x4xsi32>) -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:   [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32> -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:   VPU.Yield [[RES2]]
    //CHECK: }
    //CHECK: [[OP3:%.*]] = VPU.NCE.ClusterTiling ([[OP0]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[OP1]] as %arg2: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[OP2]] as %arg3: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:   [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    //CHECK-SAME: {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES3]]
    //CHECK: }
    //CHECK: [[OP4:%.*]] = VPU.NCE.ClusterTiling ([[OP3]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:   [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES4]]
    //CHECK: }
    //CHECK: return [[OP4]] : tensor<1x80x28x28xf16, {order = #NHWC}>


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @TanHIncrementalPipeline(%arg0: tensor<1x1x16x4xf16, {order = #NHWC}>) -> tensor<1x1x16x4xf16, {order = #NHWC}> {
    %1 = VPU.Tanh(%arg0) : tensor<1x1x16x4xf16, {order = #NHWC}> -> tensor<1x1x16x4xf16, {order = #NHWC}>
    return %1 : tensor<1x1x16x4xf16, {order = #NHWC}>
}

//CHECK: [[OP0:%.*]] = VPU.Tanh(%arg0) : tensor<1x1x16x4xf16, {order = #NHWC}> -> tensor<1x1x16x4xf16, {order = #NHWC}>
//CHECK: return [[OP0]] : tensor<1x1x16x4xf16, {order = #NHWC}>
