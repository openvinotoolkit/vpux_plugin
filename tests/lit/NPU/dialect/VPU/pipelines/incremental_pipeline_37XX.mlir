//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --incremental-pipeline %s | FileCheck %s
// REQUIRES: arch-VPUX37XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @MaxpoolIncrementalPipelineCheck(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) { kernel_size = [1, 1], pad =  #VPU.Padding<bottom = 0, left = 0, right = 0, top = 0>, strides = [1, 1]} -> tensor<1x16x1x4xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>
}

    //CHECK: [[OP0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x16x1x4xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:   [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x16x1x4xf16, {order = #NHWC}> -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES0]]
    //CHECK: }

    //CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[OP0]] as %arg1: tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:   [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES3]]
    //CHECK: }

    //CHECK: [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OP1]] as %arg1: tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    //CHECK:   [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x1x4xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES4]]
    //CHECK: }

    //CHECK: return [[OUT]] : tensor<1x16x1x4xf16, {order = #NHWC}>


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @ConvIncrementalPipeline(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>


    //CHECK: [[CONST:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK: [[CONST_0:%.*]] =  const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK: [[OP0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES0]]
    //CHECK: }

    //CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[CONST_0]] as %arg1: tensor<80x64x3x3xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:   [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES1]]
    //CHECK: }

    //CHECK: [[OP2:%.*]] = VPU.NCE.ClusterTiling ([[CONST]] as %arg1: tensor<80x1x1x4xsi32>) -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:   [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32> -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:   VPU.Yield [[RES2]]
    //CHECK: }

    //CHECK: [[OP3:%.*]] = VPU.NCE.ClusterTiling ([[OP0]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[OP1]] as %arg2: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[OP2]] as %arg3: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK: [[OP4:%.*]] = VPU.NCE.ClusterTiling ([[OP3]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:   [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES4]]
    //CHECK: }

    //CHECK: return [[OP4]] : tensor<1x80x28x28xf16, {order = #NHWC}>

}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @TanHIncrementalPipeline(%arg0: tensor<1x1x16x4xf16, {order = #NHWC}>) -> tensor<1x1x16x4xf16, {order = #NHWC}> {
    %1 = VPU.Tanh(%arg0) : tensor<1x1x16x4xf16, {order = #NHWC}> -> tensor<1x1x16x4xf16, {order = #NHWC}>
    return %1 : tensor<1x1x16x4xf16, {order = #NHWC}>
}

    //CHECK: [[OP0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x1x16x4xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x16x4xf16, {order = #NHWC}> -> tensor<1x1x16x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES0]]
    //CHECK: }
    //CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[OP0]] as %arg1: tensor<1x1x16x4xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x1x16x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   [[RES1:%.*]] = VPU.Tanh(%arg1) : tensor<1x1x16x4xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1x16x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES1]]
    //CHECK: }
    //CHECK: [[OP2:%.*]] = VPU.NCE.ClusterTiling ([[OP1]] as %arg1: tensor<1x1x16x4xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x1x16x4xf16, {order = #NHWC}> {
    //CHECK:   [[RES2:%.*]] = VPU.Copy(%arg1) : tensor<1x1x16x4xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1x16x4xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES2]]
    //CHECK: }
    //CHECK: return [[OP2]] : tensor<1x1x16x4xf16, {order = #NHWC}>

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEPermuteIncrementalPipeline
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x3x64x64xf16>
func.func @NCEPermuteIncrementalPipeline(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x16x32x32x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x1x48xf16>,
                [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %nce_permute = VPU.NCE.Permute(%arg0)
            {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64}
    -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%nce_permute, %weights, %wt)
            {cm_sp_pattern = 1 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 1, 3, 3], strides = [2, 2]}
    -> tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    return %compress_conv : tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
    //CHECK:        [[WT:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    //CHECK:        [[IN_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:[^:]+]]: tensor<1x3x64x64xf16>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x3x64x64xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:           pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x3x64x64xf16> -> tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling ([[IN_CMX]] as [[ARG1:[^:]+]]: tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:           pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}> {
    //CHECK:            VPU.NCE.Permute([[ARG1]]) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64} -> tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[COPY_TO_DDR:%.*]] = VPU.NCE.ClusterTiling ([[NCE_PERMUTE]] as [[ARG1:[^:]+]]: tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x4x64x64x!qElemType, {order = #NHWC}> {
    //CHECK:            VPU.Copy([[ARG1]]) : tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>

    //CHECK:        [[COPY_TO_CMX:%.*]] = VPU.NCE.ClusterTiling ([[COPY_TO_DDR]] as [[ARG1:[^:]+]]: tensor<1x4x64x64x!qElemType, {order = #NHWC}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:           pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x4x64x64x!qElemType, {order = #NHWC}> -> tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling (%cst as [[ARG1:[^:]+]]: tensor<16x1x1x48x!qElemType, {order = #NHWC}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<16x1x1x48x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x48x!qElemType, {order = #NHWC}> -> tensor<16x1x1x48x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WT_CMX:%.*]] = VPU.NCE.ClusterTiling (%cst_0 as [[ARG1:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[COMPRESS_CONV:%.*]] = VPU.NCE.ClusterTiling ([[COPY_TO_CMX]] as [[ARG1:[^:]+]]: tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:           [[WEIGHTS_CMX]] as [[ARG2:[^:]+]]: tensor<16x1x1x48x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:           [[WT_CMX]] as [[ARG3:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            VPU.NCE.CompressConvolution([[ARG1]], [[ARG2]], [[ARG3]])
    //CHECK-SAME:           {cm_sp_pattern = 1 : i64, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    //CHECK-SAME:           rawFilterShape = [16, 1, 3, 3], strides = [2, 2]}
    //CHECK-SAME:           -> tensor<1x16x32x32x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[COMPRESS_CONV]] as [[ARG1:[^:]+]]: tensor<1x16x32x32x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x16x32x32x!qElemType, {order = #NHWC}> {
    //CHECK:            VPU.Copy([[ARG1]]) : tensor<1x16x32x32x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    //CHECK:    return [[OUT]] : tensor<1x16x32x32x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @ConvTanHIncrementalPipeline(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    %1 = VPU.Tanh(%0) : tensor<1x80x28x28xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK: [[CONST:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK: [[CONST_0:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK: [[OP0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES0]]
    //CHECK: }

    //CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[CONST_0]] as %arg1: tensor<80x64x3x3xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:   [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES1]]
    //CHECK: }

    //CHECK: [[OP2:%.*]] = VPU.NCE.ClusterTiling ([[CONST]] as %arg1: tensor<80x1x1x4xsi32>) -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:   [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32> -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:   VPU.Yield [[RES2]]
    //CHECK: }

    //CHECK: [[OP3:%.*]] = VPU.NCE.ClusterTiling ([[OP0]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[OP1]] as %arg2: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[OP2]] as %arg3: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES3]]
    //CHECK: }

    //CHECK: [[OP4:%.*]] = VPU.NCE.ClusterTiling ([[OP3]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:   [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES4]]
    //CHECK: }

    //CHECK: [[OP5:%.*]] = VPU.NCE.ClusterTiling ([[OP4]] as %arg1: tensor<1x80x28x28xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   [[RES5:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x80x28x28xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES5]]
    //CHECK: }

    //CHECK: [[OP6:%.*]] = VPU.NCE.ClusterTiling ([[OP5]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   [[RES6:%.*]] = VPU.Tanh(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES6]]
    //CHECK: }

    //CHECK: [[OP7:%.*]] = VPU.NCE.ClusterTiling ([[OP6]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:   [[RES7:%.*]] = VPU.Copy(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES7]]
    //CHECK: }

    //CHECK: return [[OP7]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAssignedSOHWithOddWidthAndSmallHeight
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x4x331776xf16, {order = #NHWC}>)
func.func @EltwiseAssignedSOHWithOddWidthAndSmallHeight(%arg0: tensor<1x16x4x331776xf16, {order = #NHWC}>) -> tensor<1x16x4x16186xf16, {order = #NHWC}> {
    %eltwise1_input2 = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]
    %eltwise2_input2 = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]

    %eltwise1_input1 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    %eltwise1 = VPU.NCE.Eltwise(%eltwise1_input1, %eltwise1_input2) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    %eltwise2_input1 = VPU.Slice %arg0 [0, 0, 0, 8093] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    %eltwise2 = VPU.NCE.Eltwise(%eltwise2_input1, %eltwise2_input2) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    %concat = VPU.Concat(%eltwise1, %eltwise2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 8093]]} : tensor<1x16x4x8093xf16, {order = #NHWC}>, tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x16186xf16, {order = #NHWC}>
    return %concat : tensor<1x16x4x16186xf16, {order = #NHWC}>

    //CHECK-DAG: [[ELTWISE_INPUT2:%.*]] = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[SLICE_0:%.*]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    //CHECK: [[ELTWISE_0_COPY_0:%.*]] = VPU.NCE.ClusterTiling ([[SLICE_0]] as [[ARG1:[^:]+]]: tensor<1x16x4x8093xf16, {order = #NHWC}>)
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:         VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK: [[ELTWISE_0_COPY_1:%.*]] = VPU.NCE.ClusterTiling ([[ELTWISE_INPUT2]] as [[ARG1:[^:]+]]: tensor<1x16x4x8093xf16, {order = #NHWC}>)
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:         VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK: [[ELTWISE_0:%.*]]  = VPU.NCE.ClusterTiling ([[ELTWISE_0_COPY_0]] as [[ARG1:[^:]+]]: tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[ELTWISE_0_COPY_1]] as [[ARG2:[^:]+]]: tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:         VPU.NCE.Eltwise([[ARG1]], [[ARG2]]) {op_type = #VPU.eltwise_type<ADD>
    //CHECK:         -> tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK: [[ELTWISE_0_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ELTWISE_0]] as [[ARG1:[^:]+]]: tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x4x8093xf16, {order = #NHWC}> {
    //CHECK:    VPU.Copy([[ARG1]]) : tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    //CHECK: [[SLICE_1:%.*]] = VPU.Slice [[ARG0]] [0, 0, 0, 8093] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    //CHECK: [[ELTWISE_1_COPY_0:%.*]] = VPU.NCE.ClusterTiling ([[SLICE_1]] as [[ARG1:[^:]+]]: tensor<1x16x4x8093xf16, {order = #NHWC}>)
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:         VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK: [[ELTWISE_1_COPY_1:%.*]] = VPU.NCE.ClusterTiling ([[ELTWISE_INPUT2]] as [[ARG1:[^:]+]]: tensor<1x16x4x8093xf16, {order = #NHWC}>)
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:         VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK: [[ELTWISE_1:%.*]]  = VPU.NCE.ClusterTiling ([[ELTWISE_1_COPY_0]] as [[ARG1:[^:]+]]: tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[ELTWISE_1_COPY_1]] as [[ARG2:[^:]+]]: tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x4x8093xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:         VPU.NCE.Eltwise([[ARG1]], [[ARG2]]) {op_type = #VPU.eltwise_type<ADD>
    //CHECK:         -> tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK: [[ELTWISE_1_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ELTWISE_1]] as [[ARG1:[^:]+]]: tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x4x8093xf16, {order = #NHWC}> {
    //CHECK:    VPU.Copy([[ARG1]]) : tensor<1x16x4x8093xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    //CHECK: [[CONCAT:%.*]] = VPU.Concat([[ELTWISE_0_COPY_OUT]], [[ELTWISE_1_COPY_OUT]]) {
    //CHECK:     static_offsets = [
    //CHECK-SAME:   [0, 0, 0, 0], [0, 0, 0, 8093]
    //CHECK:     ]} : tensor<1x16x4x8093xf16, {order = #NHWC}>, tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x16186xf16, {order = #NHWC}>
    //CHECK: return [[CONCAT]] : tensor<1x16x4x16186xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @InterpolateIncrementalPipeline(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<1x2x540x1920xf16>) -> tensor<1x2x256x512xf16> {
func.func @InterpolateIncrementalPipeline(%arg0: tensor<1x2x540x1920xf16>) -> tensor<1x2x256x512xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [256, 512]} : tensor<1x2x540x1920xf16> -> tensor<1x2x256x512xf16>
    return %0 : tensor<1x2x256x512xf16>

    // CHECK:           %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [1, 1, 270, 1920] : tensor<1x2x540x1920xf16> to tensor<1x1x270x1920xf16>
    // CHECK:           %[[VAL_2:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_3:.*]]: tensor<1x1x270x1920xf16>) -> !VPU.DistributedTensor<1x1x270x1920xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 1, 135, 1920], [1, 1, 135, 1920]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 135, 0]], memory_shapes = {{\[\[}}1, 1, 135, 1920], [1, 1, 135, 1920]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 135, 0]]}> {
    // CHECK:             %[[VAL_4:.*]] = VPU.Copy(%[[VAL_3]]) {out_mem_space = @CMX_NN} : tensor<1x1x270x1920xf16> -> tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:             VPU.Yield %[[VAL_4]]
    // CHECK:           }
    // CHECK:           %[[VAL_5:.*]] = VPU.NCE.ClusterTiling (%[[VAL_2]] as %[[VAL_6:.*]]: tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            %[[VAL_7:.*]] = VPU.Interpolate(%[[VAL_6]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 2, 540, 1920], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 2, 256, 512], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.47407407407407409, 0.26666666666666666], sizes_attr = [256, 512], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:             VPU.Yield %[[VAL_7]]
    // CHECK:           }
    // CHECK:           %[[VAL_8:.*]] = VPU.NCE.ClusterTiling (%[[VAL_5]] as %[[VAL_9:.*]]: tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x128x512xf16> {
    // CHECK:             %[[VAL_10:.*]] = VPU.Copy(%[[VAL_9]]) : tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x128x512xf16>
    // CHECK:             VPU.Yield %[[VAL_10]]
    // CHECK:           }
    // CHECK:           %[[VAL_11:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 270, 0] [1, 1, 270, 1920] : tensor<1x2x540x1920xf16> to tensor<1x1x270x1920xf16>
    // CHECK:           %[[VAL_12:.*]] = VPU.NCE.ClusterTiling (%[[VAL_11]] as %[[VAL_13:.*]]: tensor<1x1x270x1920xf16>) -> !VPU.DistributedTensor<1x1x270x1920xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 1, 135, 1920], [1, 1, 135, 1920]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 135, 0]], memory_shapes = {{\[\[}}1, 1, 135, 1920], [1, 1, 135, 1920]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 135, 0]]}> {
    // CHECK:             %[[VAL_14:.*]] = VPU.Copy(%[[VAL_13]]) {out_mem_space = @CMX_NN} : tensor<1x1x270x1920xf16> -> tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:             VPU.Yield %[[VAL_14]]
    // CHECK:           }
    // CHECK:           %[[VAL_15:.*]] = VPU.NCE.ClusterTiling (%[[VAL_12]] as %[[VAL_16:.*]]: tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             %[[VAL_17:.*]] = VPU.Interpolate(%[[VAL_16]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 2, 540, 1920], initial_input_offset_attr = [0, 0, 270, 0], initial_output_dims_attr = [1, 2, 256, 512], initial_output_offset_attr = [0, 0, 128, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.47407407407407409, 0.26666666666666666], sizes_attr = [256, 512], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:             VPU.Yield %[[VAL_17]]
    // CHECK:           }
    // CHECK:           %[[VAL_18:.*]] = VPU.NCE.ClusterTiling (%[[VAL_15]] as %[[VAL_19:.*]]: tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x128x512xf16> {
    // CHECK:             %[[VAL_20:.*]] = VPU.Copy(%[[VAL_19]]) : tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x128x512xf16>
    // CHECK:             VPU.Yield %[[VAL_20]]
    // CHECK:           }
    // CHECK:           %[[VAL_21:.*]] = VPU.Slice %[[VAL_0]] [0, 1, 0, 0] [1, 1, 270, 1920] : tensor<1x2x540x1920xf16> to tensor<1x1x270x1920xf16>
    // CHECK:           %[[VAL_22:.*]] = VPU.NCE.ClusterTiling (%[[VAL_21]] as %[[VAL_23:.*]]: tensor<1x1x270x1920xf16>) -> !VPU.DistributedTensor<1x1x270x1920xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 1, 135, 1920], [1, 1, 135, 1920]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 135, 0]], memory_shapes = {{\[\[}}1, 1, 135, 1920], [1, 1, 135, 1920]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 135, 0]]}> {
    // CHECK:             %[[VAL_24:.*]] = VPU.Copy(%[[VAL_23]]) {out_mem_space = @CMX_NN} : tensor<1x1x270x1920xf16> -> tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:             VPU.Yield %[[VAL_24]]
    // CHECK:           }
    // CHECK:           %[[VAL_25:.*]] = VPU.NCE.ClusterTiling (%[[VAL_22]] as %[[VAL_26:.*]]: tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             %[[VAL_27:.*]] = VPU.Interpolate(%[[VAL_26]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 2, 540, 1920], initial_input_offset_attr = [0, 1, 0, 0], initial_output_dims_attr = [1, 2, 256, 512], initial_output_offset_attr = [0, 1, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.47407407407407409, 0.26666666666666666], sizes_attr = [256, 512], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:             VPU.Yield %[[VAL_27]]
    // CHECK:           }
    // CHECK:           %[[VAL_28:.*]] = VPU.NCE.ClusterTiling (%[[VAL_25]] as %[[VAL_29:.*]]: tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x128x512xf16> {
    // CHECK:             %[[VAL_30:.*]] = VPU.Copy(%[[VAL_29]]) : tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x128x512xf16>
    // CHECK:             VPU.Yield %[[VAL_30]]
    // CHECK:           }
    // CHECK:           %[[VAL_31:.*]] = VPU.Slice %[[VAL_0]] [0, 1, 270, 0] [1, 1, 270, 1920] : tensor<1x2x540x1920xf16> to tensor<1x1x270x1920xf16>
    // CHECK:           %[[VAL_32:.*]] = VPU.NCE.ClusterTiling (%[[VAL_31]] as %[[VAL_33:.*]]: tensor<1x1x270x1920xf16>) -> !VPU.DistributedTensor<1x1x270x1920xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 1, 135, 1920], [1, 1, 135, 1920]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 135, 0]], memory_shapes = {{\[\[}}1, 1, 135, 1920], [1, 1, 135, 1920]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 135, 0]]}> {
    // CHECK:             %[[VAL_34:.*]] = VPU.Copy(%[[VAL_33]]) {out_mem_space = @CMX_NN} : tensor<1x1x270x1920xf16> -> tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:             VPU.Yield %[[VAL_34]]
    // CHECK:           }
    // CHECK:           %[[VAL_35:.*]] = VPU.NCE.ClusterTiling (%[[VAL_32]] as %[[VAL_36:.*]]: tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             %[[VAL_37:.*]] = VPU.Interpolate(%[[VAL_36]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 2, 540, 1920], initial_input_offset_attr = [0, 1, 270, 0], initial_output_dims_attr = [1, 2, 256, 512], initial_output_offset_attr = [0, 1, 128, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.47407407407407409, 0.26666666666666666], sizes_attr = [256, 512], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x1x270x1920xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:             VPU.Yield %[[VAL_37]]
    // CHECK:           }
    // CHECK:           %[[VAL_38:.*]] = VPU.NCE.ClusterTiling (%[[VAL_35]] as %[[VAL_39:.*]]: tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x128x512xf16> {
    // CHECK:             %[[VAL_40:.*]] = VPU.Copy(%[[VAL_39]]) : tensor<1x1x128x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x128x512xf16>
    // CHECK:             VPU.Yield %[[VAL_40]]
    // CHECK:           }
    // CHECK:           %[[VAL_41:.*]] = VPU.Concat(%[[VAL_8]], %[[VAL_18]], %[[VAL_28]], %[[VAL_38]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 128, 0], [0, 1, 0, 0], [0, 1, 128, 0]]} : tensor<1x1x128x512xf16>, tensor<1x1x128x512xf16>, tensor<1x1x128x512xf16>, tensor<1x1x128x512xf16> -> tensor<1x2x256x512xf16>
    // CHECK:           return %[[VAL_41]] : tensor<1x2x256x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSOK(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
func.func @ReduceSumSOK(%arg0: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32> -> tensor<1x1024x1x1xf32>
  %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 1024]} : tensor<1x1024x1x1xf32> -> tensor<1x1024xf32>
  return %1 : tensor<1x1024xf32>

// CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf32>) -> !VPU.DistributedTensor<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
// CHECK:    %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf32> -> tensor<1x1024x7x7xf32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:    VPU.Yield %[[VAL_3]]
// CHECK:   }
// CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
// CHECK:    %[[VAL_6:.*]] = VPU.ReduceSum(%[[VAL_5]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:    VPU.Yield %[[VAL_6]]
// CHECK:   }
// CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1024x1x1xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1024x1x1xf32> {
// CHECK:    %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1024x1x1xf32, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf32>
// CHECK:    VPU.Yield %[[VAL_9]]
// CHECK:   }
// CHECK:   %[[VAL_10:.*]] = VPU.AffineReshape(%[[VAL_7]]) {dim_mapping = {{\[\[}}0], [1], [1], [1]], shape_value = [1, 1024]} : tensor<1x1024x1x1xf32> -> tensor<1x1024xf32>
// CHECK:   return %[[VAL_10]] : tensor<1x1024xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSOH(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x16x32x64xf32>) -> tensor<1x1x32x1xf32> {
func.func @ReduceSumSOH(%arg0: tensor<1x16x32x64xf32>) -> tensor<1x1x32x1xf32> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x16x32x64xf32> -> tensor<1x1x32x1xf32>
  return %0 : tensor<1x1x32x1xf32>

// CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x16x32x64xf32>) -> !VPU.DistributedTensor<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
// CHECK:    %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x16x32x64xf32> -> tensor<1x16x32x64xf32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:    VPU.Yield %[[VAL_3]]
// CHECK:   }
// CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x16x32x64xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
// CHECK:    %[[VAL_6:.*]] = VPU.ReduceSum(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x16x32x64xf32, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x32x1xf32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:    VPU.Yield %[[VAL_6]]
// CHECK:   }
// CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x32x1xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x32x1xf32> {
// CHECK:    %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x32x1xf32, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x32x1xf32>
// CHECK:    VPU.Yield %[[VAL_9]]
// CHECK:   }
// CHECK:   return %[[VAL_7]] : tensor<1x1x32x1xf32>
}
