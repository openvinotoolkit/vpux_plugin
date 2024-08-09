//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%  allow-custom-values=true" --incremental-pipeline %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxpoolIncrementalPipelineCheck
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
func.func @MaxpoolIncrementalPipelineCheck(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) { kernel_size = [1, 1], pad =  #VPU.Padding<bottom = 0, left = 0, right = 0, top = 0>, strides = [1, 1]} -> tensor<1x16x1x4xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>


    //CHECK: [[OP0:%.*]] =  VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:   [[RES0:%.*]] = VPU.Copy([[IN_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x4xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES0]]
    //CHECK: }

    //CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[OP0]] as [[IN_ARG1:[^:]+]]: tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4], [1, 16, 1, 4]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:   [[RES3:%.*]] = VPU.NCE.MaxPool([[IN_ARG1]])
    //CHECK-SAME:   {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]}
    //CHECK-SAME:       -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES3]]
    //CHECK: }

    //CHECK: [[OP4:%.*]] = VPU.NCE.ClusterTiling ([[OP1]] as [[IN_ARG2:[^:]+]]: tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    //CHECK:   [[RES4:%.*]] = VPU.Copy([[IN_ARG2]]) : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x1x4xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES4]]
    //CHECK: }

    //CHECK: return [[OP4]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvIncrementalPipeline
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @ConvIncrementalPipeline(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>


    //CHECK: [[CONST:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK: [[CONST_0:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK: [[OP0:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}> {
    //CHECK:   [[RES0:%.*]] = VPU.Copy([[IN_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES0]]
    //CHECK: }

    //CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[CONST_0]] as [[IN_ARG1:[^:]+]]: tensor<80x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:   [[RES1:%.*]] = VPU.Copy([[IN_ARG1]]) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES1]]
    //CHECK: }

    //CHECK: [[OP2:%.*]] = VPU.NCE.ClusterTiling ([[CONST]] as [[IN_ARG2:[^:]+]]: tensor<80x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:   [[RES2:%.*]] = VPU.Copy([[IN_ARG2]]) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32>
    //CHECK-SAME:   -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:   VPU.Yield [[RES2]]
    //CHECK: }

    //CHECK: [[OP3:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:  [[OP0]] as [[IN_ARG3:[^:]+]]: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:  [[OP1]] as [[IN_ARG4:[^:]+]]: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:  [[OP2]] as [[IN_ARG5:[^:]+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}> {
    //CHECK:   [[RES3:%.*]] = VPU.NCE.Convolution([[IN_ARG3]], [[IN_ARG4]], [[IN_ARG5]])
    //CHECK-SAME:       {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:        rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:   VPU.Yield [[RES3]]
    //CHECK: }

    //CHECK: [[OP4:%.*]] = VPU.NCE.ClusterTiling ([[OP3]] as [[IN_ARG6:[^:]+]]: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:   [[RES4:%.*]] = VPU.Copy([[IN_ARG6]]) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES4]]
    //CHECK: }

    //CHECK: return [[OP4]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

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
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x3x64x64xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 10, 64], [1, 3, 10, 64]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 10, 64], [1, 3, 10, 64]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}> {
    //CHECK:            VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x3x64x64xf16> -> tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling ([[IN_CMX]] as [[ARG1:[^:]+]]: tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 10, 64], [1, 4, 10, 64]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 4, 13, 64], [1, 4, 14, 64], [1, 4, 13, 64], [1, 4, 12, 64], [1, 4, 11, 64], [1, 4, 10, 64]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}> {
    //CHECK:            VPU.NCE.Permute([[ARG1]]) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64} -> tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[COPY_TO_DDR:%.*]] = VPU.NCE.ClusterTiling ([[NCE_PERMUTE]] as [[ARG1:[^:]+]]: tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x4x64x64x!qElemType, {order = #NHWC}> {
    //CHECK:            VPU.Copy([[ARG1]]) : tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>

    //CHECK:        [[COPY_TO_CMX:%.*]] = VPU.NCE.ClusterTiling ([[COPY_TO_DDR]] as [[ARG1:[^:]+]]: tensor<1x4x64x64x!qElemType, {order = #NHWC}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x64x64x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 11, 64], [1, 4, 10, 64], [1, 4, 10, 64]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 4, 13, 64], [1, 4, 14, 64], [1, 4, 13, 64], [1, 4, 12, 64], [1, 4, 11, 64], [1, 4, 10, 64]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}> {
    //CHECK:            VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x4x64x64x!qElemType, {order = #NHWC}> -> tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling (%cst as [[ARG1:[^:]+]]: tensor<16x1x1x48x!qElemType, {order = #NHWC}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<16x1x1x48x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48], [16, 1, 1, 48]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:            VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x48x!qElemType, {order = #NHWC}> -> tensor<16x1x1x48x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WT_CMX:%.*]] = VPU.NCE.ClusterTiling (%cst_0 as [[ARG1:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:            VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[COMPRESS_CONV:%.*]] = VPU.NCE.ClusterTiling ([[COPY_TO_CMX]] as [[ARG1:[^:]+]]: tensor<1x4x64x64x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:           [[WEIGHTS_CMX]] as [[ARG2:[^:]+]]: tensor<16x1x1x48x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:           [[WT_CMX]] as [[ARG3:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}> {
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

// CHECK-LABEL: @TanHIncrementalPipeline
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x1x16x4xf16, {order = #NHWC}>)
func.func @TanHIncrementalPipeline(%arg0: tensor<1x1x16x4xf16, {order = #NHWC}>) -> tensor<1x1x16x4xf16, {order = #NHWC}> {
    %1 = VPU.Tanh(%arg0) : tensor<1x1x16x4xf16, {order = #NHWC}> -> tensor<1x1x16x4xf16, {order = #NHWC}>
    return %1 : tensor<1x1x16x4xf16, {order = #NHWC}>
}

    //CHECK: [[OP0:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x1x16x4xf16, {order = #NHWC}>)
    //CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[OP0]] as [[IN_ARG1:[^:]+]]: tensor<1x1x16x4xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK: [[OP2:%.*]] = VPU.NCE.ClusterTiling ([[OP1]] as [[IN_ARG2:[^:]+]]: tensor<1x1x16x4xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x1x16x4xf16, {order = #NHWC}> {
    //CHECK:   [[RES2:%.*]] = VPU.Copy([[IN_ARG2]]) : tensor<1x1x16x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:   -> tensor<1x1x16x4xf16, {order = #NHWC}>
    //CHECK:   VPU.Yield [[RES2]]
    //CHECK: }

    //CHECK: return [[OP2]] : tensor<1x1x16x4xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvTanHIncrementalPipeline
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @ConvTanHIncrementalPipeline(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    %1 = VPU.Tanh(%0) : tensor<1x80x28x28xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x80x28x28xf16, {order = #NHWC}>
}

	//CHECK: [[CONST:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
	//CHECK: [[CONST_0:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
	//CHECK: [[OP0:%.*]] =  VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]]
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]]
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}> {
	//CHECK:   [[RES0:%.*]] = VPU.Copy([[IN_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
	//CHECK:   VPU.Yield [[RES0]]
	//CHECK: }

	//CHECK: [[OP1:%.*]] = VPU.NCE.ClusterTiling ([[CONST_0]] as [[IN_ARG1:[^:]+]]: tensor<80x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
	//CHECK:   [[RES1:%.*]] = VPU.Copy([[IN_ARG1]]) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:      -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
	//CHECK:   VPU.Yield [[RES1]]
	//CHECK: }

	//CHECK: [[OP2:%.*]] = VPU.NCE.ClusterTiling ([[CONST]] as [[IN_ARG2:[^:]+]]: tensor<80x1x1x4xsi32>)
    //CHECK-SAME: -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
	//CHECK:   [[RES2:%.*]] = VPU.Copy([[IN_ARG2]]) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32>
    //CHECK-SAME:       -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
	//CHECK:   VPU.Yield [[RES2]]
	//CHECK: }

	//CHECK: [[OP3:%.*]] = VPU.NCE.ClusterTiling ([[OP0]] as [[IN_ARG3:[^:]+]]: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                                 [[OP1]] as [[IN_ARG4:[^:]+]]: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                                 [[OP2]] as [[IN_ARG5:[^:]+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}> {
	//CHECK:   [[RES3:%.*]] = VPU.NCE.Convolution([[IN_ARG3]], [[IN_ARG4]], [[IN_ARG5]])
    //CHECK-SAME:       {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:        rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
	//CHECK:   VPU.Yield [[RES3]]
	//CHECK: }

	//CHECK: [[OP4:%.*]] = VPU.NCE.ClusterTiling ([[OP3]] as [[IN_ARG6:[^:]+]]: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}> {
	//CHECK:   [[RES4:%.*]] = VPU.Copy([[IN_ARG6]]) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}>
	//CHECK:   VPU.Yield [[RES4]]
	//CHECK: }

	//CHECK: [[OP5:%.*]] = VPU.NCE.ClusterTiling ([[OP4]] as [[IN_ARG7:[^:]+]]: tensor<1x80x28x28xf16, {order = #NHWC}>)
	//CHECK: [[OP6:%.*]] = VPU.NCE.ClusterTiling ([[OP5]] as [[IN_ARG8:[^:]+]]: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
	//CHECK: [[OP7:%.*]] = VPU.NCE.ClusterTiling ([[OP6]] as [[IN_ARG9:[^:]+]]: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}> {
	//CHECK:   [[RES7:%.*]] = VPU.Copy([[IN_ARG9]]) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:   -> tensor<1x80x28x28xf16, {order = #NHWC}>
	//CHECK:   VPU.Yield [[RES7]]
	//CHECK: }

	//CHECK: return [[OP7]] : tensor<1x80x28x28xf16, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @InterpolateIncrementalPipeline(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<1x2x540x1920xf16>) -> tensor<1x2x256x512xf16> {
func.func @InterpolateIncrementalPipeline(%arg0: tensor<1x2x540x1920xf16>) -> tensor<1x2x256x512xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [256, 512]} : tensor<1x2x540x1920xf16> -> tensor<1x2x256x512xf16>
    return %0 : tensor<1x2x256x512xf16>

    // CHECK:   %[[VAL_1:.*]] = const.Declare tensor<1x1x1x512xsi32> = dense<
    // CHECK:   %[[VAL_2:.*]] = const.Declare tensor<1x1x1x1024xf16> = dense<

    // CHECK:   %[[VAL_3:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_4:.*]]: tensor<1x2x540x1920xf16>) -> !VPU.DistributedTensor<1x2x540x1920xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 2, 90, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 89, 1920], [1, 2, 88, 1920]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 90, 0], [0, 0, 181, 0], [0, 0, 272, 0], [0, 0, 363, 0], [0, 0, 452, 0]], memory_shapes = {{\[\[}}1, 2, 90, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 91, 1920], [1, 2, 89, 1920], [1, 2, 88, 1920]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 90, 0], [0, 0, 181, 0], [0, 0, 272, 0], [0, 0, 363, 0], [0, 0, 452, 0]]}> {
    // CHECK:     %[[VAL_5:.*]] = VPU.Copy(%[[VAL_4]]) {out_mem_space = @CMX_NN} : tensor<1x2x540x1920xf16> -> tensor<1x2x540x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:     VPU.Yield %[[VAL_5]]
    // CHECK:   }
    // CHECK:   %[[VAL_6:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_7:.*]]: tensor<1x1x1x512xsi32>) -> !VPU.DistributedTensor<1x1x1x512xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:     %[[VAL_8:.*]] = VPU.Copy(%[[VAL_7]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xsi32> -> tensor<1x1x1x512xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:     VPU.Yield %[[VAL_8]]
    // CHECK:   }
    // CHECK:   %[[VAL_9:.*]] = VPU.NCE.ClusterTiling (%[[VAL_2]] as %[[VAL_10:.*]]: tensor<1x1x1x1024xf16>) -> !VPU.DistributedTensor<1x1x1x1024xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:     %[[VAL_11:.*]] = VPU.Copy(%[[VAL_10]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1024xf16> -> tensor<1x1x1x1024xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:     VPU.Yield %[[VAL_11]]
    // CHECK:   }
    // CHECK:   %[[VAL_12:.*]] = VPU.NCE.ClusterTiling (%[[VAL_3]] as %[[VAL_13:.*]]: tensor<1x2x540x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>, %[[VAL_6]] as %[[VAL_14:.*]]: tensor<1x1x1x512xsi32, {mem_space = @CMX_NN, order = #NCHW}>, %[[VAL_9]] as %[[VAL_15:.*]]: tensor<1x1x1x1024xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x2x256x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 42, 512], [1, 2, 42, 512]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]], memory_shapes = {{\[\[}}1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 43, 512], [1, 2, 42, 512], [1, 2, 42, 512]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]]}> {
    // CHECK:     %[[VAL_16:.*]] = VPU.Interpolate(%[[VAL_13]], %[[VAL_14]], %[[VAL_15]]) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 2, 540, 1920], initial_output_dims_attr = [1, 2, 256, 512], operandSegmentSizes = array<i32: 1, 0, 0, 0, 1, 1>, scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [256, 512], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x2x540x1920xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x512xsi32, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1024xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x2x256x512xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:     VPU.Yield %[[VAL_16]]
    // CHECK:   }
    // CHECK:   %[[VAL_17:.*]] = VPU.NCE.ClusterTiling (%[[VAL_12]] as %[[VAL_18:.*]]: tensor<1x2x256x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x2x256x512xf16> {
    // CHECK:     %[[VAL_19:.*]] = VPU.Copy(%[[VAL_18]]) : tensor<1x2x256x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x2x256x512xf16>
    // CHECK:     VPU.Yield %[[VAL_19]]
    // CHECK:   }
    // CHECK:   return %[[VAL_17]] : tensor<1x2x256x512xf16>
    // CHECK:   }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSOK(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
func.func @ReduceSumSOK(%arg0: tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32> -> tensor<1x1024x1x1xf32>
  %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 1024]} : tensor<1x1024x1x1xf32> -> tensor<1x1024xf32>
  return %1 : tensor<1x1024xf32>

// CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf32>) -> !VPU.DistributedTensor<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}> {
// CHECK:    %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf32> -> tensor<1x1024x7x7xf32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:    VPU.Yield %[[VAL_3]]
// CHECK:   }
// CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}> {
// CHECK:    %[[VAL_6:.*]] = VPU.ReduceSum(%[[VAL_5]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf32, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:    VPU.Yield %[[VAL_6]]
// CHECK:   }
// CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1024x1x1xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1024x1x1xf32> {
// CHECK:    %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1024x1x1xf32, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf32>
// CHECK:    VPU.Yield %[[VAL_9]]
// CHECK:   }
// CHECK:   %[[VAL_10:.*]] = VPU.AffineReshape(%[[VAL_7]]) {dim_mapping = {{\[\[}}0], [1], [1], [1]], shape_value = [1, 1024]} : tensor<1x1024x1x1xf32> -> tensor<1x1024xf32>
// CHECK:   return %[[VAL_10]] : tensor<1x1024xf32>
// CHECK:   }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @ReduceSumSOH(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x16x32x64xf32>) -> tensor<1x1x32x1xf32> {
func.func @ReduceSumSOH(%arg0: tensor<1x16x32x64xf32>) -> tensor<1x1x32x1xf32> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x16x32x64xf32> -> tensor<1x1x32x1xf32>
  return %0 : tensor<1x1x32x1xf32>

// CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x16x32x64xf32>) -> !VPU.DistributedTensor<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 16, 6, 64], [1, 16, 6, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]], memory_shapes = {{\[\[}}1, 16, 6, 64], [1, 16, 6, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64], [1, 16, 5, 64]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}> {
// CHECK:    %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x16x32x64xf32> -> tensor<1x16x32x64xf32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:    VPU.Yield %[[VAL_3]]
// CHECK:   }
// CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x16x32x64xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 6, 1], [1, 1, 6, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]], memory_shapes = {{\[\[}}1, 1, 6, 1], [1, 1, 6, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1], [1, 1, 5, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}> {
// CHECK:    %[[VAL_6:.*]] = VPU.ReduceSum(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x16x32x64xf32, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x32x1xf32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:    VPU.Yield %[[VAL_6]]
// CHECK:   }
// CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x32x1xf32, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x32x1xf32> {
// CHECK:    %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x32x1xf32, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x32x1xf32>
// CHECK:    VPU.Yield %[[VAL_9]]
// CHECK:   }
// CHECK:   return %[[VAL_7]] : tensor<1x1x32x1xf32>
// CHECK:   }
}
