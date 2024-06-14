//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --cmx-concat --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsType = tensor<
    128x256x3x3xf16,
    {mem_space = @CMX_NN, order = #NHWC}
>

!WeightsTableType = tensor<
    128x1x1x4xsi32,
    {mem_space = @CMX_NN, order = #NCHW}
>

!SparseInput = !VPU.SparseTensor<
    data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!SparseOutput = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPU.DistributedTensor<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
>

!SparseConvOutput = !VPU.SparseTensor<
    data=tensor<1x128x14x14xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x128x14x14xi1, {order = #NHWC}>
>

!SparseConvOutputCMX = !VPU.SparseTensor<
    data=tensor<1x128x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x128x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!SparseConvOutputDist = !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
>

!SparseConcatOutput = !VPU.SparseTensor<
    data=tensor<1x256x14x14xf16, {order = #NHWC}>,
    sparsity_map=tensor<1x256x14x14xi1, {order = #NHWC}>
>

!SparseConcatOutputCMX = !VPU.SparseTensor<
    data=tensor<1x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x256x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

// CHECK:      func.func @SparseConvolution([[INPUT:%.+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
// CHECK-SAME:                         [[WEIGHTS_TABLE:%.+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
// CHECK-SAME:                         [[WEIGHTS:%.+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME:         -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>> {
func.func @SparseConvolution(%input: !SparseInput,
                        %weightsTable: !WeightsTableType,
                        %weights: !WeightsType)
        -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>> {

    // Convolution 0
    %0 = VPU.NCE.ClusterTiling (
        %input as %arg1: !SparseInput,
        %weights as %arg2: !WeightsType,
        %weightsTable as %arg3: !WeightsTableType) -> !SparseConvOutputDist {
        %20 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
                {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                rawFilterShape = [128, 256, 3, 3], strides = [2, 2]}
                    -> !SparseConvOutputCMX
        VPU.Yield %20
    }

    // Convolution 0 output copy
    %1 = VPU.NCE.ClusterTiling (%0 as %arg1: !SparseConvOutputCMX) -> !SparseConvOutput {
            %20 = VPU.Copy(%arg1) : !SparseConvOutputCMX -> !SparseConvOutput
        VPU.Yield %20
    }

    // Convolution 1
    %2 = VPU.NCE.ClusterTiling (
        %input as %arg1: !SparseInput,
        %weights as %arg2: !WeightsType,
        %weightsTable as %arg3: !WeightsTableType)  -> !SparseConvOutputDist {
        %20 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
                {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                rawFilterShape = [128, 256, 3, 3], strides = [2, 2]}
                    -> !SparseConvOutputCMX
        VPU.Yield %20
    }

    // Convolution 1 output copy
    %3 = VPU.NCE.ClusterTiling (%2 as %arg1: !SparseConvOutputCMX) -> !SparseConvOutput {
            %20 = VPU.Copy(%arg1) : !SparseConvOutputCMX -> !SparseConvOutput
        VPU.Yield %20
    }

    // Concat
    %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : !SparseConvOutput, !SparseConvOutput -> !SparseConcatOutput

    // Concat output copy
    %5 = VPU.NCE.ClusterTiling (%4 as %arg1: !SparseConcatOutput) -> !SparseOutput {
        %20 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : !SparseConcatOutput -> !SparseConcatOutputCMX
        VPU.Yield %20
    }

    // Convolution 2
    %6 = VPU.NCE.ClusterTiling (
        %5 as %arg1: !SparseConcatOutputCMX,
        %weights as %arg2: !WeightsType,
        %weightsTable as %arg3: !WeightsTableType)
            -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x6x6xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPU.DistributedTensor<1x128x6x6xi1, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>> {
        %20 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
                {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                rawFilterShape = [128, 256, 3, 3], strides = [2, 2]} -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>
        VPU.Yield %20
    }
    %7 = VPU.NCE.ClusterTiling (
        %6 as %arg1: !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>)
            -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>> {
        %20 = VPU.Copy(%arg1) : !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>> -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>
        VPU.Yield %20
    }

    return %7 : !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<1x128x6x6xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>>

    // Convolution 0
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:                                         [[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                         [[WEIGHTS_TABLE]] as [[INNER_ARG2:[^:]+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:           VPU.NCE.Convolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])

    // Convolution 1
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:                                         [[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                         [[WEIGHTS_TABLE]] as [[INNER_ARG2:[^:]+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:           VPU.NCE.Convolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])

    // DistributedCast for Conv0
    // CHECK:       [[DISTR_CAST0:%.+]] = VPU.DistributedCast([[CONV0]] :
    // CHECK-SAME:         !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME                                  {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // DistributedCast for Conv1
    // CHECK:       [[DISTR_CAST1:%.+]] = VPU.DistributedCast([[CONV1]] :
    // CHECK-SAME:         !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME                                  {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x128x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Concat
    // CHECK:       [[CONCAT_CMX:%.+]] = VPU.Concat([[DISTR_CAST0]], [[DISTR_CAST1]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x256x14x14xf16, #NHWC, @CMX_NN,
    // CHECK-SAME                                  {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x256x14x14xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                                 {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Convolution 2
    // CHECK:       [[CONV2:%.+]] = VPU.NCE.ClusterTiling ([[CONCAT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x256x14x14xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:                                         [[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<128x256x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                         [[WEIGHTS_TABLE]] as [[INNER_ARG2:[^:]+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x128x6x6xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x128x6x6xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>> {
    // CHECK:           VPU.NCE.Convolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])

    // CHECK:       [[COPY_OUT:%.+]] = VPU.NCE.ClusterTiling ([[CONV2]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:          -> !VPU.SparseTensor<data=tensor<1x128x6x6xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x6x6xi1, {order = #NHWC}>> {
    // CHECK:           VPU.Copy([[INNER_ARG0]])

    // CHECK:       return [[COPY_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ConcatInputsDistributed = !VPU.DistributedTensor<
    1x640x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!ConcatUserDistributed = !VPU.DistributedTensor<
    1x1280x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: func.func @ConcatUsersHaveDifferentOperandNums
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x640x16x16xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x640x16x16xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x640x16x16xf16, {order = #NHWC}>, [[INPUT3:%.+]]: tensor<1x640x16x16xf16, {order = #NHWC}>, [[INPUT4:%.+]]: tensor<1x1280x16x16xf16, {order = #NHWC}>)
func.func @ConcatUsersHaveDifferentOperandNums(
            %input0: tensor<1x640x16x16xf16, {order = #NHWC}>,
            %input1: tensor<1x640x16x16xf16, {order = #NHWC}>,
            %input2: tensor<1x640x16x16xf16, {order = #NHWC}>,
            %input3: tensor<1x640x16x16xf16, {order = #NHWC}>,
            %input4: tensor<1x1280x16x16xf16, {order = #NHWC}>)
           -> (tensor<1x1280x16x16xf16, {order = #NHWC}>, tensor<1x1280x16x16xf16, {order = #NHWC}>) {
    // One input of Concat
    %0 = VPU.NCE.ClusterTiling (%input0 as %arg0: tensor<1x640x16x16xf16, {order = #NHWC}>) -> !ConcatInputsDistributed {
        %20 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}> -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %1 = VPU.NCE.ClusterTiling (%input1 as %arg0: tensor<1x640x16x16xf16, {order = #NHWC}>) -> !ConcatInputsDistributed {
        %20 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}> -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %2 = VPU.NCE.ClusterTiling (%0 as %arg0: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                %1 as %arg1: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                                -> !ConcatInputsDistributed {
        %20 = VPU.NCE.Eltwise(%arg0, %arg1) {
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
        } -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %3 = VPU.NCE.ClusterTiling (%2 as %arg0: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x640x16x16xf16, {order = #NHWC}> {
        %20 = VPU.Copy(%arg0) : tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x640x16x16xf16, {order = #NHWC}>
        VPU.Yield %20
    }

    // Another input of Concat
    %4 = VPU.NCE.ClusterTiling (%input2 as %arg0: tensor<1x640x16x16xf16, {order = #NHWC}>) -> !ConcatInputsDistributed {
        %20 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}> -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %5 = VPU.NCE.ClusterTiling (%input3 as %arg0: tensor<1x640x16x16xf16, {order = #NHWC}>) -> !ConcatInputsDistributed {
        %20 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}> -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %6 = VPU.NCE.ClusterTiling (%4 as %arg0: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                %5 as %arg1: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                                -> !ConcatInputsDistributed {
        %20 = VPU.NCE.Eltwise(%arg0, %arg1) {
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
        } -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %7 = VPU.NCE.ClusterTiling (%6 as %arg0: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x640x16x16xf16, {order = #NHWC}> {
        %20 = VPU.Copy(%arg0) : tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x640x16x16xf16, {order = #NHWC}>
        VPU.Yield %20
    }

    // Concat
    %8 = VPU.Concat(%3, %7) {static_offsets = [[0, 0, 0, 0], [0, 640, 0, 0]]} :
        tensor<1x640x16x16xf16, {order = #NHWC}>,
        tensor<1x640x16x16xf16, {order = #NHWC}> -> tensor<1x1280x16x16xf16, {order = #NHWC}>

    // One user is MaxPool
    %9 = VPU.NCE.ClusterTiling (%8 as %arg0: tensor<1x1280x16x16xf16, {order = #NHWC}>) -> !ConcatUserDistributed {
        %20 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x1280x16x16xf16, {order = #NHWC}> -> tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %10 = VPU.NCE.ClusterTiling (%9 as %arg0: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !ConcatUserDistributed {
        %20 = VPU.NCE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]
        } -> tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %11 = VPU.NCE.ClusterTiling (%10 as %arg0: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x1280x16x16xf16, {order = #NHWC}> {
        %20 = VPU.Copy(%arg0) : tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1280x16x16xf16, {order = #NHWC}>
        VPU.Yield %20
    }

    // Another user is Eltwise
    %12 = VPU.NCE.ClusterTiling (%8 as %arg0: tensor<1x1280x16x16xf16, {order = #NHWC}>) -> !ConcatUserDistributed {
        %20 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x1280x16x16xf16, {order = #NHWC}> -> tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %13 = VPU.NCE.ClusterTiling (%input4 as %arg0: tensor<1x1280x16x16xf16, {order = #NHWC}>) -> !ConcatUserDistributed {
        %20 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x1280x16x16xf16, {order = #NHWC}> -> tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %14 = VPU.NCE.ClusterTiling (%13 as %arg0: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                 %12 as %arg1: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !ConcatUserDistributed {
        %20 = VPU.NCE.Eltwise(%arg0, %arg1) {
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
        } -> tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %20
    }
    %15 = VPU.NCE.ClusterTiling (%14 as %arg0: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x1280x16x16xf16, {order = #NHWC}> {
        %20 = VPU.Copy(%arg0) : tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1280x16x16xf16, {order = #NHWC}>
        VPU.Yield %20
    }

    return %11, %15 : tensor<1x1280x16x16xf16, {order = #NHWC}>, tensor<1x1280x16x16xf16, {order = #NHWC}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg5: tensor<1x640x16x16xf16, {order = #NHWC}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:                      -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as %arg5: tensor<1x640x16x16xf16, {order = #NHWC}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:                      -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:                      [[COPY_IN_0]] as %arg5: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                      [[COPY_IN_1]] as %arg6: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_IN_2:%.+]] = VPU.NCE.ClusterTiling ([[INPUT2]] as %arg5: tensor<1x640x16x16xf16, {order = #NHWC}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:                      -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_IN_3:%.+]] = VPU.NCE.ClusterTiling ([[INPUT3]] as %arg5: tensor<1x640x16x16xf16, {order = #NHWC}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x640x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:                      -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[ELTWISE_1:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:                      [[COPY_IN_2]] as %arg5: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                      [[COPY_IN_3]] as %arg6: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0]], [[ELTWISE_1]]) {
    // CHECK-SAME:                          static_offsets = [
    // CHECK-SAME:                              [0, 0, 0, 0],
    // CHECK-SAME:                              [0, 640, 0, 0]
    // CHECK-SAME:                          ]
    // CHECK-SAME:                      } : !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                          !VPU.DistributedTensor<1x640x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x1280x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[USER_MAX_POOL:%.+]] = VPU.NCE.ClusterTiling ([[CONCAT]] as %arg5: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x1280x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.NCE.MaxPool(%arg5) {
    // CHECK-SAME:                      kernel_size = [1, 1],
    // CHECK-SAME:                      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]}
    // CHECK-SAME:                      -> tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_OUT_0:%.+]] = VPU.NCE.ClusterTiling ([[USER_MAX_POOL]] as %arg5: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:                      -> tensor<1x1280x16x16xf16, {order = #NHWC}> {
    // CHECK:           VPU.Copy(%arg5) : tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:                      -> tensor<1x1280x16x16xf16, {order = #NHWC}>
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_IN_4:%.+]] = VPU.NCE.ClusterTiling ([[INPUT4]] as %arg5: tensor<1x1280x16x16xf16, {order = #NHWC}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x1280x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x1280x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:                      -> tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[USER_ELTWISE:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:                      [[COPY_IN_4]] as %arg5: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                      [[CONCAT]] as %arg6: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x1280x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.NCE.Eltwise(%arg5, %arg6) {
    // CHECK-SAME:                      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:                      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:                      -> tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_OUT_1:%.+]] = VPU.NCE.ClusterTiling ([[USER_ELTWISE]] as %arg5: tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:                      -> tensor<1x1280x16x16xf16, {order = #NHWC}> {
    // CHECK:           VPU.Copy(%arg5) : tensor<1x1280x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:                      -> tensor<1x1280x16x16xf16, {order = #NHWC}>
    // CHECK:           VPU.Yield
    // CHECK:       }

    // CHECK:       return [[COPY_OUT_0]], [[COPY_OUT_1]] : tensor<1x1280x16x16xf16, {order = #NHWC}>, tensor<1x1280x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = !VPU.DistributedTensor<
    1x16x90x160xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!Distributed2 = !VPU.DistributedTensor<
    1x32x90x160xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @SkipCMXConcatForNCEPermute
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x16x90x160xf16, {order = #NHWC}>,
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x16x90x160xf16, {order = #NHWC}>)
func.func @SkipCMXConcatForNCEPermute(%arg0: tensor<1x16x90x160xf16, {order = #NHWC}>,
           %arg1: tensor<1x16x90x160xf16, {order = #NHWC}>)
           -> tensor<1x32x90x160xf16, {order = #NHWC}> {
    %maxPoolWeightsTable = const.Declare tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    %activationWindow = const.Declare tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>

    %maxPoolWeightsTable1 = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    %activationWindow1 = const.Declare tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}> =
        dense<1> : tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>

    // Input 1 of Concat
    %0 = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x16x90x160xf16, {order = #NHWC}>) -> !Distributed {
        %21 = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x90x160xf16, {order = #NHWC}> -> tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }
    %1 = VPU.NCE.ClusterTiling (
        %0 as %arg2: tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %maxPoolWeightsTable as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !Distributed {
        %21 = VPU.NCE.MaxPool(%arg2, %arg3, %arg4) {
                activation_window_channel_length = 4 : i64,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }
    %2 = VPU.NCE.ClusterTiling (%1 as %arg2: tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x90x160xf16, {order = #NHWC}> {
        %21 = VPU.Copy(%arg2) : tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x90x160xf16, {order = #NHWC}>
        VPU.Yield %21
    }

    // Input 2 of Concat
    %3 = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x16x90x160xf16>) -> !Distributed {
        %21 = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x90x160xf16> -> tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NCHW}>
        VPU.Yield %21
    }

    %4 = VPU.NCE.ClusterTiling (%3 as %arg2: tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !Distributed {
        %21 = VPU.NCE.Permute(%arg2) {
            dstElemType = f16,
            dstOrder = #NHWC,
            expandedChannels = 16 : i64,
            ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        } -> tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }
    %5 = VPU.NCE.ClusterTiling (%4 as %arg2: tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x90x160xf16, {order = #NHWC}> {
        %21 = VPU.Copy(%arg2) : tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x90x160xf16, {order = #NHWC}>
        VPU.Yield %21
    }

    %6 = VPU.Concat(%2, %5) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x90x160xf16, {order = #NHWC}>, tensor<1x16x90x160xf16, {order = #NHWC}> -> tensor<1x32x90x160xf16, {order = #NHWC}>

    // Concat output
    %7 = VPU.NCE.ClusterTiling (%6 as %arg2: tensor<1x32x90x160xf16, {order = #NHWC}>) -> !Distributed2 {
        %21 = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x90x160xf16, {order = #NHWC}> -> tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %8 = VPU.NCE.ClusterTiling (
        %7 as %arg2: tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %maxPoolWeightsTable1 as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow1 as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !Distributed2 {
        %21 = VPU.NCE.MaxPool(%arg2, %arg3, %arg4) {
                activation_window_channel_length = 4 : i64,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %9 = VPU.NCE.ClusterTiling (%8 as %arg2: tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x90x160xf16, {order = #NHWC}> {
        %21 = VPU.Copy(%arg2) : tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x90x160xf16, {order = #NHWC}>
        VPU.Yield %21
    }

    return %9 : tensor<1x32x90x160xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:       [[CST_1:%.+]] = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<1> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       [[COPY_IN_0:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[ARG0:%[^:]+]]: tensor<1x16x90x160xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x16x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x90x160xf16, {order = #NHWC}> -> tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:               VPU.Yield
    // CHECK:       }

    // CHECK:       [[MAXPOOL_0:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:                          [[COPY_IN_0]] as [[ARG1:%[^:]+]]: tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                          [[CST]] as [[ARG2:%[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
    // CHECK-SAME:                          [[CST_0]] as [[ARG3:%[^:]+]]: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x16x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPU.NCE.MaxPool([[ARG1]], [[ARG2]] , [[ARG3]] ) {
    // CHECK-SAME:                  activation_window_channel_length = 4 : i64,
    // CHECK-SAME:                  kernel_size = [1, 1],
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:               VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_OUT_0:%.+]] = VPU.NCE.ClusterTiling ([[MAXPOOL_0]] as [[ARG4:%[^:]+]]: tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x90x160xf16, {order = #NHWC}> {
    // CHECK:               VPU.Copy([[ARG4]]) : tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x90x160xf16, {order = #NHWC}>
    // CHECK:               VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_IN_1:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[ARG5:%[^:]+]]: tensor<1x16x90x160xf16>) -> !VPU.DistributedTensor<1x16x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPU.Copy([[ARG5]]) {out_mem_space = @CMX_NN} : tensor<1x16x90x160xf16> -> tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:               VPU.Yield
    // CHECK:       }

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.ClusterTiling ([[COPY_IN_1]] as [[ARG6:%[^:]+]]: tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x16x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPU.NCE.Permute([[ARG6]]) {
    // CHECK-SAME:                  dstElemType = f16,
    // CHECK-SAME:                  dstOrder = #NHWC,
    // CHECK-SAME:                  expandedChannels = 16 : i64,
    // CHECK-SAME:                  ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:                  lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
    // CHECK-SAME:          } -> tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:               VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_OUT_1:%.+]] = VPU.NCE.ClusterTiling ([[NCE_PERMUTE]] as [[ARG7:%[^:]+]]: tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x90x160xf16, {order = #NHWC}> {
    // CHECK:               VPU.Copy([[ARG7]]) : tensor<1x16x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x90x160xf16, {order = #NHWC}>
    // CHECK:               VPU.Yield
    // CHECK:       }

    // CHECK:       [[CMX_CONCAT:%.+]] = VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]]) {
    // CHECK-SAME:          static_offsets = [
    // CHECK-SAME:              [0, 0, 0, 0],
    // CHECK-SAME:              [0, 16, 0, 0]
    // CHECK-SAME:          ]
    // CHECK-SAME:  } : tensor<1x16x90x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x16x90x160xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x90x160xf16, {order = #NHWC}>

    // CHECK:       [[COPY_IN_2:%.+]] = VPU.NCE.ClusterTiling ([[CMX_CONCAT]] as [[ARG8:%[^:]+]]: tensor<1x32x90x160xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x32x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPU.Copy([[ARG8]]) {out_mem_space = @CMX_NN} : tensor<1x32x90x160xf16, {order = #NHWC}> -> tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:               VPU.Yield
    // CHECK:       }

    // CHECK:       [[MAXPOOL_1:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:                          [[COPY_IN_2]] as [[ARG9:%[^:]+]]: tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                          %cst_1 as [[ARG10:%[^:]+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
    // CHECK-SAME:                          %cst_0 as [[ARG11:%[^:]+]]: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                      -> !VPU.DistributedTensor<1x32x90x160xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:               VPU.NCE.MaxPool([[ARG9]], [[ARG10]] , [[ARG11]] ) {
    // CHECK-SAME:                  activation_window_channel_length = 4 : i64,
    // CHECK-SAME:                  kernel_size = [1, 1],
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:               VPU.Yield
    // CHECK:       }

    // CHECK:       [[COPY_OUT_2:%.+]] = VPU.NCE.ClusterTiling ([[MAXPOOL_1]] as [[ARG12:%[^:]+]]: tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x90x160xf16, {order = #NHWC}> {
    // CHECK:               VPU.Copy([[ARG12]]) : tensor<1x32x90x160xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x90x160xf16, {order = #NHWC}>
    // CHECK:               VPU.Yield
    // CHECK:       }

    // CHECK:       return [[COPY_OUT_2]] : tensor<1x32x90x160xf16, {order = #NHWC}>
}
