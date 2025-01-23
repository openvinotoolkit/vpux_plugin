//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --make-ops-with-distributed-tensor --remove-output-sparse-to-avoid-suboptimal-dpu-workloads %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveOutputSparseForConvSOK
func.func @RemoveOutputSparseForConvSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x128x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_2 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [128, 128, 1, 1], strides = [1, 1]} -> !VPU.SparseTensor<data=tensor<1x128x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x28x28xi1, {order = #NHWC}>>
    %1 = VPU.NCE.Convolution(%0, %cst_2, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [128, 128, 1, 1], strides = [1, 1]} -> tensor<1x128x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x128x28x28xf16, {order = #NHWC}>

    //CHECK:    [[WEIGHTSTABLE_0:%.*]] = const.Declare tensor<128x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_0:%.*]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTSTABLE_1:%.*]] = const.Declare tensor<128x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_1:%.*]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>

    //CHECK:    [[UNROLLED_INPUT_0:%.*]] = VPU.UnrolledType(%arg0 : tensor<1x128x28x28xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTS_0:%.*]] = VPU.UnrolledType([[WEIGHTS_0]] : tensor<128x128x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTSTABLE_0:%.*]] = VPU.UnrolledType([[WEIGHTSTABLE_0]] : tensor<128x1x1x4xsi32>) -> !VPU.DistributedTensor<128x1x1x4xsi32

    //CHECK:        [[CONV0:%.*]] = VPU.NCE.Convolution([[UNROLLED_INPUT_0]], [[UNROLLED_WEIGHTS_0]], [[UNROLLED_WEIGHTSTABLE_0]])
    //CHECK:     [[UNROLLED_OUTPUT_0:%.*]] = VPU.UnrolledType([[CONV0]]
    //CHECK-SAME: -> !VPU.SparseTensor<data=tensor<1x128x28x28xf16, {order = #NHWC}>

    //CHECK:    [[UNROLLED_INPUT_1:%.*]] = VPU.UnrolledType([[UNROLLED_OUTPUT_0]] : !VPU.SparseTensor<data=tensor<1x128x28x28xf16, {order = #NHWC}>
    //CHECK:    [[UNROLLED_WEIGHTS_1:%.*]] = VPU.UnrolledType([[WEIGHTS_1]] : tensor<128x128x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTSTABLE_1:%.*]] = VPU.UnrolledType([[WEIGHTSTABLE_1]] : tensor<128x1x1x4xsi32>) -> !VPU.DistributedTensor<128x1x1x4xsi32


    //CHECK:    [[CONV1:%.*]] = VPU.NCE.Convolution([[UNROLLED_INPUT_1]], [[UNROLLED_WEIGHTS_1]], [[UNROLLED_WEIGHTSTABLE_1]])
    //CHECK:        VPU.UnrolledType([[CONV1]]
    //CHECK-SAME: -> tensor<1x128x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotRemoveOutputSparseForConvSOK
func.func @DoNotRemoveOutputSparseForConvSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x128x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_0 = const.Declare tensor<256x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_2 = const.Declare tensor<128x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 128, 1, 1], strides = [1, 1]} -> !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {order = #NHWC}>>
    %1 = VPU.NCE.Convolution(%0, %cst_2, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [128, 256, 1, 1], strides = [1, 1]} -> tensor<1x128x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x128x28x28xf16, {order = #NHWC}>

    //CHECK:    [[WEIGHTSTABLE_0:%.*]] = const.Declare tensor<256x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_0:%.*]] = const.Declare tensor<256x128x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTSTABLE_1:%.*]] = const.Declare tensor<128x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_1:%.*]] = const.Declare tensor<128x256x1x1xf16, {order = #NHWC}>

    //CHECK:    [[UNROLLED_INPUT_0:%.*]] = VPU.UnrolledType(%arg0 : tensor<1x128x28x28xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTS_0:%.*]] = VPU.UnrolledType([[WEIGHTS_0]] : tensor<256x128x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTSTABLE_0:%.*]] = VPU.UnrolledType([[WEIGHTSTABLE_0]] : tensor<256x1x1x4xsi32>) -> !VPU.DistributedTensor<256x1x1x4xsi32

    //CHECK:        [[CONV0:%.*]] = VPU.NCE.Convolution([[UNROLLED_INPUT_0]], [[UNROLLED_WEIGHTS_0]], [[UNROLLED_WEIGHTSTABLE_0]])
    //CHECK:     [[UNROLLED_OUTPUT_0:%.*]] = VPU.UnrolledType([[CONV0]]
    //CHECK-SAME:   -> !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {order = #NHWC}>>

    //CHECK:    [[UNROLLED_INPUT_1:%.*]] = VPU.UnrolledType([[UNROLLED_OUTPUT_0]] : !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {order = #NHWC}>
    //CHECK:    [[UNROLLED_WEIGHTS_1:%.*]] = VPU.UnrolledType([[WEIGHTS_1]] : tensor<128x256x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    //CHECK:    [[UNROLLED_WEIGHTSTABLE_1:%.*]] = VPU.UnrolledType([[WEIGHTSTABLE_1]] : tensor<128x1x1x4xsi32>) -> !VPU.DistributedTensor<128x1x1x4xsi32

    //CHECK:    [[CONV1:%.*]] = VPU.NCE.Convolution([[UNROLLED_INPUT_1]], [[UNROLLED_WEIGHTS_1]], [[UNROLLED_WEIGHTSTABLE_1]])
    //CHECK:        VPU.UnrolledType([[CONV1]]
    //CHECK-SAME:   -> tensor<1x128x28x28xf16, {order = #NHWC}>
}
