//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --remove-output-sparse-to-avoid-suboptimal-dpu-workloads %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseType = !VPU.SparseTensor<data=tensor<1x128x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x28x28xi1, {order = #NHWC}>>

// CHECK-LABEL: @RemoveOutputSparseForConvSOKFollowedByConcat
func.func @RemoveOutputSparseForConvSOKFollowedByConcat(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x128x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_2 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [128, 128, 1, 1], strides = [1, 1]} -> !SparseType
    %1 = VPU.NCE.Convolution(%arg0, %cst_2, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [128, 128, 1, 1], strides = [1, 1]} -> !SparseType

    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : !SparseType, !SparseType -> !VPU.SparseTensor<data=tensor<1x256x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x28x28xi1, {order = #NHWC}>>

    %cst_3 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_4 = const.Declare tensor<128x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %3 = VPU.NCE.Convolution(%2, %cst_4, %cst_3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [128, 256, 1, 1], strides = [1, 1]} -> tensor<1x128x28x28xf16, {order = #NHWC}>

    return %3 : tensor<1x128x28x28xf16, {order = #NHWC}>

    //CHECK:    [[WEIGHTSTABLE_0:%.*]] = const.Declare tensor<128x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_0:%.*]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTSTABLE_1:%.*]] = const.Declare tensor<128x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_1:%.*]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>

    //CHECK:        [[CONV0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS_0]], [[WEIGHTSTABLE_0]])
    //CHECK-SAME:   -> tensor<1x128x28x28xf16, {order = #NHWC}>

    //CHECK:        [[CONV1:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS_1]], [[WEIGHTSTABLE_1]])
    //CHECK-SAME:   -> tensor<1x128x28x28xf16, {order = #NHWC}>

    //CHECK:        VPU.Concat([[CONV0]], [[CONV1]])
    //CHECK-SAME:   -> tensor<1x256x28x28xf16, {order = #NHWC}>
}
