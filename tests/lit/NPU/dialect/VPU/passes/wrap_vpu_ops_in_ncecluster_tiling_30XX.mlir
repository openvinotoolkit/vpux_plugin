//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --wrap-vpu-ops-in-ncecluster-tiling %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOH
func.func @ConvToNCEClusterTilingSOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<80x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<80x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32> -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [80, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOKUse3Clusters
func.func @ConvToNCEClusterTilingSOKUse3Clusters(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [80, 128, 1, 1], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}> -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<80x128x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x128x1x1xf16, {order = #NHWC}> -> tensor<80x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<80x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32> -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<80x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [80, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOK
func.func @ConvToNCEClusterTilingSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}> -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<64x128x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x128x1x1xf16, {order = #NHWC}> -> tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<64x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingClustering
func.func @ConvToNCEClusterTilingClustering(%arg0: tensor<1x64x14x14xf16, {order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} -> tensor<1x48x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x48x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x14x14xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<48x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<48x64x3x3xf16, {order = #NHWC}> -> tensor<48x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<48x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<48x1x1x4xsi32> -> tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<48x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x48x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [48, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) : tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x48x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @CMConvToNCEClusterTilingSOHOverlapped
func.func @CMConvToNCEClusterTilingSOHOverlapped(%arg0: tensor<1x3x224x224xf16, {order = #NCHW}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x32xui8> = dense<10> : tensor<1x1x1x32xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x1x1x32xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x32xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_1, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, activation_window_channel_length = 81 : i64, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, rawFilterShape = [32, 3, 3, 3], strides = [2, 2]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x32xui8> = dense<10> : tensor<1x1x1x32xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x1x1x32xf16, {order = #NHWC}>
    //CHECK-SAME:   dense<1.000000e+00> : tensor<32x1x1x32xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x224x224xf16, {order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x3x224x224xf16, {order = #NCHW}> -> tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x1x1x32xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:           [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x32xf16, {order = #NHWC}> -> tensor<32x1x1x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:           VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x32xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x32xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x32xui8> -> tensor<1x1x1x32xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x1x1x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<1x1x1x32xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3, %arg4 ) {
    //CHECK-SAME:                 activation_window_channel_length = 81 : i64, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [32, 3, 3, 3], strides = [2, 2]
    //CHECK-SAME:             } -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @CMConvToNCEClusterTilingClustering
func.func @CMConvToNCEClusterTilingClustering(%arg0: tensor<1x3x16x16xf16, {order = #NCHW}>) -> tensor<1x48x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_1 = const.Declare tensor<48x1x1x32xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x1x1x32xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_1, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, activation_window_channel_length = 54 : i64, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [48, 3, 3, 3], strides = [1, 1]} -> tensor<1x48x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x48x16x16xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x1x1x32xf16, {order = #NHWC}>
    //CHECK-SAME:   dense<1.000000e+00> : tensor<48x1x1x32xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x16x16xf16, {order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x3x16x16xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x3x16x16xf16, {order = #NCHW}> -> tensor<1x3x16x16xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<48x1x1x32xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x1x1x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:           [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<48x1x1x32xf16, {order = #NHWC}> -> tensor<48x1x1x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:           VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<48x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<48x1x1x4xsi32> -> tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x3x16x16xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<48x1x1x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x48x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3, %arg4 ) {
    //CHECK-SAME:                 activation_window_channel_length = 54 : i64, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [48, 3, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x16x16xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x16x16xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x48x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOH
func.func @DepthConvToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, activation_window_channel_length = 18 : i64, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>) -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>) -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3, %arg4 ) {
    //CHECK-SAME:               activation_window_channel_length = 18 : i64, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling (%4 as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5:%.*]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOK
func.func @DepthConvToNCEClusterTilingSOK(%arg0: tensor<1x128x56x56xf16, {order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, activation_window_channel_length = 18 : i64, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x56x56xf16, {order = #NHWC}>
    return %0 : tensor<1x128x56x56xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x56x56xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x56x56xf16, {order = #NHWC}> -> tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<128x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<128x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<128x16x1x1xf16, {order = #NHWC}> -> tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<128x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<128x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<128x1x1x4xsi32> -> tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3, %arg4 ) {
    //CHECK-SAME:                 activation_window_channel_length = 18 : i64, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [128, 1, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x128x56x56xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x128x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingClustering
func.func @DepthConvToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, activation_window_channel_length = 18 : i64, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3, %arg4 ) {
    //CHECK-SAME:                 activation_window_channel_length = 18 : i64, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [32, 1, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCEClusterTilingSOH
func.func @MaxPoolToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            activation_window_channel_length = 4 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg2: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1, %arg2 , %arg3 ) {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCEClusterTilingClustering
func.func @MaxPoolToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            activation_window_channel_length = 4 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg2: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1, %arg2 , %arg3 ) {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingSOH
func.func @EltwiseAddToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX]] as %arg2: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX]] as %arg3: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = #VPU.eltwise_type<ADD>} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg2: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }
    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingClustering
func.func @EltwiseAddToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>, %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x14x14xf16, {order = #NHWC}>, tensor<1x32x14x14xf16, {order = #NHWC}>
         -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0: tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX]] as %arg2: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX]] as %arg3: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = #VPU.eltwise_type<ADD>} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg2: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeMultiClusteringSubgraphAlignment
func.func @OptimizeMultiClusteringSubgraphAlignment(%arg0: tensor<1x80x22x22xf16, {order = #NHWC}>) -> tensor<1x64x22x22xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_1 = const.Declare tensor<80x80x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x80x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 80, 3, 3], strides = [1, 1]} -> tensor<1x80x22x22xf16, {order = #NHWC}>
    %cst_2 = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_3 = const.Declare tensor<64x80x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x80x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%0, %cst_3, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [64, 80, 3, 3], strides = [1, 1]} -> tensor<1x64x22x22xf16, {order = #NHWC}>
    return %1 : tensor<1x64x22x22xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_0:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS_0:%.*]] = const.Declare tensor<80x80x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x80x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x80x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x80x22x22xf16, {order = #NHWC}> -> tensor<1x80x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX_0:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<80x80x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x80x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x80x3x3xf16, {order = #NHWC}> -> tensor<80x80x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX_0:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE_0]] as %arg1: tensor<80x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32> -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_0]] as %arg1: tensor<1x80x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX_0]] as %arg2: tensor<80x80x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX_0]] as %arg3: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [80, 80, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x80x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_0]] as %arg1: tensor<1x80x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x80x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS_1:%.*]] = const.Declare tensor<64x80x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x80x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x80x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x80x22x22xf16, {order = #NHWC}> -> tensor<1x80x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<64x80x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x80x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES6:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x80x3x3xf16, {order = #NHWC}> -> tensor<64x80x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES6]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE_1]] as %arg1: tensor<64x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES7:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES7]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_1]] as %arg1: tensor<1x80x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX_1]] as %arg2: tensor<64x80x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX_1]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES8:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [64, 80, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES8]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_1]] as %arg1: tensor<1x64x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES9:%.*]] = VPU.Copy(%arg1) : tensor<1x64x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES9]]
    //CHECK:        }

    //CHECK:        return [[OUT_1]] : tensor<1x64x22x22xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvToNCEClusterTilingSOH
func.func @SparseConvToNCEClusterTilingSOH(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    %weights_table = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32>

    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[INPUT_SPARSE]] as [[INPUT_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                       sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 1, 2, 1]}>> {
    // CHECK:         [[VAR0:%.+]] = VPU.Copy([[INPUT_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR0]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]] as [[WEIGHTS_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                           sparsity_map=tensor<80x1x1x640xi1>,
    // CHECK-SAME:                                                                           is_weights>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                           is_weights> {
    // CHECK:         [[VAR1:%.+]] = VPU.Copy([[WEIGHTS_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR1]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[CST_WEIGHTS_TABLE]] as [[WEIGHTS_TABLE_ARG:%.+]]: tensor<80x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    // CHECK:         [[VAR2:%.+]] = VPU.Copy([[WEIGHTS_TABLE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR2]]
    // CHECK:       }

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[INPUT_SPARSE_CMX]] as [[INPUT_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:      [[WEIGHTS_SPARSE_CMX]] as [[WEIGHTS_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<80x1x1x640xi1, {mem_space = @CMX_NN, order = #NCHW}>, is_weights>,
    // CHECK-SAME:      [[WEIGHTS_TABLE_CMX]] as [[WEIGHTS_TABLE_CMX_ARG:[^:]+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>> {
    // CHECK:         [[VAR3:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_CMX_ARG]], [[WEIGHTS_SPARSE_CMX_ARG]], [[WEIGHTS_TABLE_CMX_ARG]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>
    // CHECK:         VPU.Yield [[VAR3]]
    // CHECK:       }

    // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:[^:]+]]:
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                        sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
    // CHECK:         [[VAR4:%.+]] = VPU.Copy([[OUT_CMX_ARG]])
    // CHECK:         VPU.Yield [[VAR4]]
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEInterpolateToNCEClusterTilingClustering
func.func @NCEInterpolateToNCEClusterTilingClustering(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x2x2xi1> = dense<1> : tensor<1x16x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>
    } -> tensor<1x1x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x2x2xi1>,
                           storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        scales_attr = [2, 2],
        ppe = #VPU.PPETask<clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = <NOOP>>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 16, 1, 1],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:       -> tensor<1x1x2x2xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x16x2x2xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                                sparsity_map=!VPU.DistributedTensor<1x16x2x2xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                                storage_element_table=!VPU.DistributedTensor<1x1x2x2xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 1, 1, 1]}>,
    // CHECK-SAME:                                #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<16x1x1x4xsi32>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x16x2x2xi1, {mem_space = @CMX_NN, order = #NCHW}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x2x2xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                                                                         nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x16x2x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:           ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:           scales_attr = [2, 2]
    // CHECK-SAME:           strides = [1, 1]
    // CHECK-SAME:           } -> tensor<1x16x2x2xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x16x2x2xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEInterpolateToNCEClusterTilingSOH
func.func @NCEInterpolateToNCEClusterTilingSOH(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x10x20xi1>,
                           storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        scales_attr = [2, 2],
        ppe = #VPU.PPETask<clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = <NOOP>>
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 64, 5, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 64 : i64}
    // CHECK-SAME:       -> tensor<1x1x10x20xi32, {order = #NHWC}>

    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x5x10xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:                                sparsity_map=!VPU.DistributedTensor<1x64x10x20xi1, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                                storage_element_table=!VPU.DistributedTensor<1x1x10x20xi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 1, 2, 1]}>,
    // CHECK-SAME:                                #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<64x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<64x1x1x4xsi32>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN, order = #NCHW}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECKSAME:                                                                                          nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x10x20xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:           ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:           scales_attr = [2, 2]
    // CHECK-SAME:           strides = [1, 1]
    // CHECK-SAME:           } -> tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEInterpolateToNCEClusterTilingSOK
func.func @NCEInterpolateToNCEClusterTilingSOK(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x10x20xi1>,
                           storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        scales_attr = [2, 2],
        ppe = #VPU.PPETask<clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = <NOOP>>
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 64, 5, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 64 : i64}
    // CHECK-SAME:       -> tensor<1x1x10x20xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x5x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                                sparsity_map=!VPU.DistributedTensor<1x64x10x20xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                                storage_element_table=!VPU.DistributedTensor<1x1x10x20xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 1, 1, 1]}>,
    // CHECK-SAME:                                #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<64x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<64x1x1x4xsi32>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN, order = #NCHW}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                                                                         nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:   -> !VPU.DistributedTensor<1x64x10x20xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:           ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:           scales_attr = [2, 2]
    // CHECK-SAME:           strides = [1, 1]
    // CHECK-SAME:           } -> tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}
