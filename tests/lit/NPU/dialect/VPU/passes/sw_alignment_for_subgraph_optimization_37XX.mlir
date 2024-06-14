//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --wrap-vpu-ops-in-ncecluster-tiling %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNInputAlignForSegmentedInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @MVNInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x32x64x64xf16, {order = #NHWC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true
            } : tensor<1x32x64x64xf16, {order = #NWHC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INPUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WEIGHTS_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WEIGHTS_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x32x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:               -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WEIGHTS_COPY]] as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WT_COPY]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CONV_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_CONV]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[CONV_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[PERMUTE:%.*]] = VPU.PermuteCast([[CONV_OUT_COPY]]) {dst_order = #NWHC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.*]] = VPU.NCE.ClusterTiling ([[PERMUTE]] as %arg1: tensor<1x32x64x64xf16, {order = #NWHC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_COPY_TO_MVN_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_MVN:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_COPY_TO_MVN]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[CLUSTERED_MVN_INNER:%.*]] = VPU.MVN(%arg1) {across_channels = false, eps = 1.000000e-04 : f64, normalize_variance = true} : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK:        }

    //CHECK:        [[MVN_OUT_COPY:%.*]] =  VPU.NCE.ClusterTiling ([[CLUSTERED_MVN]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    //CHECK:            [[MVN_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NWHC}>
    //CHECK:        }

    //CHECK:        return [[MVN_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NWHC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SOKMVNInputAlignForSegmentedInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOKMVNInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x32x64x64xf16, {order = #NHWC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x32x64x64xf16, {order = #NWHC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INPUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WEIGHTS_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WEIGHTS_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x32x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:               -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WEIGHTS_COPY]] as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WT_COPY]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CONV_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_CONV]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[CONV_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[PERMUTE:%.*]] = VPU.PermuteCast([[CONV_OUT_COPY]]) {dst_order = #NWHC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.*]] = VPU.NCE.ClusterTiling ([[PERMUTE]] as %arg1: tensor<1x32x64x64xf16, {order = #NWHC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_COPY_TO_MVN_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_MVN:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_COPY_TO_MVN]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[CLUSTERED_MVN_INNER:%.*]] = VPU.MVN(%arg1) {across_channels = false, eps = 1.000000e-04 : f64, normalize_variance = true} : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK:        }

    //CHECK:        [[MVN_OUT_COPY:%.*]] =  VPU.NCE.ClusterTiling ([[CLUSTERED_MVN]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    //CHECK:            [[MVN_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NWHC}>
    //CHECK:        }

    //CHECK:        return [[MVN_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NWHC}>
}


// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanhInputAlignForSegmentedInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @TanhInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %tanh = VPU.Tanh(%conv) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %tanh : tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INPUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WEIGHTS_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WEIGHTS_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x32x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:               -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WEIGHTS_COPY]] as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WT_COPY]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CONV_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_CONV]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[CONV_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_COPY_TO_TANH:%.*]] = VPU.NCE.ClusterTiling ([[CONV_OUT_COPY]] as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_COPY_TO_TANH_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_TANH:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_COPY_TO_TANH]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[CLUSTERED_TANH_INNER:%.*]] = VPU.Tanh(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[TANH_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_TANH]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[TANH_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        return [[TANH_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SOKTanhInputAlignForSegmentedInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOKTanhInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %tanh = VPU.Tanh(%conv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %tanh : tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INPUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WEIGHTS_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WEIGHTS_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x32x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:               -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WEIGHTS_COPY]] as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WT_COPY]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CONV_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_CONV]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[CONV_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_COPY_TO_TANH:%.*]] = VPU.NCE.ClusterTiling ([[CONV_OUT_COPY]] as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_COPY_TO_TANH_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_TANH:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_COPY_TO_TANH]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[CLUSTERED_TANH_INNER:%.*]] = VPU.Tanh(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[TANH_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_TANH]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[TANH_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        return [[TANH_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNOutputAlignForSegmentedOutput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NWHC}>
func.func @MVNOutputAlignForSegmentedOutput(%input: tensor<1x32x64x64xf16, {order = #NWHC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %act_win = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 1.0E-4 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true}
            : tensor<1x32x64x64xf16, {order = #NWHC}> -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %permute_cast = VPU.PermuteCast(%mvn) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NWHC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %dwconv = VPU.NCE.DepthConvolution(%permute_cast, %weights, %weights_table, %act_win) {
            activation_window_channel_length = 4 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %dwconv : tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK-DAG:    [[ACT_WIN:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    //CHECK:        [[INPUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x32x64x64xf16, {order = #NWHC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[INPUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK:        }

    //CHECK:        [[MVN:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[MVN_INNER:%.*]] = VPU.MVN(%arg1) {across_channels = false, eps = 1.000000e-04 : f64, normalize_variance = true} : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK:        }

    //CHECK:        [[MVN_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[MVN]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    //CHECK:            [[MVN_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NWHC}>
    //CHECK:        }

    //CHECK:        [[PERMUTE:%.*]] = VPU.PermuteCast([[MVN_OUT_COPY]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NWHC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[CONV_IN_COPY:%.*]] = VPU.NCE.ClusterTiling ([[PERMUTE]] as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CONV_IN_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WEIGHTS_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WEIGHTS_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:               -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[ACT_WIN_COPY:%.*]] = VPU.NCE.ClusterTiling ([[ACT_WIN]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[ACT_WIN_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8>
    //CHECK-SAME:               -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[DWCONV:%.*]] = VPU.NCE.ClusterTiling ([[CONV_IN_COPY]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WEIGHTS_COPY]] as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WT_COPY]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>, [[ACT_WIN_COPY]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[DWCONV_INNER:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3, %arg4) {activation_window_channel_length = 4 : i64, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]}
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[DWCONV_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[DWCONV]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[DWCONV_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        return %9 : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DoNotAlignForSOHInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @DoNotAlignForSOHInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x32x64x64xf16, {order = #NHWC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x32x64x64xf16, {order = #NWHC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x32x64x64xf16, {order = #NWHC}>

    // Don't add alignment when the input is SOH
    // since SOH and channel split cannot be compatible
    //CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[INPUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WEIGHTS_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[WEIGHTS_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x32x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[WT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:               -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WEIGHTS_COPY]] as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WT_COPY]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:            [[CLUSTERED_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CONV_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_CONV]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[CONV_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[PERMUTE:%.*]] = VPU.PermuteCast([[CONV_OUT_COPY]]) {dst_order = #NWHC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.*]] = VPU.NCE.ClusterTiling ([[PERMUTE]] as %arg1: tensor<1x32x64x64xf16, {order = #NWHC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[CLUSTERED_COPY_TO_MVN_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_MVN:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_COPY_TO_MVN]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[CLUSTERED_MVN_INNER:%.*]] = VPU.MVN(%arg1) {across_channels = false, eps = 1.000000e-04 : f64, normalize_variance = true} : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK:        }

    //CHECK:        [[MVN_OUT_COPY:%.*]] =  VPU.NCE.ClusterTiling ([[CLUSTERED_MVN]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    //CHECK:            [[MVN_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NWHC}>
    //CHECK:        }

    //CHECK:        return [[MVN_OUT_COPY]] : tensor<1x32x64x64xf16, {order = #NWHC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DoNotAlignForUnalignedChannel
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @DoNotAlignForUnalignedChannel(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x3x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %slice = VPU.Slice %conv [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
    %tanh = VPU.Tanh(%slice) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x3x64x64xf16, {order = #NHWC}>

    return %tanh : tensor<1x3x64x64xf16, {order = #NHWC}>

    // Don't add alignment when the channel needs extra expansion to align
    // to avoid regression caused by redundant alignment
    //CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INPUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WEIGHTS_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x32x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WEIGHTS_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x32x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:               -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WEIGHTS_COPY]] as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WT_COPY]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   ->  !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    //CHECK:            [[CLUSTERED_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CONV_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_CONV]] as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[CONV_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[SLICE:%.*]] = VPU.Slice %4 [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>

    //CHECK:        [[TANH_IN_COPY:%.*]] = VPU.NCE.ClusterTiling ([[SLICE]] as %arg1: tensor<1x3x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x3x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[TANH_IN_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x3x64x64xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[TANH:%.*]] = VPU.NCE.ClusterTiling ([[TANH_IN_COPY]] as %arg1: tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x3x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[TANH_INNER:%.*]] = VPU.Tanh(%arg1) : tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[TANH_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[TANH]] as %arg1: tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x3x64x64xf16, {order = #NHWC}> {
    //CHECK:            [[TANH_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x3x64x64xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        return [[TANH_OUT_COPY]] : tensor<1x3x64x64xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SWOpHasAlignedInputChannelReq
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x384x1x1xf16, {order = #NHWC}>
func.func @SWOpHasAlignedInputChannelReq(%input: tensor<1x384x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x384x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x384x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 384, 1, 1], strides = [1, 1]}
                -> tensor<1x32x1x1xf16, {order = #NHWC}>
    %swish = VPU.Swish(%conv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x1xf16, {order = #NHWC}>

    return %swish : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x384x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x384x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x384x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INPUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x384x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WEIGHTS_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x384x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   !VPU.DistributedTensor<32x384x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WEIGHTS_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x384x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<32x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:               -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as %arg1: tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WEIGHTS_COPY]] as %arg2: tensor<32x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WT_COPY]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   ->  !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 384, 1, 1], strides = [1, 1]}
    //CHECK-SAME:               -> tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CONV_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_CONV]] as %arg1: tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    //CHECK:            [[CONV_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x1x1xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[SWISH_IN_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CONV_OUT_COPY]] as %arg1: tensor<1x32x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[SWISH_IN_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[SWISH:%.*]] = VPU.NCE.ClusterTiling ([[SWISH_IN_COPY]] as %arg1: tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[SWISH_INNER:%.*]] = VPU.Swish(%arg1) : tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[SWISH_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[SWISH]] as %arg1: tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    //CHECK:            [[SWISH_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x1x1xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        return [[SWISH_OUT_COPY]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SWOpHasAlignedOutputChannelReq
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x384x1x1xf16, {order = #NHWC}>
func.func @SWOpHasAlignedOutputChannelReq(%input: tensor<1x384x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %swish = VPU.Swish(%input) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x384x1x1xf16, {order = #NHWC}> -> tensor<1x384x1x1xf16, {order = #NHWC}>
    %weights = const.Declare tensor<32x384x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x384x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%swish, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [32, 384, 1, 1], strides = [1, 1]}
                -> tensor<1x32x1x1xf16, {order = #NHWC}>

    return %conv : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[SWISH_IN_COPY:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x384x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[SWISH_IN_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x384x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[SWISH:%.*]] = VPU.NCE.ClusterTiling ([[SWISH_IN_COPY]] as %arg1: tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[SWISH_INNER:%.*]] = VPU.Swish(%arg1) : tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[SWISH_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[SWISH]] as %arg1: tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x384x1x1xf16, {order = #NHWC}> {
    //CHECK:            [[SWISH_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x384x1x1xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x384x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x384x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[SWISH_OUT_COPY]] as %arg1: tensor<1x384x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x384x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INPUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x384x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WEIGHTS_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x384x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   !VPU.DistributedTensor<32x384x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WEIGHTS_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x384x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:               -> tensor<32x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[WT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[WT_COPY_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:               -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as %arg1: tensor<1x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WEIGHTS_COPY]] as %arg2: tensor<32x384x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[WT_COPY]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   ->  !VPU.DistributedTensor<1x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[CLUSTERED_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 384, 1, 1], strides = [1, 1]}
    //CHECK-SAME:               -> tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        }

    //CHECK:        [[CONV_OUT_COPY:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_CONV]] as %arg1: tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    //CHECK:            [[CONV_OUT_COPY_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:               -> tensor<1x32x1x1xf16, {order = #NHWC}>
    //CHECK:        }

    //CHECK:        return [[CONV_OUT_COPY]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}
