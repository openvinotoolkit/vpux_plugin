//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitDepthConvWithBigC
func.func @SplitDepthConvWithBigC(%arg0: tensor<1x5120x64x4xf16, {order = #NHWC}>) -> tensor<1x5120x64x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<5120x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<5120x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<5120x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<5120x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %wt) {
            opaque_ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [5120, 1, 1, 1], strides = [1, 1]
        } -> tensor<1x5120x64x4xf16, {order = #NHWC}>

    return %0 : tensor<1x5120x64x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<5120x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<5120x1x1x4xsi32, {order = #NHWC}>
    // CHECK: [[DWConv:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[CST]], [[CST0]])
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:               rawFilterShape = [5120, 1, 1, 1], strides = [1, 1],
    // CHECK-SAME:               tilingStrategy = [1, 4, 1, 1]} -> tensor<1x5120x64x4xf16, {order = #NHWC}>
    // CHECK:  return [[DWConv]] : tensor<1x5120x64x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEMaxPoolWithBigC
func.func @SplitNCEMaxPoolWithBigC(%arg0: tensor<1x5120x32x4xf16, {order = #NHWC}>) -> tensor<1x5120x32x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        opaque_ppe = #VPU.PPEStub<>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x5120x32x4xf16, {order = #NHWC}>

    return %0 : tensor<1x5120x32x4xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      } -> tensor<1x5120x32x4xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<1x5120x32x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolWithBigC
func.func @SplitNCEAveragePoolWithBigC(%arg0: tensor<1x5120x32x4xf16, {order = #NHWC}>) -> tensor<1x5120x32x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        opaque_ppe = #VPU.PPEStub<>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x5120x32x4xf16, {order = #NHWC}>
    return %0 : tensor<1x5120x32x4xf16, {order = #NHWC}>

    // CHECK:  [[AVGPOOL:%.+]] = VPU.NCE.AveragePool(%arg0) {
    // CHECK-SAME:   kernel_size = [1, 1],
    // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:   strides = [1, 1],
    // CHECK-SAME:   tilingStrategy = [1, 2, 1, 1]} -> tensor<1x5120x32x4xf16, {order = #NHWC}>
    // CHECK:  return [[AVGPOOL]] : tensor<1x5120x32x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitSparseDepthConvWithBigC
func.func @SplitSparseDepthConvWithBigC(%arg0: tensor<1x4080x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>> {
    %cst0 = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst0, %wt) {
            opaque_ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [4080, 1, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    return %0 : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG: [[INPUT:%.+]] = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG: [[WT:%.*]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    // CHECK: [[DWConv:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[INPUT]], [[WT]]) {
    // CHECK:            tilingStrategy = [1, 19, 1, 1]
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
    // CHECK: return [[DWConv]] : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitSparseNCEMaxPoolWithBigC
func.func @SplitSparseNCEMaxPoolWithBigC(%arg0: tensor<1x4080x16x16xf16, {order = #NHWC}>) -> tensor<1x4080x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %wt) {
        opaque_ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>> -> tensor<1x4080x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x4080x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    // CHECK-DAG: [[WT:%.+]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[WT]] )
    // CHECK:              tilingStrategy = [1, 5, 1, 1]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitSparseDepthConvWithBigCWithSOK
func.func @SplitSparseDepthConvWithBigCWithSOK(%arg0: tensor<1x4080x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>> {
    %cst0 = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst0, %wt) {
            opaque_ppe = #VPU.PPEStub<>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [4080, 1, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    return %0 : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG: [[INPUT:%.+]] = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG: [[WT:%.*]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    // CHECK: [[DWConv:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[INPUT]], [[WT]]) {
    // CHECK:            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK:            tilingStrategy = [1, 11, 1, 1]
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
    // CHECK: return [[DWConv]] : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
}

// -----

// CHECK-LABEL: @TileGatherDMA
func.func @TileGatherDMA(%arg0: tensor<880x960xf16>, %arg1: tensor<1x880xsi32>) -> tensor<1x880x960xf16> {
    %0 = VPU.Reshape(%arg1) {shape_value = [880, 1]} : tensor<1x880xsi32> -> tensor<880x1xsi32>
    %1 = VPU.Convert(%0) {dstElemType = i64} : tensor<880x1xsi32> -> tensor<880x1xi64>
    %2 = VPU.Copy(%1) {out_mem_space = [@CMX_NN, 0]} : tensor<880x1xi64> -> tensor<880x1xi64, {mem_space = [@CMX_NN, 0], order = affine_map<(d0, d1) -> (d0, d1)>}>
    %3 = VPU.GatherDMA(%arg0, %2) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<880x960xf16>, tensor<880x1xi64, {mem_space = [@CMX_NN, 0], order = affine_map<(d0, d1) -> (d0, d1)>}> -> tensor<880x960xf16, {mem_space = [@CMX_NN, 0], order = affine_map<(d0, d1) -> (d0, d1)>}>
    %4 = VPU.Copy(%3) {out_mem_space = @DDR} : tensor<880x960xf16, {mem_space = [@CMX_NN, 0], order = affine_map<(d0, d1) -> (d0, d1)>}> -> tensor<880x960xf16>
    %5 = VPU.Reshape(%4) {shape_value = [1, 880, 960]} : tensor<880x960xf16> -> tensor<1x880x960xf16>
    return %5 : tensor<1x880x960xf16>

    // CHECK:       [[RESHAPE0:%.+]] = VPU.Reshape({{[^:]+}}) {shape_value = [880, 1]} : tensor<1x880xsi32> -> tensor<880x1xsi32>
    // CHECK:       [[CONVERT:%.+]]  = VPU.Convert([[RESHAPE0]]) {dstElemType = i64} : tensor<880x1xsi32> -> tensor<880x1xi64>
    // CHECK:       [[COPY0:%.+]] = VPU.Copy([[CONVERT]]) {out_mem_space = [@CMX_NN, 0]} : tensor<880x1xi64> -> tensor<880x1xi64, {mem_space = [@CMX_NN, 0], order = #NC}>
    // CHECK:       [[GATHERDMA:%.+]] = VPU.GatherDMA({{[^:]+}}, [[COPY0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 2]} : tensor<880x960xf16>, tensor<880x1xi64, {mem_space = [@CMX_NN, 0], order = #NC}> -> tensor<880x960xf16, {mem_space = [@CMX_NN, 0], order = #NC}>
    // CHECK:       [[COPY1:%.+]] = VPU.Copy([[GATHERDMA]]) {out_mem_space = @DDR} : tensor<880x960xf16, {mem_space = [@CMX_NN, 0], order = #NC}> -> tensor<880x960xf16>
    // CHECK:       [[RESHAPE1:%.+]] = VPU.Reshape([[COPY1]]) {shape_value = [1, 880, 960]} : tensor<880x960xf16> -> tensor<1x880x960xf16>

    // CHECK:       return [[RESHAPE1]] : tensor<1x880x960xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TileGatherDMA4D
func.func @TileGatherDMA4D(%arg0: tensor<1x30522x2100x1xf16>, %arg1: tensor<512xsi32>) -> tensor<1x512x2100x1xf16> {
    %0 = VPU.Reshape(%arg1) {shape_value = [1, 512, 1, 1]} : tensor<512xsi32> -> tensor<1x512x1x1xsi32>
    %1 = VPU.Convert(%0) {dstElemType = i64} : tensor<1x512x1x1xsi32> -> tensor<1x512x1x1xi64>
    %2 = VPU.Copy(%1) {out_mem_space = [@CMX_NN, 0]} : tensor<1x512x1x1xi64> -> tensor<1x512x1x1xi64, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    %3 = VPU.GatherDMA(%arg0, %2) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x30522x2100x1xf16>, tensor<1x512x1x1xi64, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x512x2100x1xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    %4 = VPU.Copy(%3) {out_mem_space = @DDR} : tensor<1x512x2100x1xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x512x2100x1xf16>
    %5 = VPU.Reshape(%4) {shape_value = [1, 512, 2100, 1]} : tensor<1x512x2100x1xf16> -> tensor<1x512x2100x1xf16>
    return %5 : tensor<1x512x2100x1xf16>

    // CHECK:       [[RESHAPE0:%.+]] = VPU.Reshape({{[^:]+}}) {shape_value = [1, 512, 1, 1]} : tensor<512xsi32> -> tensor<1x512x1x1xsi32>
    // CHECK:       [[CONVERT:%.+]]  = VPU.Convert([[RESHAPE0]]) {dstElemType = i64} : tensor<1x512x1x1xsi32> -> tensor<1x512x1x1xi64>
    // CHECK:       [[COPY0:%.+]] = VPU.Copy([[CONVERT]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x512x1x1xi64> -> tensor<1x512x1x1xi64, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[GATHERDMA:%.+]] = VPU.GatherDMA({{[^:]+}}, [[COPY0]]) {axis_value = 1 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x30522x2100x1xf16>, tensor<1x512x1x1xi64, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x512x2100x1xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:       [[COPY1:%.+]] = VPU.Copy([[GATHERDMA]]) {out_mem_space = @DDR} : tensor<1x512x2100x1xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x512x2100x1xf16>
    // CHECK:       [[RESHAPE1:%.+]] = VPU.Reshape([[COPY1]]) {shape_value = [1, 512, 2100, 1]} : tensor<1x512x2100x1xf16> -> tensor<1x512x2100x1xf16>

    // CHECK:       return [[RESHAPE1]] : tensor<1x512x2100x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitSparseNCEMaxPoolWithBigCWithSOK
func.func @SplitSparseNCEMaxPoolWithBigCWithSOK(%arg0: tensor<1x4080x16x16xf16, {order = #NHWC}>) -> tensor<1x4080x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %wt) {
        opaque_ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>> -> tensor<1x4080x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x4080x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT:%.+]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[WT]] )
    // CHECK:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK:              tilingStrategy = [1, 5, 1, 1]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseType = !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
!SparseType1 = !VPU.SparseTensor<data=tensor<1x4064x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4064x16x16xi1, {order = #NHWC}>>


// CHECK-LABEL: @SplitOutputSparseForConvSOKFollowedByConcat
func.func @SplitOutputSparseForConvSOKFollowedByConcat(%arg0: tensor<1x2032x16x16xf16, {order = #NHWC}>) -> tensor<1x4064x16x16xf16, {order = #NHWC}> {
    %s0 = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}> -> !SparseType
    %wt0 = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    %maxpool0 = VPU.NCE.MaxPool(%s0, %wt0) {
        opaque_ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !SparseType

    %s1 = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}> -> !SparseType
    %wt1 = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    %maxpool1 = VPU.NCE.MaxPool(%s1, %wt1) {
        opaque_ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !SparseType


    %concat = VPU.Concat(%maxpool0, %maxpool1) {static_offsets = [[0, 0, 0, 0], [0, 2032, 0, 0]]} : !SparseType, !SparseType -> !SparseType1

    %wt2 = const.Declare tensor<4064x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4064x1x1x4xsi32>
    %maxpool2 = VPU.NCE.MaxPool(%concat, %wt2) {
        opaque_ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !SparseType1

    %result = VPU.Desparsify(%maxpool2) : !SparseType1 -> tensor<1x4064x16x16xf16, {order = #NHWC}>
    return %result : tensor<1x4064x16x16xf16, {order = #NHWC}>

    // CHECK: [[ToSparsity_0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}>
    // CHECK:        -> !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT_0:%.+]] = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    // CHECK: [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[ToSparsity_0]], [[WT_0]] )
    // CHECK:              tilingStrategy = [1, 3, 1, 1]

    // CHECK: [[ToSparsity_1:%.+]] = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}>
    // CHECK:        -> !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT_1:%.+]] = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    // CHECK: [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[ToSparsity_1]], [[WT_1]] )
    // CHECK-SAME:              tilingStrategy = [1, 3, 1, 1]

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL_0]], [[MAXPOOL_1]])
    // CHECK-DAG: [[WT_2:%.+]] = const.Declare tensor<4064x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4064x1x1x4xsi32>
    // CHECK: [[MAXPOOL_2:%.+]] = VPU.NCE.MaxPool([[CONCAT]], [[WT_2]] )
    // CHECK-SAME:              tilingStrategy = [1, 5, 1, 1]
    // CHECK: [[RESULT:%.+]] = VPU.Desparsify([[MAXPOOL_2]])

    // CHECK: return [[RESULT]]
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0043085547638874429:24>

// CHECK-LABEL: @DontTileD2SDMA
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x64x128x128x!qElemType, {order = #NHWC}>
func.func @DontTileD2SDMA(%arg0: tensor<1x64x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        opaque_ppe = #VPU.PPEStub<>, strides = [1, 1]}
            -> tensor<1x64x128x128x!qElemType, {order = #NHWC}>
    %d2s = VPU.DepthToSpace(%avgpool) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x64x128x128x!qElemType, {order = #NHWC}>
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    %eltwise = VPU.NCE.Eltwise(%d2s, %d2s) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, opaque_ppe = #VPU.PPEStub<>}
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    return %eltwise : tensor<1x16x256x256x!qElemType, {order = #NHWC}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[AVGPOOL]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 1]
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[D2S]], [[D2S]])
    // CHECK:       return [[ELTWISE]]
}

// -----

// CHECK-LABEL: @MVNSOKAndKTile
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x32x61440x1xf16>
func.func @MVNSOKAndKTile(%arg0: tensor<1x32x61440x1xf16>) -> tensor<1x32x61440x1xf16> {
    %mvn = VPU.MVN(%arg0) {
        across_channels = false, eps = 9.9999997473787516E-6 : f64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        normalize_variance = true
    } : tensor<1x32x61440x1xf16> -> tensor<1x32x61440x1xf16>

    return %mvn : tensor<1x32x61440x1xf16>

    // CHECK:       VPU.MVN([[INPUT]])
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
}
