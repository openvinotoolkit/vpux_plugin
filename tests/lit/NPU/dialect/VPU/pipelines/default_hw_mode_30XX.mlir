//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpu %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX30XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @Convolution
module @Convolution {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_0 = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> :
                    tensor<48x3x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x16x62x62xf16>
        %1 = VPU.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x62x62xf16> -> tensor<1x16x62x62xf16, {order = #NHWC}>
        %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {
                     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [48, 16, 3, 3], strides = [1, 1] }
                        -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %3 = VPU.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>

        return %3 : tensor<1x48x60x60xf16>

        // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x16x62x62xf16>
        // CHECK:       [[PERM:%.+]] = VPU.MemPermute([[EXPAND]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x62x62xf16> -> tensor<1x16x62x62xf16, {order = #NHWC}>

        // CHECK:       [[COPY0:%.+]] = VPU.NCE.ClusterTiling ([[PERM]] as {{[^:]+}}: tensor<1x16x62x62xf16, {order = #NHWC}>) ->
        // CHECK-SAME:        !VPU.DistributedTensor<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[COPY1:%.+]] = VPU.NCE.ClusterTiling ([[CST1]] as {{[^:]+}}: tensor<48x16x3x3xf16, {order = #NHWC}>) ->
        // CHECK-SAME:        !VPU.DistributedTensor<48x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[COPY2:%.+]] = VPU.NCE.ClusterTiling ([[CST0]] as {{[^:]+}}: tensor<48x1x1x4xsi32>) ->
        // CHECK-SAME:        !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[CONV:%.+]] = VPU.NCE.ClusterTiling
        // CHECK-SAME:        ([[COPY0]] as {{[^:]+}}: tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:        [[COPY1]] as {{[^:]+}}: tensor<48x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:        [[COPY2]] as {{[^:]+}}: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>) ->
        // CHECK-SAME:        !VPU.DistributedTensor<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
        // CHECK-NEXT:           VPU.NCE.Convolution

        // CHECK:       [[COPY3:%.+]] = VPU.NCE.ClusterTiling ([[CONV]] as {{[^:]+}}: tensor<1x48x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>) ->
        // CHECK-SAME:        tensor<1x48x60x60xf16, {order = #NHWC}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[OUT:%.+]] = VPU.MemPermute([[COPY3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        // CHECK:       return [[OUT:%.+]] : tensor<1x48x60x60xf16>
    }
}

// -----

// CHECK-LABEL: @SoftMax
module @SoftMax {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
      DataInfo "softmax" : tensor<1x1000xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
    func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
        %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        %1 = VPU.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
        %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        return %2 : tensor<1x1000xf16>

        // CHECK:               [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG0]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[SOFTMAX:%.+]] = VPU.SoftMax([[RESHAPE]]) {axisInd = 3 : i64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[OUT:%.+]] = VPU.AffineReshape([[SOFTMAX]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        // CHECK:               return [[OUT]] : tensor<1x1000xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x1x1x4xsi32> =  dense<1> : tensor<48x1x1x4xsi32>
        %cst_0 = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x16x62x62xf16>
        %1 = VPU.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x62x62xf16> -> tensor<1x16x62x62xf16, {order = #NHWC}>
        %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [48, 16, 3, 3], strides = [1, 1]} -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %3 = VPU.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %3 : tensor<1x48x60x60xf16>

        // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x16x62x62xf16>
        // CHECK:       [[PERM:%.+]] = VPU.MemPermute([[EXPAND]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x62x62xf16> -> tensor<1x16x62x62xf16, {order = #NHWC}>

        // CHECK:       [[COPY0:%.+]] = VPU.NCE.ClusterTiling ([[PERM]] as {{[^:]+}}: tensor<1x16x62x62xf16, {order = #NHWC}>) ->
        // CHECK-SAME:        !VPU.DistributedTensor<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[COPY1:%.+]] = VPU.NCE.ClusterTiling ([[CST1]] as {{[^:]+}}: tensor<48x16x3x3xf16, {order = #NHWC}>) ->
        // CHECK-SAME:        !VPU.DistributedTensor<48x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[COPY2:%.+]] = VPU.NCE.ClusterTiling ([[CST0]] as {{[^:]+}}: tensor<48x1x1x4xsi32>) ->
        // CHECK-SAME:        !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[CONV:%.+]] = VPU.NCE.ClusterTiling
        // CHECK-SAME:        ([[COPY0]] as {{[^:]+}}: tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:        [[COPY1]] as {{[^:]+}}: tensor<48x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:        [[COPY2]] as {{[^:]+}}: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>) ->
        // CHECK-SAME:        !VPU.DistributedTensor<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
        // CHECK-NEXT:           VPU.NCE.Convolution

        // CHECK:       [[COPY3:%.+]] = VPU.NCE.ClusterTiling ([[CONV]] as {{[^:]+}}: tensor<1x48x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>) ->
        // CHECK-SAME:        tensor<1x48x60x60xf16, {order = #NHWC}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[OUT:%.+]] = VPU.MemPermute([[COPY3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        // CHECK:       return [[OUT:%.+]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        return %0 : tensor<1x48x60x60xf16>

        // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        // CHECK: return [[SOFTMAX]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16> {
        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        %1 = call @foo1(%0) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        %2 = call @foo2(%1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %2 : tensor<1x48x60x60xf16>

        // CHECK: [[CONVERT:%.+]] = VPU.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        // CHECK: [[FOO1_RES:%.+]] = call @foo1([[CONVERT]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: return [[FOO2_RES]] : tensor<1x48x60x60xf16>
    }
}
