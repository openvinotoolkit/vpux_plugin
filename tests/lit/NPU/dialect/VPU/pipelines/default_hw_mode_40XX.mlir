//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpu %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_0 = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}>
                      = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [48, 16, 3, 3], strides = [1, 1]}
                  -> tensor<1x48x60x60xf16>
        return %3 : tensor<1x48x60x60xf16>

        // CHECK:       [[CST0:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense_resource<__elided__> : tensor<48x1x1x4xsi32>
        // CHECK:       [[CST1:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
        // CHECK:       [[CST2:%.+]] = const.Declare tensor<48x1x1x256xi1> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
        // CHECK:       [[SPARSE:%.+]] = VPU.GroupSparseTensor([[CST1]], [[CST2]])
        // CHECK-SAME:        {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>} ->
        // CHECK-SAME:        !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<48x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[COPY0:%.+]] = VPU.NCE.ClusterTiling ([[EXPAND]] as {{[^:]+}}: tensor<1x3x62x64xf16>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x3x62x64xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 10, 64], [1, 3, 10, 64], [1, 3, 10, 64], [1, 3, 10, 64]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 10, 64], [1, 3, 10, 64], [1, 3, 10, 64], [1, 3, 10, 64]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[PERM:%.+]] = VPU.NCE.ClusterTiling ([[COPY0]] as {{[^:]+}}: tensor<1x3x62x64xf16, {mem_space = @CMX_NN, order = #NCHW}>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x16x62x64xf16, #NHWC, @CMX_NN,
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 11, 64], [1, 16, 11, 64], [1, 16, 10, 64], [1, 16, 10, 64], [1, 16, 10, 64], [1, 16, 10, 64]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 11, 64], [1, 16, 11, 64], [1, 16, 10, 64], [1, 16, 10, 64], [1, 16, 10, 64], [1, 16, 10, 64]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]}> {
        // CHECK-NEXT:         VPU.NCE.Permute

        // CHECK:       [[COPY1:%.+]] = VPU.NCE.ClusterTiling ([[PERM]] as {{[^:]+}}: tensor<1x16x62x64xf16, {mem_space = @CMX_NN, order = #NHWC}>) ->
        // CHECK-SAME:       tensor<1x16x62x64xf16, {order = #NHWC}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[COPY1]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[IN:%.+]] = VPU.NCE.ClusterTiling ([[SLICE]] as {{[^:]+}}: tensor<1x16x62x62xf16, {order = #NHWC}>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x16x62x62xf16, #NHWC, @CMX_NN,
        // CHECK-SAME:        {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 11, 62], [1, 16, 11, 62], [1, 16, 10, 62], [1, 16, 10, 62], [1, 16, 10, 62], [1, 16, 10, 62]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 12, 62], [1, 16, 12, 62], [1, 16, 12, 62], [1, 16, 12, 62], [1, 16, 12, 62], [1, 16, 12, 62]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[COPY2:%.+]] = VPU.NCE.ClusterTiling ([[SPARSE]] as {{[^:]+}}: !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<48x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>) ->
        // CHECK-SAME:       !VPU.SparseTensor<data=!VPU.DistributedTensor<48x16x3x3xf16, #NHWC, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
        // CHECK-SAME:                         sparsity_map=!VPU.DistributedTensor<48x1x1x256xi1, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
        // CHECK-SAME:                         is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[COPY3:%.+]] = VPU.NCE.ClusterTiling ([[CST0]] as {{[^:]+}}: tensor<48x1x1x4xsi32>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[CONV:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:       [[COPY2]] as {{[^:]+}}: !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:                                                 sparsity_map=tensor<48x1x1x256xi1, {mem_space = @CMX_NN, order = #NCHW}>,
        // CHECK-SAME:                                                 is_weights,
        // CHECK-SAME:                                                 #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>,
        // CHECK-SAME:       [[COPY3]] as {{[^:]+}}: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
        // CHECK-SAME:  -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]}> {
        // CHECK-NEXT:         VPU.NCE.Convolution

        // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[CONV]] as {{[^:]+}}: tensor<1x48x60x60xf16, {mem_space = @CMX_NN, order = #NCHW}>)
        // CHECK-SAME:       -> tensor<1x48x60x60xf16> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
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
        // CHECK:               [[COPY0:%.+]] = VPU.NCE.ClusterTiling ([[RESHAPE]] as {{[^:]+}}: tensor<1x1x1x1000xf16>) ->
        // CHECK-SAME:              !VPU.DistributedTensor<1x1x1x1000xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
        // CHECK-NEXT:                   VPU.Copy

        // CHECK:               [[SOFTMAX:%.+]] = VPU.NCE.ClusterTiling ([[COPY0]] as {{[^:]+}}: tensor<1x1x1x1000xf16, {mem_space = @CMX_NN, order = #NCHW}>) ->
        // CHECK-SAME:              !VPU.DistributedTensor<1x1x1x1000xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
        // CHECK-NEXT:                  VPU.SoftMax

        // CHECK:               [[COPY1:%.+]] = VPU.NCE.ClusterTiling ([[SOFTMAX]] as {{[^:]+}}: tensor<1x1x1x1000xf16, {mem_space = @CMX_NN, order = #NCHW}>) ->
        // CHECK-SAME:              tensor<1x1x1x1000xf16> {
        // CHECK-NEXT:                  VPU.Copy

        // CHECK:               [[OUT:%.+]] = VPU.AffineReshape([[COPY1]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        // CHECK:               return [[OUT]] : tensor<1x1000xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_0 = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [48, 16, 3, 3], strides = [1, 1]}
                    -> tensor<1x48x60x60xf16>
        return %3 : tensor<1x48x60x60xf16>

        // CHECK:       [[CST0:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense_resource<__elided__> : tensor<48x1x1x4xsi32>
        // CHECK:       [[CST1:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
        // CHECK:       [[CST2:%.+]] = const.Declare tensor<48x1x1x256xi1> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
        // CHECK:       [[SPARSE:%.+]] = VPU.GroupSparseTensor([[CST1]], [[CST2]])
        // CHECK-SAME:        {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>} ->
        // CHECK-SAME:        !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<48x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[COPY0:%.+]] = VPU.NCE.ClusterTiling ([[EXPAND]] as {{[^:]+}}: tensor<1x3x62x64xf16>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x3x62x64xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 10, 64], [1, 3, 10, 64], [1, 3, 10, 64], [1, 3, 10, 64]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 11, 64], [1, 3, 11, 64], [1, 3, 10, 64], [1, 3, 10, 64], [1, 3, 10, 64], [1, 3, 10, 64]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[PERM:%.+]] = VPU.NCE.ClusterTiling ([[COPY0]] as {{[^:]+}}: tensor<1x3x62x64xf16, {mem_space = @CMX_NN, order = #NCHW}>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x16x62x64xf16, #NHWC, @CMX_NN,
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 11, 64], [1, 16, 11, 64], [1, 16, 10, 64], [1, 16, 10, 64], [1, 16, 10, 64], [1, 16, 10, 64]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 11, 64], [1, 16, 11, 64], [1, 16, 10, 64], [1, 16, 10, 64], [1, 16, 10, 64], [1, 16, 10, 64]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]}> {
        // CHECK-NEXT:         VPU.NCE.Permute

        // CHECK:       [[COPY1:%.+]] = VPU.NCE.ClusterTiling ([[PERM]] as {{[^:]+}}: tensor<1x16x62x64xf16, {mem_space = @CMX_NN, order = #NHWC}>) ->
        // CHECK-SAME:       tensor<1x16x62x64xf16, {order = #NHWC}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[COPY1]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[IN:%.+]] = VPU.NCE.ClusterTiling ([[SLICE]] as {{[^:]+}}: tensor<1x16x62x62xf16, {order = #NHWC}>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x16x62x62xf16, #NHWC, @CMX_NN,
        // CHECK-SAME:        {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 11, 62], [1, 16, 11, 62], [1, 16, 10, 62], [1, 16, 10, 62], [1, 16, 10, 62], [1, 16, 10, 62]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 12, 62], [1, 16, 12, 62], [1, 16, 12, 62], [1, 16, 12, 62], [1, 16, 12, 62], [1, 16, 12, 62]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[COPY2:%.+]] = VPU.NCE.ClusterTiling ([[SPARSE]] as {{[^:]+}}: !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<48x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>) ->
        // CHECK-SAME:       !VPU.SparseTensor<data=!VPU.DistributedTensor<48x16x3x3xf16, #NHWC, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
        // CHECK-SAME:                         sparsity_map=!VPU.DistributedTensor<48x1x1x256xi1, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
        // CHECK-SAME:                         is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[COPY3:%.+]] = VPU.NCE.ClusterTiling ([[CST0]] as {{[^:]+}}: tensor<48x1x1x4xsi32>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[CONV:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x16x62x62xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:       [[COPY2]] as {{[^:]+}}: !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:                                                 sparsity_map=tensor<48x1x1x256xi1, {mem_space = @CMX_NN, order = #NCHW}>,
        // CHECK-SAME:                                                 is_weights,
        // CHECK-SAME:                                                 #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>,
        // CHECK-SAME:       [[COPY3]] as {{[^:]+}}: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
        // CHECK-SAME:  -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]}> {
        // CHECK-NEXT:         VPU.NCE.Convolution

        // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[CONV]] as {{[^:]+}}: tensor<1x48x60x60xf16, {mem_space = @CMX_NN, order = #NCHW}>)
        // CHECK-SAME:       -> tensor<1x48x60x60xf16> {
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        return %0 : tensor<1x48x60x60xf16>

        // CHECK:       [[COPY:%.+]] = VPU.NCE.ClusterTiling ([[ARG0]] as {{[^:]+}}: tensor<1x48x60x60xf16>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0], [0, 32, 0, 0], [0, 40, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0], [0, 32, 0, 0], [0, 40, 0, 0]]
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[SOFTMAX:%.+]] = VPU.NCE.ClusterTiling ([[COPY]] as {{[^:]+}}: tensor<1x48x60x60xf16, {mem_space = @CMX_NN, order = #NCHW}>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0], [0, 32, 0, 0], [0, 40, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0], [0, 32, 0, 0], [0, 40, 0, 0]]
        // CHECK-NEXT:         VPU.SoftMax

        // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[SOFTMAX]] as {{[^:]+}}: tensor<1x48x60x60xf16, {mem_space = @CMX_NN, order = #NCHW}>)
        // CHECK-SAME:      -> tensor<1x48x60x60xf16>
        // CHECK-NEXT:         VPU.Copy

        // CHECK: return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16> {
        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        %1 = call @foo1(%0) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        %2 = call @foo2(%1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %2 : tensor<1x48x60x60xf16>

        // CHECK:       [[COPY:%.+]] = VPU.NCE.ClusterTiling ([[ARG0]] as {{[^:]+}}: tensor<1x3x62x62xui8>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x3x62x62xui8, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 11, 62], [1, 3, 11, 62], [1, 3, 10, 62], [1, 3, 10, 62], [1, 3, 10, 62], [1, 3, 10, 62]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 11, 62], [1, 3, 11, 62], [1, 3, 10, 62], [1, 3, 10, 62], [1, 3, 10, 62], [1, 3, 10, 62]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]}>
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[CONVERT:%.+]] = VPU.NCE.ClusterTiling ([[COPY]] as {{[^:]+}}: tensor<1x3x62x62xui8, {mem_space = @CMX_NN, order = #NCHW}>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x3x62x62xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 11, 62], [1, 3, 11, 62], [1, 3, 10, 62], [1, 3, 10, 62], [1, 3, 10, 62], [1, 3, 10, 62]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 11, 62], [1, 3, 11, 62], [1, 3, 10, 62], [1, 3, 10, 62], [1, 3, 10, 62], [1, 3, 10, 62]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 42, 0], [0, 0, 52, 0]]}>
        // CHECK-NEXT:         VPU.Convert

        // CHECK:       [[COPY_BACK:%.+]] = VPU.NCE.ClusterTiling ([[CONVERT]] as {{[^:]+}}: tensor<1x3x62x62xf16, {mem_space = @CMX_NN, order = #NCHW}>)
        // CHECK-SAME:      -> tensor<1x3x62x62xf16>
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[FOO1_RES:%.+]] = call @foo1([[COPY_BACK]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       return [[FOO2_RES]] : tensor<1x48x60x60xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RepeatingBlocks
module @RepeatingBlocks {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    func.func private @main_fn1(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %shape_cast1 = VPU.ShapeCast {shape = [1, 48, 225, 16]} inputs(%arg0 : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        %permute = VPU.NCE.Permute(%shape_cast1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64} -> tensor<1x48x225x16xf16, {order = #NHWC}>
        %shape_cast2 = VPU.ShapeCast {shape = [1, 48, 60, 60]} inputs(%permute : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>

        %cst_weights_table = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_weights = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
        %conv = VPU.NCE.Convolution(%shape_cast2, %cst_weights, %cst_weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
            rawFilterShape = [48, 48, 3, 3], strides = [1, 1]
        } -> tensor<1x48x60x60xf16>

        return %conv : tensor<1x48x60x60xf16>

        // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

        // CHECK:       [[SHAPE_CAST1:%.+]] = VPU.ShapeCast {shape = [1, 48, 225, 16]} inputs([[ARG0]] : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        // CHECK:       [[INPUT_COPY1:%.+]] = VPU.NCE.ClusterTiling ([[SHAPE_CAST1:%.+]] as {{[^:]+}}: tensor<1x48x225x16xf16>)
        // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x225x16xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 38, 16], [1, 48, 38, 16], [1, 48, 38, 16], [1, 48, 37, 16], [1, 48, 37, 16], [1, 48, 37, 16]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 114, 0], [0, 0, 151, 0], [0, 0, 188, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 38, 16], [1, 48, 38, 16], [1, 48, 38, 16], [1, 48, 37, 16], [1, 48, 37, 16], [1, 48, 37, 16]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 114, 0], [0, 0, 151, 0], [0, 0, 188, 0]]}>
        // CHECK-NEXT:      VPU.Copy

        // CHECK:       [[PERM:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_COPY1]] as {{[^:]+}}: tensor<1x48x225x16xf16, {mem_space = @CMX_NN, order = #NCHW}>)
        // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x225x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 38, 16], [1, 48, 38, 16], [1, 48, 38, 16], [1, 48, 37, 16], [1, 48, 37, 16], [1, 48, 37, 16]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 114, 0], [0, 0, 151, 0], [0, 0, 188, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 38, 16], [1, 48, 38, 16], [1, 48, 38, 16], [1, 48, 37, 16], [1, 48, 37, 16], [1, 48, 37, 16]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 114, 0], [0, 0, 151, 0], [0, 0, 188, 0]]}>
        // CHECK:           VPU.NCE.Permute

        // CHECK:       [[INPUT_COPY2:%.+]] = VPU.NCE.ClusterTiling ([[PERM]] as {{[^:]+}}: tensor<1x48x225x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x225x16xf16, {order = #NHWC}> {
        // CHECK-NEXT:      VPU.Copy

        // CHECK:       [[SHAPE_CAST2:%.+]] = VPU.ShapeCast {shape = [1, 48, 60, 60]} inputs([[INPUT_COPY2]] : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[INPUT_COPY3:%.+]] = VPU.NCE.ClusterTiling ([[SHAPE_CAST2]] as {{[^:]+}}: tensor<1x48x60x60xf16, {order = #NHWC}>)
        // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 11, 60], [1, 48, 12, 60], [1, 48, 12, 60], [1, 48, 12, 60], [1, 48, 12, 60], [1, 48, 11, 60]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 19, 0], [0, 0, 29, 0], [0, 0, 39, 0], [0, 0, 49, 0]]}>
        // CHECK-NEXT:      VPU.Copy

        // CHECK:       [[COPY_WEIGHTS:%.+]] = VPU.NCE.ClusterTiling ([[CST_WEIGHTS]] as {{[^:]+}}: tensor<48x48x3x3xf16, {order = #NHWC}>)
        // CHECK-SAME:      -> !VPU.DistributedTensor<48x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        // CHECK-NEXT:      VPU.Copy

        // CHECK:       [[COPY_WEIGHTS_TABLE:%.+]] = VPU.NCE.ClusterTiling ([[CST_WEIGHTS_TABLE]] as {{[^:]+}}: tensor<48x1x1x4xsi32>)
        // CHECK-SAME:      -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        // CHECK-NEXT:      VPU.Copy

        // CHECK:       [[CONV:%.+]] = VPU.NCE.ClusterTiling (
        // CHECK-SAME:      [[INPUT_COPY3]] as {{[^:]+}}: tensor<1x48x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:      [[COPY_WEIGHTS]] as {{[^:]+}}: tensor<48x48x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        // CHECK-SAME:      [[COPY_WEIGHTS_TABLE]] as {{[^:]+}}: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
        // CHECK-SAME:  -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60], [1, 48, 10, 60]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]}>
        // CHECK:       VPU.NCE.Convolution

        // CHECK:       [[OUTPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[CONV]] as {{[^:]+}}: tensor<1x48x60x60xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x48x60x60xf16> {
        // CHECK-NEXT:      VPU.Copy

        // CHECK:       return [[OUTPUT_COPY]]
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
        %convert = VPU.Convert(%input) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        %call1 = call @main_fn1(%convert) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %call2 = call @main_fn1(%call1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %call2 : tensor<1x48x60x60xf16>

        // CHECK:       [[COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x48x60x60xf32>)
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x48x60x60xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0], [0, 32, 0, 0], [0, 40, 0, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0], [0, 32, 0, 0], [0, 40, 0, 0]]}>
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[CONVERT:%.+]] = VPU.NCE.ClusterTiling ([[COPY]] as {{[^:]+}}: tensor<1x48x60x60xf32, {mem_space = @CMX_NN, order = #NCHW}>) ->
        // CHECK-SAME:       !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0], [0, 32, 0, 0], [0, 40, 0, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60], [1, 8, 60, 60]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0], [0, 32, 0, 0], [0, 40, 0, 0]]}>
        // CHECK-NEXT:         VPU.Convert

        // CHECK:       [[COPY_BACK:%.+]] = VPU.NCE.ClusterTiling ([[CONVERT]] as {{[^:]+}}: tensor<1x48x60x60xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x48x60x60xf16>
        // CHECK-NEXT:         VPU.Copy

        // CHECK:       [[CALL1:%.+]] = call @main_fn1([[COPY_BACK]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       return [[CALL2]] : tensor<1x48x60x60xf16>
    }
}
