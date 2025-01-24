//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpu="vf-outlining-tile-threshold=1 vf-outlining-instance-threshold=2" %s | FileCheck %s --strict-whitespace
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
                      = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
              ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
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
        %cst_0 = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
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

    // CHECK: func.func private @main_outline1([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x3x62x62xf16>
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

    // CHECK: return [[COPY_BACK]] : tensor<1x3x62x62xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16> {
        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        %1 = call @foo1(%0) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        %2 = call @foo2(%1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %2 : tensor<1x48x60x60xf16>

        // CHECK:       [[OUTLINE1_RES:%.+]] = call @main_outline1([[ARG0]]) : (tensor<1x3x62x62xui8>) -> tensor<1x3x62x62xf16>
        // CHECK:       [[FOO1_RES:%.+]] = call @foo1([[OUTLINE1_RES]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
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
        %permute = VPU.NCE.Permute(%shape_cast1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>} -> tensor<1x48x225x16xf16, {order = #NHWC}>
        %shape_cast2 = VPU.ShapeCast {shape = [1, 48, 60, 60]} inputs(%permute : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>

        %cst_weights_table = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_weights = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %conv = VPU.NCE.Convolution(%shape_cast2, %cst_weights, %cst_weights_table) {
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [48, 48, 3, 3], strides = [1, 1]
        } -> tensor<1x48x60x60xf16>

        return %conv : tensor<1x48x60x60xf16>

        // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
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

    // CHECK: func.func private @main_outline1([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
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

    // CHECK: return [[COPY_BACK]] : tensor<1x48x60x60xf16>

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
        %convert = VPU.Convert(%input) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        %call1 = call @main_fn1(%convert) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %call2 = call @main_fn1(%call1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %call2 : tensor<1x48x60x60xf16>

        // CHECK:       [[OUTLINE1_RES:%.+]] = call @main_outline1([[INPUT]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16>
        // CHECK:       [[CALL1:%.+]] = call @main_fn1([[OUTLINE1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       return [[CALL2]] : tensor<1x48x60x60xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VerticalFusionOutlining {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x30x256x256xf32, {order = #NHWC}>
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x256x256xf16, {order = #NHWC}>
    }

    func.func @main(%arg0: tensor<1x30x256x256xf32, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
        %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x30x256x256xf32, {order = #NHWC}> -> tensor<1x30x256x256xf16, {order = #NHWC}>
        %1 = VPU.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 2, 0, 0]} : tensor<1x30x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

        %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]}
                    -> tensor<1x32x256x256xf16, {order = #NHWC}>
        %3 = VPU.SoftMax(%2) {axisInd = 3 : i64} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

        return %3  : tensor<1x32x256x256xf16, {order = #NHWC}>
    }
}

// CHECK:     func.func private @main_vf1([[ARG0:%.+]]: tensor<1x30x256x256xf32, {order = #NHWC}>) -> tensor<1x30x256x256xf16, {order = #NHWC}> {
// CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 30, 128, 256] : tensor<1x30x256x256xf32, {order = #NHWC}> to tensor<1x30x128x256xf32, {order = #NHWC}>
// CHECK:       [[COPY0_0:%.+]] = VPU.NCE.ClusterTiling ([[SLICE_0]] as {{[^:]+}}: tensor<1x30x128x256xf32, {order = #NHWC}>) -> !VPU.DistributedTensor<1x30x128x256xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 30, 22, 256], [1, 30, 22, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]], memory_shapes = [[1, 30, 22, 256], [1, 30, 22, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[CONVERT0:%.+]] = VPU.NCE.ClusterTiling ([[COPY0_0]] as {{[^:]+}}: tensor<1x30x128x256xf32, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x30x128x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 30, 22, 256], [1, 30, 22, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]], memory_shapes = [[1, 30, 22, 256], [1, 30, 22, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]]}> {
// CHECK-NEXT:      VPU.Convert
// CHECK:       [[COPY0_1:%.+]] = VPU.NCE.ClusterTiling ([[CONVERT0]] as {{[^:]+}}: tensor<1x30x128x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x30x128x256xf16, {order = #NHWC}> {
// CHECK-NEXT:      VPU.Copy

// CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 128, 0] [1, 30, 128, 256] : tensor<1x30x256x256xf32, {order = #NHWC}> to tensor<1x30x128x256xf32, {order = #NHWC}>
// CHECK:       [[COPY1_0:%.+]] = VPU.NCE.ClusterTiling ([[SLICE_1]] as {{[^:]+}}: tensor<1x30x128x256xf32, {order = #NHWC}>) -> !VPU.DistributedTensor<1x30x128x256xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 30, 22, 256], [1, 30, 22, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]], memory_shapes = [[1, 30, 22, 256], [1, 30, 22, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[CONVERT1:%.+]] = VPU.NCE.ClusterTiling ([[COPY1_0]] as {{[^:]+}}: tensor<1x30x128x256xf32, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x30x128x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 30, 22, 256], [1, 30, 22, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]], memory_shapes = [[1, 30, 22, 256], [1, 30, 22, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256], [1, 30, 21, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 44, 0], [0, 0, 65, 0], [0, 0, 86, 0], [0, 0, 107, 0]]}> {
// CHECK-NEXT:      VPU.Convert
// CHECK:       [[COPY1_1:%.+]] = VPU.NCE.ClusterTiling ([[CONVERT1]] as {{[^:]+}}: tensor<1x30x128x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x30x128x256xf16, {order = #NHWC}> {
// CHECK-NEXT:      VPU.Copy

// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[COPY0_1]], [[COPY1_1]])
// CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 128, 0]]}
// CHECK-SAME:           : tensor<1x30x128x256xf16, {order = #NHWC}>, tensor<1x30x128x256xf16, {order = #NHWC}> -> tensor<1x30x256x256xf16, {order = #NHWC}>
// CHECK:       return [[CONCAT]] : tensor<1x30x256x256xf16, {order = #NHWC}>

// CHECK:     func.func private @main_vf2([[ARG0:%.+]]: tensor<1x30x256x256xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
// CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
// CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
// CHECK:       [[EXAPND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 2, 0, 0]} : tensor<1x30x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[EXAPND]] [0, 0, 0, 0] [1, 32, 87, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x87x256xf16, {order = #NHWC}>
// CHECK:       [[COPY0_0:%.+]] = VPU.NCE.ClusterTiling ([[SLICE_0]] as {{[^:]+}}: tensor<1x32x87x256xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x32x87x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 45, 0], [0, 0, 59, 0], [0, 0, 73, 0]], memory_shapes = [[1, 32, 16, 256], [1, 32, 17, 256], [1, 32, 16, 256], [1, 32, 16, 256], [1, 32, 16, 256], [1, 32, 16, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[COPY0_1:%.+]] = VPU.NCE.ClusterTiling ([[CST_0]] as {{[^:]+}}: tensor<32x32x3x3xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[COPY0_2:%.+]] = VPU.NCE.ClusterTiling ([[CST]] as {{[^:]+}}: tensor<32x1x1x4xsi32>) -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[CONV0:%.+]] = VPU.NCE.ClusterTiling ([[COPY0_0]] as {{[^:]+}}: tensor<1x32x87x256xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[COPY0_1]] as {{[^:]+}}: tensor<32x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[COPY0_2]] as {{[^:]+}}: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x86x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 44, 0], [0, 0, 58, 0], [0, 0, 72, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 44, 0], [0, 0, 58, 0], [0, 0, 72, 0]]}> {
// CHECK-NEXT:      VPU.NCE.Convolution
// CHECK:       [[CAST0:%.+]] = VPU.DistributedCast([[CONV0]] : !VPU.DistributedTensor<1x32x86x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 44, 0], [0, 0, 58, 0], [0, 0, 72, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 44, 0], [0, 0, 58, 0], [0, 0, 72, 0]]}>) -> !VPU.DistributedTensor<1x32x86x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 44, 0], [0, 0, 58, 0], [0, 0, 72, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 44, 0], [0, 0, 58, 0], [0, 0, 72, 0]]}>
// CHECK:       [[SOFTMAX0:%.+]] = VPU.NCE.ClusterTiling ([[CAST0]] as {{[^:]+}}: tensor<1x32x86x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x86x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 44, 0], [0, 0, 58, 0], [0, 0, 72, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 44, 0], [0, 0, 58, 0], [0, 0, 72, 0]]}> {
// CHECK-NEXT:      VPU.SoftMax
// CHECK:       [[COPY0_3:%.+]] = VPU.NCE.ClusterTiling ([[SOFTMAX0]] as {{[^:]+}}: tensor<1x32x86x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x86x256xf16, {order = #NHWC}> {
// CHECK-NEXT:      VPU.Copy

// CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[EXAPND]] [0, 0, 85, 0] [1, 32, 87, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x87x256xf16, {order = #NHWC}>
// CHECK:       [[COPY1_0:%.+]] = VPU.NCE.ClusterTiling ([[SLICE_1]] as {{[^:]+}}: tensor<1x32x87x256xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x32x87x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 45, 0], [0, 0, 59, 0], [0, 0, 73, 0]], memory_shapes = [[1, 32, 17, 256], [1, 32, 16, 256], [1, 32, 16, 256], [1, 32, 16, 256], [1, 32, 16, 256], [1, 32, 16, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[COPY1_1:%.+]] = VPU.NCE.ClusterTiling ([[CST_0]] as {{[^:]+}}: tensor<32x32x3x3xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[COPY1_2:%.+]] = VPU.NCE.ClusterTiling ([[CST]] as {{[^:]+}}: tensor<32x1x1x4xsi32>) -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[CONV1:%.+]] = VPU.NCE.ClusterTiling ([[COPY1_0]] as {{[^:]+}}: tensor<1x32x87x256xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[COPY1_1]] as {{[^:]+}}: tensor<32x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[COPY1_2]] as {{[^:]+}}: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x85x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}> {
// CHECK-NEXT:      VPU.NCE.Convolution
// CHECK:       [[CAST1:%.+]] = VPU.DistributedCast([[CONV1]] : !VPU.DistributedTensor<1x32x85x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}>) -> !VPU.DistributedTensor<1x32x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}>
// CHECK:       [[SOFTMAX1:%.+]] = VPU.NCE.ClusterTiling ([[CAST1]] as {{[^:]+}}: tensor<1x32x85x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}> {
// CHECK-NEXT:      VPU.SoftMax
// CHECK:       [[COPY1_3:%.+]] = VPU.NCE.ClusterTiling ([[SOFTMAX1]] as {{[^:]+}}: tensor<1x32x85x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x85x256xf16, {order = #NHWC}> {
// CHECK-NEXT:      VPU.Copy

// CHECK:       [[SLICE_2:%.+]] = VPU.Slice [[EXAPND]] [0, 0, 170, 0] [1, 32, 86, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x86x256xf16, {order = #NHWC}>
// CHECK:       [[COPY2_0:%.+]] = VPU.NCE.ClusterTiling ([[SLICE_2]] as {{[^:]+}}: tensor<1x32x86x256xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x32x86x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 44, 0], [0, 0, 58, 0], [0, 0, 72, 0]], memory_shapes = [[1, 32, 17, 256], [1, 32, 16, 256], [1, 32, 16, 256], [1, 32, 16, 256], [1, 32, 16, 256], [1, 32, 15, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[COPY2_1:%.+]] = VPU.NCE.ClusterTiling ([[CST_0]] as {{[^:]+}}: tensor<32x32x3x3xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3], [32, 32, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[COPY2_2:%.+]] = VPU.NCE.ClusterTiling ([[CST]] as {{[^:]+}}: tensor<32x1x1x4xsi32>) -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
// CHECK-NEXT:      VPU.Copy
// CHECK:       [[CONV2:%.+]] = VPU.NCE.ClusterTiling ([[COPY2_0]] as {{[^:]+}}: tensor<1x32x86x256xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[COPY2_1]] as {{[^:]+}}: tensor<32x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[COPY2_2]] as {{[^:]+}}: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x32x85x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}> {
// CHECK-NEXT:      VPU.NCE.Convolution
// CHECK:       [[CAST2:%.+]] = VPU.DistributedCast([[CONV2]] : !VPU.DistributedTensor<1x32x85x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}>) -> !VPU.DistributedTensor<1x32x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}>
// CHECK:       [[SOFTMAX2:%.+]] = VPU.NCE.ClusterTiling ([[CAST2]] as {{[^:]+}}: tensor<1x32x85x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]], memory_shapes = [[1, 32, 15, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256], [1, 32, 14, 256]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 29, 0], [0, 0, 43, 0], [0, 0, 57, 0], [0, 0, 71, 0]]}> {
// CHECK-NEXT:      VPU.SoftMax
// CHECK:       [[COPY2_3:%.+]] = VPU.NCE.ClusterTiling ([[SOFTMAX2]] as {{[^:]+}}: tensor<1x32x85x256xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x85x256xf16, {order = #NHWC}> {
// CHECK-NEXT:      VPU.Copy


// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[COPY0_3]], [[COPY1_3]], [[COPY2_3]])
// CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]]}
// CHECK-SAME:           : tensor<1x32x86x256xf16, {order = #NHWC}>, tensor<1x32x85x256xf16, {order = #NHWC}>, tensor<1x32x85x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:       return [[CONCAT]] : tensor<1x32x256x256xf16, {order = #NHWC}>

// CHECK:     func.func @main([[ARG0:%.+]]: tensor<1x30x256x256xf32, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
// CHECK:       [[CALL0:%.+]] = call @main_vf1([[ARG0]]) : (tensor<1x30x256x256xf32, {order = #NHWC}>) -> tensor<1x30x256x256xf16, {order = #NHWC}>
// CHECK:       [[CALL1:%.+]] = call @main_vf2([[CALL0]]) : (tensor<1x30x256x256xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:       return [[CALL1]] : tensor<1x32x256x256xf16, {order = #NHWC}>
