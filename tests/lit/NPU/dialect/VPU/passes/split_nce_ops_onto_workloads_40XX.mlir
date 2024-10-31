//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-NCE-ops-onto-workloads %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2
}>

!Output_CMX = !VPU.DistributedTensor<
    1x4x224x224xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 3 , right = 2, top = 3, bottom = 2>,
    strides = [2, 2],
    num_clusters = 2
}>

!Input_DDR = tensor<1x3x224x224xf16>
!InputStub_CMX = tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>
!OutputStub_CMX = tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteDifferentOverlap
func.func @NCEPermuteDifferentOverlap(%arg0: !Input_DDR) -> !Output_CMX {
    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !Input_CMX {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling (%input_cmx as %arg2: !InputStub_CMX) -> !Output_CMX {
        %0 = VPU.NCE.Permute(%arg2) {
                dstElemType = !quant.uniform<u8:f16, 1.000000e+00>,
                dstOrder = #NHWC,
                expandedChannels = 4 : i64,
                opaque_ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>
        } -> !OutputStub_CMX
        VPU.Yield %0
    }

    return %output : !Output_CMX

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64
    // CHECK-SAME:      -> tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 4, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 112, 0] outSizes [1, 4, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x128x32x64xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
}>

!Output_CMX = !VPU.DistributedTensor<
    1x128x32x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
}>

!Input_DDR = tensor<1x128x32x64xf16>
!InputStub_CMX = tensor<1x128x32x64xf16, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x128x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteSOC
func.func @NCEPermuteSOC(%arg0: !Input_DDR) -> !Output_CMX {
    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !Input_CMX {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling (%input_cmx as %arg1: !InputStub_CMX) -> !Output_CMX {
        %0 = VPU.NCE.Permute(%arg1) {
                dstElemType = f16,
                dstOrder = #NHWC,
                expandedChannels = 128 : i64,
                opaque_ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>
        } -> !OutputStub_CMX
        VPU.Yield %0
    }

    return %output : !Output_CMX

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      dstElemType = f16, dstOrder = #NHWC, expandedChannels = 128 : i64
    // CHECK-SAME:      -> tensor<1x128x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 32, 32, 64]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 32, 0, 0] outSizes [1, 32, 32, 64]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 64, 0, 0] outSizes [1, 32, 32, 64]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 2

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 96, 0, 0] outSizes [1, 32, 32, 64]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 3
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x80x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 32, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x80x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 32, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0]]
}>

!InputStub_CMX = tensor<1x80x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!OutputStub_CMX = tensor<1x80x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @SOKDistributedSEGOutput(%input_cmx: !InputDistributed) -> !OutputDistributed {
    %output_cmx = VPU.NCE.ClusterTiling (%input_cmx as %arg1: !InputStub_CMX)
              -> !OutputDistributed {
        %0 = VPU.NCE.MaxPool(%arg1) {
                opaque_ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                strides = [1, 1],
                kernel_size = [3, 3]
            } -> !OutputStub_CMX
        VPU.Yield %0
    }

    return %output_cmx: !OutputDistributed

    // CHECK:      VPU.NCE.MaxPool
    // CHECK-SAME:         kernel_size = [3, 3]
    // CHECK-SAME:         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:         strides = [1, 1]
    // CHECK-SAME:   } -> tensor<1x80x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:          DPU.Workload
    // CHECK-SAME:         outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16]
    // CHECK-SAME:         <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:         <CUBOID_16x16> attributes {cluster_id = 0 : i64}

    // CHECK:          DPU.Workload
    // CHECK-SAME:         outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16]
    // CHECK-SAME:         <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:         <CUBOID_16x16> attributes {cluster_id = 1 : i64}

    // CHECK:          DPU.Workload
    // CHECK-SAME:         outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16]
    // CHECK-SAME:         <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:         <CUBOID_16x16> attributes {cluster_id = 2 : i64}

    // CHECK:          DPU.Workload
    // CHECK-SAME:         outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16]
    // CHECK-SAME:         <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:         <CUBOID_16x16> attributes {cluster_id = 3 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x80x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 32, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x80x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 80, 16, 16], [1, 80, 16, 16], [1, 80, 16, 16], [1, 80, 16, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputStub_CMX = tensor<1x80x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!OutputStub_CMX = tensor<1x80x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @SOKDistributedDUPSEGOutput(%input_cmx: !InputDistributed) -> !OutputDistributed {
    %output_cmx = VPU.NCE.ClusterTiling (%input_cmx as %arg1: !InputStub_CMX)
              -> !OutputDistributed {
        %0 = VPU.NCE.MaxPool(%arg1) {
                opaque_ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                strides = [1, 1],
                kernel_size = [3, 3]
            } -> !OutputStub_CMX
        VPU.Yield %0
    }

    return %output_cmx: !OutputDistributed

    // CHECK:      VPU.NCE.MaxPool
    // CHECK-SAME:         kernel_size = [3, 3]
    // CHECK-SAME:         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:         strides = [1, 1]
    // CHECK-SAME:   } -> tensor<1x80x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:          DPU.Workload
    // CHECK-SAME:         outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16]
    // CHECK-SAME:         <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:         <CUBOID_16x16> attributes {cluster_id = 0 : i64}

    // CHECK:          DPU.Workload
    // CHECK-SAME:         outOffsets [0, 32, 0, 0] outSizes [1, 16, 16, 16]
    // CHECK-SAME:         <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:         <CUBOID_16x16> attributes {cluster_id = 1 : i64}

    // CHECK:          DPU.Workload
    // CHECK-SAME:         outOffsets [0, 48, 0, 0] outSizes [1, 16, 16, 16]
    // CHECK-SAME:         <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:         <CUBOID_16x16> attributes {cluster_id = 2 : i64}

    // CHECK:          DPU.Workload
    // CHECK-SAME:         outOffsets [0, 64, 0, 0] outSizes [1, 16, 16, 16]
    // CHECK-SAME:         <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:         <CUBOID_16x16> attributes {cluster_id = 3 : i64}
}
