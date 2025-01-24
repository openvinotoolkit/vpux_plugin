//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --shift-dpu-workloads-start %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPU.DistributedTensor<
    1x16x30x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 4,
    uniform_distributed_segments
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x16x30x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4,
    uniform_distributed_segments
}>

!WeightsDistributed = !VPU.DistributedTensor<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    uniform_distributed_segments
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    uniform_distributed_segments
}>

!Input_DDR = tensor<1x16x30x33xf16, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = tensor<16x16x1x1xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = tensor<16x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = tensor<1x16x30x33xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsStub_CMX = tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvSOHOverlapped
func.func @ConvSOHOverlapped(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare tensor<16x16x1x1xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>

    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %weights_cmx = VPU.NCE.ClusterTiling(%weights as %arg1: !Weights_DDR) -> !WeightsDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsStub_CMX
        VPU.Yield %0
    }

    %wt_cmx = VPU.NCE.ClusterTiling(%wt as %arg1: !WeightsTable_DDR) -> !WeightsTableDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_cmx as %arg2: !WeightsStub_CMX,
              %wt_cmx as %arg3: !WeightsTableStub_CMX)
              -> !OutputDistributed {
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> !OutputStub_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 16, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 2 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 23, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 3 : i64}
            }
        VPU.Yield %0
    }
    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    // CHECK:       [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:                   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                   strides = [1, 1]
    // CHECK-SAME:    } -> tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           DPU.Workload outOffsets [0, 0, 1, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:           DPU.Workload outOffsets [0, 0, 1, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:           DPU.Workload outOffsets [0, 0, 1, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 3 : i64}
    // CHECK:       VPU.Yield [[RES4]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDataDistributed = !VPU.DistributedTensor<
    1x16x30x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 4,
    uniform_distributed_segments
}>

!InputSMDistributed = !VPU.DistributedTensor<
    1x16x30x33xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 4,
    uniform_distributed_segments
}>

!Input_CMX = !VPU.SparseTensor<data=!InputDataDistributed,
                               sparsity_map=!InputSMDistributed>

!OutputDataDistributed = !VPU.DistributedTensor<
    1x16x30x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4,
    uniform_distributed_segments
}>

!OutputSMDistributed = !VPU.DistributedTensor<
    1x16x30x33xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4,
    uniform_distributed_segments
}>

!Output_CMX = !VPU.SparseTensor<data=!OutputDataDistributed,
                                sparsity_map=!OutputSMDistributed>

!WeightsDistributed = !VPU.DistributedTensor<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    uniform_distributed_segments
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    uniform_distributed_segments
}>

!Weights_DDR = tensor<16x16x1x1xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = tensor<16x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>

!InputDataStub_CMX = tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InputSMStub_CMX = tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
!InputStub_CMX = !VPU.SparseTensor<data=!InputDataStub_CMX,
                                   sparsity_map=!InputSMStub_CMX>

!WeightsStub_CMX = tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

!OutputDataStub_CMX = tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
!OutputSMStub_CMX = tensor<1x16x30x33xi1, {mem_space = @CMX_NN, order = #NHWC}>
!OutputStub_CMX = !VPU.SparseTensor<data=!OutputDataStub_CMX,
                                    sparsity_map=!OutputSMStub_CMX>

// CHECK-LABEL: @SparseConvSOHOverlapped
func.func @SparseConvSOHOverlapped(%arg0: !InputDataDistributed, %arg1: !InputSMDistributed) -> !Output_CMX {
    %weights = const.Declare tensor<16x16x1x1xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1) -> !Input_CMX

    %weights_cmx = VPU.NCE.ClusterTiling(%weights as %arg2: !Weights_DDR) -> !WeightsDistributed {
        %0 = VPU.Copy(%arg2) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsStub_CMX
        VPU.Yield %0
    }

    %wt_cmx = VPU.NCE.ClusterTiling(%wt as %arg2: !WeightsTable_DDR) -> !WeightsTableDistributed {
        %0 = VPU.Copy(%arg2) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_sparse as %arg2: !InputStub_CMX,
              %weights_cmx as %arg3: !WeightsStub_CMX,
              %wt_cmx as %arg4: !WeightsTableStub_CMX)
              -> !Output_CMX {
        %0 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> !OutputStub_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 16, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 2 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 23, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 3 : i64}
            }
        VPU.Yield %0
    }

    return %output_cmx: !Output_CMX

    // CHECK:       [[RES4:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:                   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                   strides = [1, 1]
    // CHECK-SAME:    } -> !VPU.SparseTensor<data=tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x16x30x33xi1, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           DPU.Workload outOffsets [0, 0, 1, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:           DPU.Workload outOffsets [0, 0, 1, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:           DPU.Workload outOffsets [0, 0, 1, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 3 : i64}
    // CHECK:       VPU.Yield [[RES4]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPU.DistributedTensor<
    1x16x30x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x16x30x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsDistributed = !VPU.DistributedTensor<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = tensor<1x16x30x33xf16, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = tensor<16x16x1x1xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = tensor<16x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = tensor<1x16x30x33xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsStub_CMX = tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvSOHOverlappedMultipleWorkloads
func.func @ConvSOHOverlappedMultipleWorkloads(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare tensor<16x16x1x1xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>

    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %weights_cmx = VPU.NCE.ClusterTiling(%weights as %arg1: !Weights_DDR) -> !WeightsDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsStub_CMX
        VPU.Yield %0
    }

    %wt_cmx = VPU.NCE.ClusterTiling(%wt as %arg1: !WeightsTable_DDR) -> !WeightsTableDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_cmx as %arg2: !WeightsStub_CMX,
              %wt_cmx as %arg3: !WeightsTableStub_CMX)
              -> !OutputDistributed {
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> !OutputStub_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 15, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 23, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
            }
        VPU.Yield %0
    }
    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    // CHECK:      [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  strides = [1, 1]
    // CHECK-SAME:   } -> tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:          DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:          DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:          DPU.Workload outOffsets [0, 0, 1, 0] outSizes [1, 16, 8, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:          DPU.Workload outOffsets [0, 0, 9, 0] outSizes [1, 16, 7, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:      VPU.Yield [[RES4]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPU.DistributedTensor<
    1x16x33x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x16x33x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [2, 2],
    num_clusters = 3,
    uniform_distributed_segments
}>

!WeightsDistributed = !VPU.DistributedTensor<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3,
    uniform_distributed_segments
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3,
    uniform_distributed_segments
}>

!Input_DDR = tensor<1x16x33x33xf16, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = tensor<16x16x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = tensor<16x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = tensor<1x16x33x33xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = tensor<1x16x33x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsStub_CMX = tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x16x33x33xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvSOHOverlappedNoOverlapAtStart
func.func @ConvSOHOverlappedNoOverlapAtStart(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare tensor<16x16x3x3xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>

    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %weights_cmx = VPU.NCE.ClusterTiling(%weights as %arg1: !Weights_DDR) -> !WeightsDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsStub_CMX
        VPU.Yield %0
    }

    %wt_cmx = VPU.NCE.ClusterTiling(%wt as %arg1: !WeightsTable_DDR) -> !WeightsTableDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_cmx as %arg2: !WeightsStub_CMX,
              %wt_cmx as %arg3: !WeightsTableStub_CMX)
              -> !OutputDistributed {
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [16, 16, 3, 3],
                strides = [1, 1]
            } -> !OutputStub_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 11, 33] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 11, 0] outSizes [1, 16, 11, 33] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 22, 0] outSizes [1, 16, 11, 33] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 2 : i64}
            }
        VPU.Yield %0
    }
    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    // CHECK:      [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:                  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                  strides = [1, 1]
    // CHECK-SAME:   } -> tensor<1x16x33x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:          DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 11, 33] <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:          DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 11, 33] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:          DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 11, 33] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:      VPU.Yield [[RES4]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPU.DistributedTensor<
    1x16x30x33xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x32x30x33xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsDistributed = !VPU.DistributedTensor<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = tensor<1x16x30x33xf16, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = tensor<32x16x1x1xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = tensor<1x32x30x33xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = tensor<1x16x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsStub_CMX = tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x32x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @ConvSOKNoChange
func.func @ConvSOKNoChange(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare tensor<32x16x1x1xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN}>

    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %weights_cmx = VPU.NCE.ClusterTiling(%weights as %arg1: !Weights_DDR) -> !WeightsDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsStub_CMX
        VPU.Yield %0
    }

    %wt_cmx = VPU.NCE.ClusterTiling(%wt as %arg1: !WeightsTable_DDR) -> !WeightsTableDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_cmx as %arg2: !WeightsStub_CMX,
              %wt_cmx as %arg3: !WeightsTableStub_CMX)
              -> !OutputDistributed {
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [32, 16, 1, 1],
                strides = [1, 1]
            } -> !OutputStub_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 30, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 30, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
            }
        VPU.Yield %0
    }
    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    // CHECK:      [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                  strides = [1, 1]
    // CHECK-SAME:   } -> tensor<1x32x30x33xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:          DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 30, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:          DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 16, 30, 33] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:      VPU.Yield [[RES4]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPU.DistributedTensor<
    1x128x32x64xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}
>

!OutputDistributed = !VPU.DistributedTensor<
    1x128x32x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}
>

!Input_DDR = tensor<1x128x32x64xf16, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = tensor<1x128x32x64xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = tensor<1x128x32x64xf16, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x128x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteSOK
func.func @NCEPermuteSOK(%arg0: !Input_DDR) -> !Output_DDR {
    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (%input_cmx as %arg1: !InputStub_CMX) -> !OutputDistributed {
        %0 = VPU.NCE.Permute(%arg1) {
                dstElemType = f16,
                dstOrder = #NHWC,
                expandedChannels = 128 : i64,
                minimumHardwareExecutionCost = 5442 : i64,
                ppe = #VPU.PPEStub<>
            } -> !OutputStub_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
                VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
                VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 3 : i64}
            }
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    // CHECK:      [[RES4:%.*]] = VPU.NCE.Permute(%arg1)
    // CHECK-SAME:   } -> tensor<1x128x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 2 : i64}
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 32, 64] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 3 : i64}
    // CHECK:      VPU.Yield [[RES4]]
}
