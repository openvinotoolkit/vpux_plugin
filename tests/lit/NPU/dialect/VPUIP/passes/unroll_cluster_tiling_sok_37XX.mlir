//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-cluster-tiling --canonicalize  %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x32x32x32xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x16x1x1xf16, #NHWC>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!InputStub_CMX0 = memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
!InputStub_CMX1 = memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
!OutputStub_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<32x16x1x1xf16, #NHWC, @CMX_NN>

//CHECK-LABEL: @UnrollNNDMA
func.func @UnrollNNDMA(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %input1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !InputStub_CMX0
    %input2 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !InputStub_CMX1

    %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed

    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33280> -> !ParentOutputDistributed
    %out_cmx1 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33280> -> !OutputDistributed
    %out_cmx2 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <66048> -> !OutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Simulate 1st task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%input1: !InputStub_CMX0) outputs(%out_cmx1 : !OutputDistributed) -> !OutputDistributed
    }

    // Simulate 2st task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%input2: !InputStub_CMX1) outputs(%out_cmx2 : !OutputDistributed) -> !OutputDistributed
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:    [[WEIGHTS1_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS2_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x32x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:        [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits(%1 : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x32x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x32x32x32xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x16x1x1xf16, #NHWC>
!WeightsTable_DDR = memref<32x1x1x4xsi32, #NCHW>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NCHW, @CMX_NN>


//CHECK-LABEL: @UnrollNCE
func.func @UnrollNCE(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table_cst: !WeightsTable_DDR) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input_cmx : !ParentInputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !ParentInputDistributed)
                    parent_output(%parent_out_cmx : !ParentOutputDistributed)
                    outputs(%parent_out_cmx : !ParentOutputDistributed)
                        -> !ParentOutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 31, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 0, 16], outEnd = [31, 31, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:    [[WEIGHTS_TABLE1_CST:%.*]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CST:%.*]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS1_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS2_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x32x32x32xf16, #NHWC, @DDR>

    //CHECK:        [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS_TABLE1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:        [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weight    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 16 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits(%1 : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x32x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed_1 = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed_1 = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed_1 = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed_1 = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!ParentInputDistributed_2 = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed_2 = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed_2 = !VPUIP.DistributedBuffer<
    16x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed_2 = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>

!Weights1_DDR = memref<32x16x1x1xf16, #NHWC>
!WeightsTable1_DDR = memref<32x1x1x4xsi32, #NCHW>

!InputStub1_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub1_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub1_CMX = memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub1_CMX = memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

!Weights2_DDR = memref<16x32x1x1xf16, #NHWC>
!WeightsTable2_DDR = memref<16x1x1x4xsi32, #NCHW>

!InputStub2_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!OutputStub2_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub2_CMX = memref<16x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub2_CMX = memref<16x1x1x4xsi32, #NCHW, @CMX_NN>


//CHECK-LABEL: @UnrollNCESequence
func.func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights1_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table1_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %weights2_cst = const.Declare memref<16x32x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table2_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> <0> -> !Output_DDR

    // CMX buffers/ 1st task
    %parent_input1_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed_1
    %weights1 = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed_1
    %weights_table1 = VPURT.DeclareBuffer <CMX_NN> <33280> -> !WeightsTableDistributed_1
    %parent_out1_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed_1

    // CMX buffers/ 2nd task
    %parent_input2_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentInputDistributed_2
    %weights2 = VPURT.DeclareBuffer <CMX_NN> <99072> -> !WeightsDistributed_2
    %weights_table2 = VPURT.DeclareBuffer <CMX_NN> <99584> -> !WeightsTableDistributed_2
    %parent_out2_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentOutputDistributed_2

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input1_cmx: !ParentInputDistributed_1) -> !ParentInputDistributed_1
    }

    // Upload weights/ 1st task
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights1_cst: !Weights1_DDR) outputs(%weights1: !WeightsDistributed_1) -> !WeightsDistributed_1
    }

    // Upload weights table/ 1st task
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table1_cst: !WeightsTable1_DDR) outputs(%weights_table1: !WeightsTableDistributed_1) -> !WeightsTableDistributed_1
    }

    // Cluster tiling/ 1st task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input1_cmx : !ParentInputDistributed_1)
                    weights(%weights1 : !WeightsDistributed_1)
                    weight_table(%weights_table1 : !WeightsTableDistributed_1)
                    parent_input(%parent_input1_cmx : !ParentInputDistributed_1)
                    parent_output(%parent_out1_cmx : !ParentOutputDistributed_1)
                    outputs(%parent_out1_cmx : !ParentOutputDistributed_1)
                        -> !ParentOutputDistributed_1 variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 31, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 0, 16], outEnd = [31, 31, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Upload weights/ 2nd task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights2_cst: !Weights2_DDR) outputs(%weights2: !WeightsDistributed_2) -> !WeightsDistributed_2
    }

    // Upload weights table/ 2nd task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table2_cst: !WeightsTable2_DDR) outputs(%weights_table2: !WeightsTableDistributed_2) -> !WeightsTableDistributed_2
    }

    // Cluster tiling/ 2nd task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = #VPUIP.nce_task_type<CONV>
                     }  input(%parent_input2_cmx : !ParentInputDistributed_2)
                         weights(%weights2 : !WeightsDistributed_2)
                         weight_table(%weights_table2 : !WeightsTableDistributed_2)
                         parent_input(%parent_input2_cmx : !ParentInputDistributed_2)
                         parent_output(%parent_out2_cmx : !ParentOutputDistributed_2)
                         outputs(%parent_out2_cmx : !ParentOutputDistributed_2)
                             -> !ParentOutputDistributed_2 variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [31, 31, 7],
                                mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                outStart = [0, 0, 8], outEnd = [31, 31, 15],
                                mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out2_cmx: !ParentOutputDistributed_2) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:    [[WEIGHTS_TABLE1_CST_2ND_TASK:%.*]] = const.Declare memref<8x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [8, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CST_2ND_TASK:%.*]] = const.Declare memref<8x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>, [#const.SubView<[8, 0, 0, 0], [8, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS1_CST_2ND_TASK:%.*]] = const.Declare memref<8x32x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [8, 32, 1, 1]>, #const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS2_CST_2ND_TASK:%.*]] = const.Declare memref<8x32x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.SubView<[8, 0, 0, 0], [8, 32, 1, 1]>, #const.Reorder<#NHWC>]

    //CHECK-DAG:    [[WEIGHTS_TABLE1_CST_1ST_TASK:%.*]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CST_1ST_TASK:%.*]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS1_CST_1ST_TASK:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS2_CST_1ST_TASK:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[PARENT_IN_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>


    //CHECK:    [[PARENT_IN_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}
    //CHECK:    [[IN1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights/ 1st task
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS1_CST_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights/ 1st task
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table/ 1st task
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST_1ST_TASK]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table/ 1st task
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST_1ST_TASK]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    // 1st task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX_1ST_TASK]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 1st task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 16 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX_1ST_TASK]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Upload 1st part of weights/ 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS1_CST_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights/ 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table/ 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST_2ND_TASK]] : memref<8x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table/ 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST_2ND_TASK]] : memref<8x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }


    // 2nd task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 7], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 8 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 15], outStart = [0, 0, 8],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

// Define non contiguous output using strides. Below is an example case
// that would happen in case of CMX concat along "C" axis in NHWC order
// Full output tensor is 1x32x32x32xf16 but given NCE task produces
// only half of it - 1x16x32x32xf16
!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, #NCHW, @CMX_NN>

func.func @UnrollNCENonContiguousOutput(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare !Weights_DDR =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table_cst: !WeightsTable_DDR) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input_cmx : !ParentInputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !ParentInputDistributed)
                    parent_output(%parent_out_cmx : !ParentOutputDistributed)
                    outputs(%parent_out_cmx : !ParentOutputDistributed)
                        -> !ParentOutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [15, 31, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 0, 16], outEnd = [15, 31, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:    [[WEIGHTS_TABLE1_CST:%.*]] = const.Declare memref<8x1x1x4xsi32, @DDR> = dense<1> : tensor<16x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [8, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CST:%.*]] = const.Declare memref<8x1x1x4xsi32, @DDR> = dense<1> : tensor<16x1x1x4xsi32>, [#const.SubView<[8, 0, 0, 0], [8, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS1_CST:%.*]] = const.Declare memref<8x16x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [8, 16, 1, 1]>, #const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS2_CST:%.*]] = const.Declare memref<8x16x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.SubView<[8, 0, 0, 0], [8, 16, 1, 1]>, #const.Reorder<#NHWC>]

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>

    //CHECK:        [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS_TABLE1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, [@CMX_NN, 0]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<8x16x1x1xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY]] : memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<8x16x1x1xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY]] : memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<8x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<8x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }


    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 8 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits(%1 : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!typeCmxDistributed = !VPUIP.DistributedBuffer<
    1x4x512x1xf16, #NCWH, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!type_CMX_memref = memref<1x4x512x1xf16, #NCWH, @CMX_NN>


!Input_DDR  = memref<1x4x512x1xf16, #NCWH, @DDR>
!Output_DDR = memref<1x4x512x1xf16, #NCWH, @DDR>


VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @UnrollSWOpInterface
func.func @UnrollSWOpInterface(%input0: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %395 = VPURT.DeclareBuffer <DDR> <4096> -> !Input_DDR
    %302 = VPURT.DeclareBuffer <DDR> <0> -> !Output_DDR

    %300 = VPURT.DeclareBuffer <CMX_NN> <0> -> !typeCmxDistributed
    %301 = VPURT.DeclareBuffer <CMX_NN> <2048> -> !typeCmxDistributed


    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %399 = VPUIP.NNDMA {port = 0 : i64} inputs(%395 : !Input_DDR) outputs(%300 : !typeCmxDistributed) -> !typeCmxDistributed
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
          @VPU.SW::@builtin_MVN inputs(%300 as %arg4: !typeCmxDistributed) outputs(%301 as %arg5: !typeCmxDistributed) on tile 0 -> !typeCmxDistributed {
          VPUIP.SW.Kernel.run {
            attrs = [false, true, 1.0013580322265625E-5]}(%arg4, %arg5) : !typeCmxDistributed, !typeCmxDistributed
        }
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier)  attributes {isTrailingSWLayer = false} {
        %399 = VPUIP.NNDMA {port = 0 : i64} inputs(%301 : !typeCmxDistributed) outputs(%302 : !Output_DDR) -> !Output_DDR
    }


    return %output: !Output_DDR

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <4096> -> memref<1x2x512x1xf16, #NCWH, @DDR>
    //CHECK:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <6144> -> memref<1x2x512x1xf16, #NCWH, @DDR>
    //CHECK:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x512x1xf16, #NCWH, @DDR>
    //CHECK:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x2x512x1xf16, #NCWH, @DDR>

    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>

    //CHECK:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>

    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x2x512x1xf16, #NCWH, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x2x512x1xf16, #NCWH, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>)
    //CHECK:        }

    // sw tasks
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.SW.Kernel
    //CHECK-SAME:       inputs([[IN1_CMX]] as %arg2: memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_CMX]] as %arg3: memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.SW.Kernel
    //CHECK-SAME:       inputs([[IN2_CMX]] as %arg2: memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_CMX]] as %arg3: memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback 1st part of output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x2x512x1xf16, #NCWH, @DDR>)
    //CHECK:        }

    // Copyback 2nd part of output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x2x512x1xf16, #NCWH, @DDR>)
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x32x32x32xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x16x1x1xf16, #NHWC>
!WeightsTable_DDR = memref<32x1x1x4xsi32, #NCHW>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NCHW, @CMX_NN>


//CHECK-LABEL: @UnrollNCEWithDuplicatedAndSegmentedWeights
func.func @UnrollNCEWithDuplicatedAndSegmentedWeights(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> <33792> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table_cst: !WeightsTable_DDR) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input_cmx : !ParentInputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !ParentInputDistributed)
                    parent_output(%parent_out_cmx : !ParentOutputDistributed)
                    outputs(%parent_out_cmx : !ParentOutputDistributed)
                        -> !ParentOutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 31, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 0, 16], outEnd = [31, 31, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:        [[WEIGHTS_TABLE1_CST:%.*]] = const.Declare memref<16x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK:        [[WEIGHTS_TABLE2_CST:%.*]] = const.Declare memref<16x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK:        [[WEIGHTS_CST:%.*]] = const.Declare memref<32x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]


    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x32x32x32xf16, #NHWC, @DDR>

    //CHECK:        [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[PARENT_WEIGHTS:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <32768> -> !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    //CHECK:        [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33792> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33792> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS_TABLE1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33792> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33792> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:        [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<32x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[PARENT_WEIGHTS]] : !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 16 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits(%1 : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x32x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}
