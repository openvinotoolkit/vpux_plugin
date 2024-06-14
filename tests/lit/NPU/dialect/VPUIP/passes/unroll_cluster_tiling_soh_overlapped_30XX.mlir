//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-cluster-tiling --canonicalize  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// In this example DPU task in SOHoverlapped mode with input 1x16x32x32
// will be split into 2 clusters where each input will be 1x16x17x32
!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x30x30xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x30x30xf16, #NHWC, @DDR>
!Weights_DDR = memref<16x16x3x3xf16, #NHWC>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x30x30xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, #NHWC, @CMX_NN>

// Below dimensions are not correct but this is because related NCETask
// is just simulated using NNDMA
!Buffer0_CMX = memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
!Buffer1_CMX = memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>

//CHECK-LABEL: @UnrollNNDMA
func.func @UnrollNNDMA(%input: memref<1x16x32x32xf16>, %output: memref<1x16x30x30xf16>) -> memref<1x16x30x30xf16> {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x3x3xf16, #NHWC> =
        dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <DDR> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <DDR> <32768> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %input1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Buffer0_CMX
    %input2 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Buffer1_CMX

    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %output1 = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> !Buffer0_CMX
    %output2 = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> !Buffer1_CMX

    %weights = VPURT.DeclareBuffer <CMX_NN> <31808> -> !WeightsDistributed

    // Reorder input

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NHWC}
            inputs(%input: memref<1x16x32x32xf16>)
            outputs(%parent_in: !Input_DDR)
            -> !Input_DDR
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload input
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !InputDistributed) -> !InputDistributed
    }

    // Simulate 1st task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%input1: !Buffer0_CMX) outputs(%output1: !Buffer0_CMX) -> !Buffer0_CMX
    }

    // Simulate 2nd task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%input2: !Buffer1_CMX) outputs(%output2: !Buffer1_CMX) -> !Buffer1_CMX
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    // Reorder output

    VPURT.Task waits(%bar3: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NCHW}
            inputs(%parent_out: !Output_DDR)
            outputs(%output: memref<1x16x30x30xf16>)
            -> memref<1x16x30x30xf16>
    }

    return %output: memref<1x16x30x30xf16>

    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x3x3xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK-DAG:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <15360> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK-DAG:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <32768> -> memref<1x16x15x30xf16, #NHWC, @DDR>
    //CHECK-DAG:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <47168> -> memref<1x16x15x30xf16, #NHWC, @DDR>

    //CHECK:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31808> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>


    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of input
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Simulate tasks
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback 1st part of output
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x15x30xf16, #NHWC, @DDR>)
    //CHECK:        }

    // Copyback 2nd part of output
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x15x30xf16, #NHWC, @DDR>)
    //CHECK:        }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// In this example DPU task in SOHoverlapped mode with input 1x16x32x32
// will be split into 2 clusters where each input will be 1x16x17x32
!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x30x30xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ActivationWindowDistributed = !VPUIP.DistributedBuffer<
    1x1x1x16xui8, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32xf16, #NCHW, @DDR>
!Output_DDR = memref<1x16x30x30xf16, #NHWC, @DDR>
!Weights_DDR = memref<16x16x3x3xf16, #NHWC>
!WeightsTable_DDR = memref<16x1x1x4xsi32>
!ActivationWindow_DDR = memref<1x1x1x16xui8>

!InputStub_CMX = memref<1x16x32x32xf16, #NCHW, @CMX_NN>
!OutputStub_CMX = memref<1x16x30x30xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>
!ActivationWindowStub_CMX = memref<1x1x1x16xui8, @CMX_NN>

//CHECK-LABEL: @UnrollNCE
func.func @UnrollNCE(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x3x3xf16, #NHWC> =
        dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %activation_window_cst = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31808> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34112> -> !WeightsTableDistributed
    %activation_window = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34378> -> !ActivationWindowDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !InputDistributed) -> !InputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table_cst: !WeightsTable_DDR) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Upload activation window
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%activation_window_cst: !ActivationWindow_DDR) outputs(%activation_window: !ActivationWindowDistributed) -> !ActivationWindowDistributed
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    activation_window_channel_length = 16 : i64,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [3, 3],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CMCONV>
                }  input(%parent_input_cmx : !InputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    activation_window(%activation_window : !ActivationWindowDistributed)
                    parent_input(%parent_input_cmx : !InputDistributed)
                    parent_output(%parent_out_cmx : !OutputDistributed)
                    outputs(%parent_out_cmx : !OutputDistributed)
                        -> !OutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [29, 14, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 15, 0], outEnd = [29, 29, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:    [[ACTWIN_CST:%.*]] = const.Declare memref<1x1x1x16xui8>
    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x3x3xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x17x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, @DDR>
    //CHECK:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <960> -> memref<1x16x17x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, @DDR>
    //CHECK:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x15x30xf16, #NHWC, @DDR>
    //CHECK:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <14400> -> memref<1x16x15x30xf16, #NHWC, @DDR>

    //CHECK:    [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x17x32xf16, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x17x32xf16, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x30x30xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <31808> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <31808> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31808> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <34112> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <34112> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34112> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    [[ACTWIN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <34378> -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    //CHECK:    [[ACTWIN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <34378> -> memref<1x1x1x16xui8, [@CMX_NN, 1]>
    //CHECK:    [[ACTWIN_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34378> -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>



    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x17x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x17x32xf16, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload activation window
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[ACTWIN_CST]] : memref<1x1x1x16xui8>)
    //CHECK-SAME:       outputs([[ACTWIN_CMX_COPY]] : !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CMCONV>
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x17x32xf16, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           activation_window([[ACTWIN1_CMX]] : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x30x30xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [29, 14, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CMCONV>
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x17x32xf16, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           activation_window([[ACTWIN2_CMX]] : memref<1x1x1x16xui8, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x30x30xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [29, 29, 15], outStart = [0, 15, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x15x30xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x15x30xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:    return %arg1 : memref<1x16x30x30xf16, #NHWC, @DDR>
}
