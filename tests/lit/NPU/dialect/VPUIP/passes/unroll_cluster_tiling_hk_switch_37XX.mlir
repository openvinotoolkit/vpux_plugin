//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-cluster-tiling --canonicalize  %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "MULTICASTED|SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x33x32xf16, #NHWC>
!Output_DDR = memref<1x16x33x32xf16, #NHWC>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

func.func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <35328> -> !WeightsTableDistributed

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

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input_cmx : !InputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !InputDistributed)
                    parent_output(%parent_out_cmx : !OutputDistributed)
                    outputs(%parent_out_cmx : !OutputDistributed)
                        -> !OutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 16, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 17, 0], outEnd = [31, 32, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK-DAG:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <17408> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    //CHECK-DAG:    [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x33x32xf16, #NHWC>
    //CHECK-DAG:    [[PARENT_IN_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[IN1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[IN2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[PARENT_OUT_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x33x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <17408> -> !VPUIP.DistributedBuffer<1x16x17x32xf16, {order = #NHWC, strides = [16896, 1, 512, 16]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[OUT2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<1x16x16x32xf16, {order = #NHWC, strides = [16896, 1, 512, 16]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[WEIGHTS1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // 1st task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN1_CMX_1ST_TASK]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x17x32xf16, {order = #NHWC, strides = [16896, 1, 512, 16]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 16, 31], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 1st task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN2_CMX_1ST_TASK]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x16x32xf16, {order = #NHWC, strides = [16896, 1, 512, 16]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 32, 31], outStart = [0, 17, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:          VPUIP.NNDMA
    // CHECK-SAME:       inputs([[OUT_CMX]] : memref<1x16x33x32xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x16x33x32xf16, #NHWC>)
    // CHECK:        }

    //CHECK:    return %arg1 : memref<1x16x33x32xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x432x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x432x16x16xf16, #NCHW, @CMX_NN, {
    mode = "MULTICASTED|SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!Input_DDR = memref<1x432x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x432x16x16xf16, @DDR>

func.func @UnrollNCEWithNCHWOutput(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    %network_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %network_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    %cmx_in = VPURT.DeclareBuffer <CMX_NN> <110592> -> !InputDistributed
    %cmx_out = VPURT.DeclareBuffer <CMX_NN> <221184> -> !OutputDistributed

    %bar_dma_in = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar_nce = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar_dma_in : !VPURT.Barrier) {
      %10 = VPUIP.NNDMA inputs(%network_in : !Input_DDR) outputs(%cmx_in : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%bar_dma_in : !VPURT.Barrier) updates(%bar_nce : !VPURT.Barrier) {
      %10 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 4699 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%cmx_in : !InputDistributed) weights(%cmx_in : !InputDistributed) parent_input(%cmx_in : !InputDistributed) parent_output(%cmx_out : !OutputDistributed) outputs(%cmx_out : !OutputDistributed) -> !OutputDistributed  variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 7, 431], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 431], outStart = [0, 8, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
      }
    }
    VPURT.Task waits(%bar_nce : !VPURT.Barrier) {
      %10 = VPUIP.NNDMA {port = 0 : i64} inputs(%cmx_out : !OutputDistributed) outputs(%network_out : !Output_DDR) -> !Output_DDR
    }
    return %output : !Output_DDR

    // CHECK-DAG: [[IN_DDR_0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x432x8x16xf16, #NHWC, @DDR>
    // CHECK-DAG: [[IN_DDR_1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <110592> -> memref<1x432x8x16xf16, #NHWC, @DDR>
    // CHECK-DAG: [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x432x16x16xf16, @DDR>

    // CHECK-DAG: [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <110592> -> !VPUIP.DistributedBuffer<1x432x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-DAG: [[WEIGHTS_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG: [[WEIGHTS_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>
    // CHECK-DAG: [[NCE_INPUT_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG: [[NCE_INPUT_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>

    // CHECK-DAG: [[DDR_TO_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG: [[DDR_TO_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>

    // CHECK-DAG: [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> <221184> -> !VPUIP.DistributedBuffer<1x432x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK-DAG: [[OUT_CMX_TO_DDR:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <221184> -> memref<1x432x16x16xf16, [@CMX_NN, 0]>

    // CHECK-DAG: [[OUT_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <221184> -> !VPUIP.DistributedBuffer<1x432x8x16xf16, {order = #NCHW, strides = [110592, 256, 16, 1]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK-DAG: [[OUT_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <221440> -> !VPUIP.DistributedBuffer<1x432x8x16xf16, {order = #NCHW, strides = [110592, 256, 16, 1]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[BAR_DMA:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR_NCE:%.*]]  = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR_DMA]] : !VPURT.Barrier) {
    // CHECK:  VPUIP.NNDMA
    // CHECK-SAMEL inputs([[IN_DDR_0]] : memref<1x432x8x16xf16, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[DDR_TO_CMX_0]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK: VPURT.Task updates([[BAR_DMA]] : !VPURT.Barrier) {
    // CHECK:    VPUIP.NNDMA
    // CHECK-SAME: inputs([[IN_DDR_1]] : memref<1x432x8x16xf16, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[DDR_TO_CMX_1]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>

    // CHECK: VPURT.Task waits([[BAR_DMA]] : !VPURT.Barrier) updates([[BAR_NCE]] : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask
    // CHECK-SAME:  {activation_window_channel_length = 0 : i64, is_segmented, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:  input([[NCE_INPUT_CMX_0]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weights([[WEIGHTS_CMX_0]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x432x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x432x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:  outputs([[OUT_CMX_0]] : !VPUIP.DistributedBuffer<1x432x8x16xf16, {order = #NCHW, strides = [110592, 256, 16, 1]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x432x8x16xf16, {order = #NCHW, strides = [110592, 256, 16, 1]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> variants : {
    // CHECK:    DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 7, 431], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:    } PPE : {
    // CHECK:      PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}

    // CHECK:  VPURT.Task waits([[BAR_DMA]] : !VPURT.Barrier) updates([[BAR_NCE]] : !VPURT.Barrier) {
    // CHECK:  VPUIP.NCEClusterTask
    // CHECK-SAME:  {activation_window_channel_length = 0 : i64, is_segmented, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:  input([[NCE_INPUT_CMX_1]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK-SAME:  weights([[WEIGHTS_CMX_1]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK-SAME:  parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x432x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x432x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:  outputs([[OUT_CMX_1]] : !VPUIP.DistributedBuffer<1x432x8x16xf16, {order = #NCHW, strides = [110592, 256, 16, 1]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x432x8x16xf16, {order = #NCHW, strides = [110592, 256, 16, 1]}, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> variants : {
    // CHECK:    DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 431], outStart = [0, 8, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:    } PPE : {
    // CHECK:      PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
    // CHECK:    }

    // CHECK: VPURT.Task waits([[BAR_NCE]] : !VPURT.Barrier) {
    // CHECK:   VPUIP.NNDMA
    // CHECK:   inputs([[OUT_CMX_TO_DDR]] : memref<1x432x16x16xf16, [@CMX_NN, 0]>)
    // CHECK:   outputs([[OUT_DDR]] : memref<1x432x16x16xf16, @DDR>) -> memref<1x432x16x16xf16, @DDR>

    // CHECK: return %arg1 : memref<1x432x16x16xf16, @DDR>
}
