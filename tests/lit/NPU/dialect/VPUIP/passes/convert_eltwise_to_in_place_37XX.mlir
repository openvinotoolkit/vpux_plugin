//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-to-in-place --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InplaceEltwiseSameType
func.func @InplaceEltwiseSameType(%in: memref<1x32x96x96xf16, #NHWC>, %out: memref<1x32x96x96xf16, #NHWC>) -> memref<1x32x96x96xf16, #NHWC> {
    %cst0 = const.Declare memref<1x32x96x96xf16, #NHWC> = dense<2.0> : tensor<1x32x96x96xf16>, [#const.Reorder<#NHWC>]

    %buf_in = memref.alloc() : memref<1x32x96x96xf16, #NHWC, @CMX_NN>
    %buf0 = memref.alloc() : memref<1x32x96x96xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x32x96x96xf16, #NHWC, @CMX_NN>

    %0 = VPUIP.Copy inputs(%in : memref<1x32x96x96xf16, #NHWC>) outputs(%buf_in : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) -> memref<1x32x96x96xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.Copy inputs(%cst0 : memref<1x32x96x96xf16, #NHWC>) outputs(%buf0 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) -> memref<1x32x96x96xf16, #NHWC, @CMX_NN>

    %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = #VPUIP.nce_task_type<ELTWISE>,
                is_inplace = true
            }
            input(%0 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
            weights(%1 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
            parent_input(%0 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
            parent_output(%buf1 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
            outputs(%buf1 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) -> memref<1x32x96x96xf16, #NHWC, @CMX_NN>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask <ADD> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }

    %3 = VPUIP.Copy inputs(%2 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) outputs(%out : memref<1x32x96x96xf16, #NHWC>) -> memref<1x32x96x96xf16, #NHWC>

    return %3 : memref<1x32x96x96xf16, #NHWC>

    // CHECK:       [[BUF0:%.*]] = memref.alloc()
    // CHECK:       [[BUF1:%.*]] = memref.alloc()
    // CHECK-NOT:   [[BUF3:%.*]] = memref.alloc()

    // CHECK:       [[VAL0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
    // CHECK:       [[VAL1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[VAL2:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:      input([[VAL0]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[VAL1]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)


    // CHECK:       [[VAL3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[VAL2]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)

    //CHECK:        return [[VAL3]] : memref<1x32x96x96xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InplaceEltwiseFpClusterTiling
// CHECK-SAME:    ([[ARG0:%.*]]: memref<1x256x56x56xf16, #NHWC>,
// CHECK-SAME:    [[ARG1:%.*]]: memref<1x256x56x56xf16, #NHWC>)
// CHECK-SAME:    -> memref<1x256x56x56xf16, #NHWC> {
func.func @InplaceEltwiseFpClusterTiling(%in: memref<1x256x56x56xf16, #NHWC>, %out: memref<1x256x56x56xf16, #NHWC>) -> memref<1x256x56x56xf16, #NHWC> {

    %cst0 = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<2.0> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]

    %buf_0  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %buf_in = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %output_buf = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %0 = VPUIP.NCEClusterTiling
        inputs(%cst0 as %arg2: memref<1x256x56x56xf16, #NHWC>)
        outputs(%buf_0 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            %internal_0 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %1 = VPUIP.NCEClusterTiling
        inputs(%in as %arg2: memref<1x256x56x56xf16, #NHWC>)
        outputs(%buf_in as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            %internal_1 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %2 = VPUIP.NCEClusterTiling
    inputs(
        %0 as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>,
        %1 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    outputs(
        %output_buf as %arg4: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %internal_2 = VPUIP.NCEClusterTask
            {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 31170 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
                -> memref<1x256x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
            }
    }

    %3 = VPUIP.NCEClusterTiling
        inputs(%2 as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%out as %arg3: memref<1x256x56x56xf16, #NHWC>)
        -> memref<1x256x56x56xf16, #NHWC> {
            %internal_3 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC>) -> memref<1x256x56x56xf16, #NHWC>
    }

    return %3 : memref<1x256x56x56xf16, #NHWC>

    //CHECK-DAG:    [[CST:%.*]] = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<2.000000e+00> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[BUF_IN0:%.*]] = VPURT.AllocDistributed
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:        [[BUF_IN1:%.*]] = VPURT.AllocDistributed
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-NOT:    VPURT.AllocDistributed

    //CHECK:        [[COPY_IN0:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[CST]] as %arg2: memref<1x256x56x56xf16, #NHWC>)
    //CHECK-SAME:       outputs([[BUF_IN0]] as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[INNER:%.*]] = VPUIP.Copy

    //CHECK:        [[COPY_IN1:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[ARG0]] as %arg2: memref<1x256x56x56xf16, #NHWC>)
    //CHECK-SAME:       outputs([[BUF_IN1]] as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[INNER:%.*]] = VPUIP.Copy

    //CHECK:        [[ELTW_RES:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[COPY_IN0]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>, [[COPY_IN1]] as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       outputs([[BUF_IN0]] as %arg4: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[INNER:%.*]] = VPUIP.NCEClusterTask

    //CHECK:        [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:        inputs([[ELTW_RES]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:        outputs([[ARG1]] as %arg3: memref<1x256x56x56xf16, #NHWC>) -> memref<1x256x56x56xf16, #NHWC> {
    //CHECK:            [[INNER:%.*]] = VPUIP.Copy

    //CHECK:        return [[OUT_COPY]] : memref<1x256x56x56xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.011588541666666667:128>
!qElemType1 = !quant.uniform<u8:f16, 0.020557598039215686:128>
!qElemType2 = !quant.uniform<u8:f16, 0.0088848039215686271>

!qType0 = memref<1x256x56x56x!qElemType, #NHWC>
!qType1 = memref<1x256x56x56x!qElemType1, #NHWC>
!qType2 = memref<1x256x56x56x!qElemType2, #NHWC>

!qType0CMX = memref<1x256x56x56x!qElemType, #NHWC, @CMX_NN>
!qType1CMX = memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>
!qType2CMX = memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>

// CHECK-LABEL: @InplaceEltwiseQuantizedView
// CHECK-SAME:    ([[ARG0:%.*]]: memref<1x256x56x56x!qElemType, #NHWC>,
// CHECK-SAME:    [[ARG1:%.*]]: memref<1x256x56x56x!qElemType1, #NHWC>,
// CHECK-SAME:    [[ARG2:%.*]]: memref<1x256x56x56x!qElemType2, #NHWC>)
// CHECK-SAME:    -> memref<1x256x56x56x!qElemType2, #NHWC> {
func.func @InplaceEltwiseQuantizedView(%in: !qType0, %in2: !qType1, %out: !qType2) -> !qType2 {
    %buf_in = memref.alloc() : !qType0CMX
    %buf0 = memref.alloc() : !qType1CMX
    %buf1 = memref.alloc() : !qType2CMX

    %0 = VPUIP.Copy inputs(%in : !qType0) outputs(%buf_in : !qType0CMX) -> !qType0CMX
    %1 = VPUIP.Copy inputs(%in2 : !qType1) outputs(%buf0 : !qType1CMX) -> !qType1CMX

    %2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 11669 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%0 : !qType0CMX)
        weights(%1 : !qType1CMX)
        parent_input(%0 : !qType0CMX)
        parent_output(%buf1 : !qType2CMX)
        outputs(%buf1 : !qType2CMX)
        -> !qType2CMX variants : {
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
                PPETask <LRELUX> {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24302], in2_quant_mult = [43112], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [28813], quant_post_shift = 0 : i64, quant_shift = [29]}
        }

    %3 = VPUIP.Copy inputs(%2 : !qType2CMX) outputs(%out : !qType2) -> !qType2

    return %3 : !qType2

    // CHECK:       [[BUF0:%.*]] = memref.alloc() : memref<1x256x56x56x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[VIEW0:%.*]] = VPUIP.ViewOp
    // CHECK-SAME:       [[BUF0]] : memref<1x256x56x56x!qElemType, #NHWC, @CMX_NN>
    // CHECK-SAME:       to memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>
    // CHECK:       [[BUF2:%.*]] = memref.alloc() : memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>

    // CHECK:       [[INP1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x256x56x56x!qElemType, #NHWC>) outputs([[BUF0]] : memref<1x256x56x56x!qElemType, #NHWC, @CMX_NN>) -> memref<1x256x56x56x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[INP2:%.*]] = VPUIP.Copy inputs([[ARG1]] : memref<1x256x56x56x!qElemType1, #NHWC>) outputs([[BUF2]] : memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>) -> memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>
    // CHECK:       [[ELTWISE_OUT:%.*]] = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 11669 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:      input([[INP1]] : memref<1x256x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[INP2]] : memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input([[INP1]] : memref<1x256x56x56x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[VIEW0]] : memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VIEW0]] : memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>)
    // CHECK-SAME:      -> memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>

    // CHECK:       [[COPY_OUT:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ELTWISE_OUT]] : memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ARG2]] : memref<1x256x56x56x!qElemType2, #NHWC>)
    // CHECK-SAME:      -> memref<1x256x56x56x!qElemType2, #NHWC>

    // CHECK:    return [[COPY_OUT]] : memref<1x256x56x56x!qElemType2, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistType0 = !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!InputDistType1 = !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!EltwiseDistType = !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

func.func @InplaceEltwiseNeedsCast(%input : memref<1x512x28x28xf16, #NHWC>, %out: memref<1x512x28x28xf16, #NHWC>) -> memref<1x512x28x28xf16, #NHWC> {

    %inputBuf0 = VPURT.AllocDistributed -> !InputDistType0
    %inputBuf1 = VPURT.AllocDistributed -> !InputDistType1

    %copyInput0 = VPUIP.NCEClusterTiling
            inputs(%input as %arg2: memref<1x512x28x28xf16, #NHWC>)
            outputs(%inputBuf0 as %arg3: memref<1x512x28x28xf16, #NHWC, @CMX_NN>) -> !InputDistType0 {
                %internalVar = VPUIP.Copy inputs(%arg2 : memref<1x512x28x28xf16, #NHWC>) outputs(%arg3 : memref<1x512x28x28xf16, #NHWC, @CMX_NN>) -> memref<1x512x28x28xf16, #NHWC, @CMX_NN>
        }

    %copyInput1 = VPUIP.NCEClusterTiling
            inputs(%input as %arg2: memref<1x512x28x28xf16, #NHWC>)
            outputs(%inputBuf1 as %arg3: memref<1x512x28x28xf16, #NHWC, @CMX_NN>) -> !InputDistType1 {
                %internalVar = VPUIP.Copy inputs(%arg2 : memref<1x512x28x28xf16, #NHWC>) outputs(%arg3 : memref<1x512x28x28xf16, #NHWC, @CMX_NN>) -> memref<1x512x28x28xf16, #NHWC, @CMX_NN>
        }

    %eltwiseIn0 = VPUIP.DistributedCast inputs(%copyInput0 : !InputDistType0) -> !EltwiseDistType
    %eltwiseIn1 = VPUIP.DistributedCast inputs(%copyInput1 : !InputDistType1) -> !EltwiseDistType
    // This buffer will be eliminated, input 0 will be used insted
    %eltwiseOutBuf = VPURT.AllocDistributed -> !EltwiseDistType

    %eltwise_output = VPUIP.NCEClusterTiling
            inputs(%eltwiseIn0 as %arg2: memref<1x512x28x28xf16, #NHWC, @CMX_NN>,
                   %eltwiseIn1 as %arg3: memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
            outputs(%eltwiseOutBuf as %arg4: memref<1x512x28x28xf16, #NHWC, @CMX_NN>) -> !EltwiseDistType {
            %internalVar = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 32317 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg2 : memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x512x28x28xf16, #NHWC, @CMX_NN>) -> memref<1x512x28x28xf16, #NHWC, @CMX_NN> variants : {
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [27, 27, 511], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [27, 27, 511], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
                PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
            }
        }

    %copyOut = VPUIP.NCEClusterTiling
        inputs(%eltwise_output as %arg2: memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
        outputs(%out as %arg3: memref<1x512x28x28xf16, #NHWC>) -> memref<1x512x28x28xf16, #NHWC> {
            %internalVar = VPUIP.Copy inputs(%arg2 : memref<1x512x28x28xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x512x28x28xf16, #NHWC>) -> memref<1x512x28x28xf16, #NHWC>
        }

    return %copyOut : memref<1x512x28x28xf16, #NHWC>

    // output of Eltwise has been redirected to this buffer which is the first input
    // CHECK:       [[BUF0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // since the first input has different distribution mode it need distribution cast operation
    // CHECK:       [[ELTW_OUT_BUF:%.*]] = VPUIP.DistributedCast
    // CHECK-SAME:       inputs([[BUF0]] : !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[BUF1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[COPY0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg2: memref<1x512x28x28xf16, #NHWC>)
    // CHECK-SAME:      outputs([[BUF0]] as %arg3: memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INNER:%.*]] = VPUIP.Copy

    // CHECK:       [[COPY1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg2: memref<1x512x28x28xf16, #NHWC>)
    // CHECK-SAME:      outputs([[BUF1]] as %arg3: memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INNER:%.*]] = VPUIP.Copy

    // CHECK:       [[ELTW_IN0:%.*]] = VPUIP.DistributedCast
    // CHECK-SAME:      inputs([[COPY0]] : !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[ELTW_IN1:%.*]] = VPUIP.DistributedCast
    // CHECK-SAME:      inputs([[COPY1]] : !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[ELTW_OUT:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[ELTW_IN0]] as %arg2: memref<1x512x28x28xf16, #NHWC, @CMX_NN>, [[ELTW_IN1]] as %arg3: memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ELTW_OUT_BUF]] as %arg4: memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[INNER:%.*]] = VPUIP.NCEClusterTask

    // CHECK:       [[COPY_OUT:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[ELTW_OUT]] as %arg2: memref<1x512x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg1 as %arg3: memref<1x512x28x28xf16, #NHWC>)
    // CHECK-SAME:      -> memref<1x512x28x28xf16, #NHWC> {
    //CHECK:            [[INNER:%.*]] = VPUIP.Copy

    // CHECK:       return [[COPY_OUT]] : memref<1x512x28x28xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedType = !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK:    func @InplaceEltwiseFirstInputHas2Consumers(%arg0: memref<1x256x56x56xf16, #NHWC>)
func.func @InplaceEltwiseFirstInputHas2Consumers(%in: memref<1x256x56x56xf16, #NHWC>) -> (!DistributedType, !DistributedType) {

    %cst0 = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<2.0> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<1.0> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    %buf_0  = VPURT.AllocDistributed -> !DistributedType
    %buf_1  = VPURT.AllocDistributed -> !DistributedType
    %buf_in = VPURT.AllocDistributed -> !DistributedType
    %buf_in_1 = VPURT.AllocDistributed -> !DistributedType
    %output_buf = VPURT.AllocDistributed -> !DistributedType
    %output_buf_1 = VPURT.AllocDistributed -> !DistributedType

    %0 = VPUIP.NCEClusterTiling
        inputs(%cst0 as %arg2: memref<1x256x56x56xf16, #NHWC>)
        outputs(%buf_0 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !DistributedType {
            %internal_0 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %1 = VPUIP.NCEClusterTiling
        inputs(%in as %arg2: memref<1x256x56x56xf16, #NHWC>)
        outputs(%buf_in as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !DistributedType {
            %internal_1 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %2 = VPUIP.NCEClusterTiling
    inputs(
        %0 as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>,
        %1 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    outputs(
        %output_buf as %arg4: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    -> !DistributedType {
        %internal_2 = VPUIP.NCEClusterTask
            {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 31170 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
                -> memref<1x256x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
            }
    }

    %4 = VPUIP.NCEClusterTiling
        inputs(%cst1 as %arg2: memref<1x256x56x56xf16, #NHWC>)
        outputs(%buf_1 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !DistributedType {
            %internal_4 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %5 = VPUIP.NCEClusterTiling
        inputs(%in as %arg2: memref<1x256x56x56xf16, #NHWC>)
        outputs(%buf_in_1 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !DistributedType {
            %internal_5 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %6 = VPUIP.NCEClusterTiling
    inputs(
        %5 as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>,
        %4 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    outputs(
        %output_buf_1 as %arg4: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    -> !DistributedType {
        %internal_6 = VPUIP.NCEClusterTask
            {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 31170 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
                -> memref<1x256x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
            }
    }

    return %2, %6 : !DistributedType, !DistributedType

    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<1.000000e+00> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<2.000000e+00> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[BUF_0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_IN0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_IN1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_BUF:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-NOT:       VPURT.AllocDistributed

    // CHECK:       [[COPY_IN0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[CST_0]] as %arg1: memref<1x256x56x56xf16, #NHWC>)
    // CHECK-SAME:       outputs([[BUF_0]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[COPY_IN1:%.*]]  = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs(%arg0 as %arg1: memref<1x256x56x56xf16, #NHWC>)
    // CHECK-SAME:        outputs([[BUF_IN0]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[ELTW_RES:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[COPY_IN0]] as %arg1: memref<1x256x56x56xf16, #NHWC, @CMX_NN>, [[COPY_IN1]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[OUTPUT_BUF]] as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.NCEClusterTask

    // CHECK:        [[COPY_IN2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[CST]] as %arg1: memref<1x256x56x56xf16, #NHWC>)
    // CHECK-SAME:        outputs([[BUF_1]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[COPY_IN3:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs(%arg0 as %arg1: memref<1x256x56x56xf16, #NHWC>)
    // CHECK-SAME:        outputs([[BUF_IN1]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[ELTW_RES1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[COPY_IN3]] as %arg1: memref<1x256x56x56xf16, #NHWC, @CMX_NN>, [[COPY_IN2]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[BUF_1]] as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.NCEClusterTask

    // CHECK:        return [[ELTW_RES]], [[ELTW_RES1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!PermuteInDistributedType = !VPUIP.DistributedBuffer<1x56x256x56xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64
}>

!PermuteOutDistributedType = !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64
}>

!EltwiseDistributedType = !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK:    func @InplaceEltwiseFirstInput2ConsumersSecondInputFromNCEPermute
// CHECK-SAME: ([[ARG0:%.*]]: memref<1x256x56x56xf16, #NHWC>)
func.func @InplaceEltwiseFirstInput2ConsumersSecondInputFromNCEPermute(%in: memref<1x256x56x56xf16, #NHWC>) -> (!EltwiseDistributedType, !EltwiseDistributedType) {

    %cst0 = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<2.0> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare memref<1x56x256x56xf16, #NHWC> = dense<1.0> : tensor<1x56x256x56xf16>, [#const.Reorder<#NHWC>]
    %buf_0  = VPURT.AllocDistributed -> !EltwiseDistributedType
    %buf_1  = VPURT.AllocDistributed -> !PermuteInDistributedType
    %permute_out_buf  = VPURT.AllocDistributed -> !PermuteOutDistributedType
    %buf_in = VPURT.AllocDistributed -> !EltwiseDistributedType
    %buf_in_1 = VPURT.AllocDistributed -> !EltwiseDistributedType
    %output_buf = VPURT.AllocDistributed -> !EltwiseDistributedType
    %elt_in_place_out_buf = VPURT.AllocDistributed -> !EltwiseDistributedType

    %0 = VPUIP.NCEClusterTiling
        inputs(%cst0 as %arg2: memref<1x256x56x56xf16, #NHWC>)
        outputs(%buf_0 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !EltwiseDistributedType {
            %internal_0 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %1 = VPUIP.NCEClusterTiling
        inputs(%in as %arg2: memref<1x256x56x56xf16, #NHWC>)
        outputs(%buf_in as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !EltwiseDistributedType {
            %internal_1 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %2 = VPUIP.NCEClusterTiling
    inputs(
        %0 as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>,
        %1 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    outputs(
        %output_buf as %arg4: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    -> !EltwiseDistributedType {
        %internal_2 = VPUIP.NCEClusterTask
            {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 31170 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
                -> memref<1x256x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
            }
    }

    %4 = VPUIP.NCEClusterTiling
        inputs(%cst1 as %arg2: memref<1x56x256x56xf16, #NHWC>)
        outputs(%buf_1 as %arg3: memref<1x56x256x56xf16, #NHWC, @CMX_NN>)
        -> !PermuteInDistributedType {
            %internal_4 = VPUIP.Copy inputs(%arg2 : memref<1x56x256x56xf16, #NHWC>) outputs(%arg3 : memref<1x56x256x56xf16, #NHWC, @CMX_NN>) -> memref<1x56x256x56xf16, #NHWC, @CMX_NN>
    }

    %5 = VPUIP.NCEClusterTiling
        inputs(%4 as %arg2: memref<1x56x256x56xf16, #NHWC, @CMX_NN>)
        outputs(%permute_out_buf as %arg3: memref<1x56x256x56xf16, #NWCH, @CMX_NN>)
        -> !PermuteOutDistributedType {
            %internal_5 = VPUIP.NCEClusterTask
                {activation_window_channel_length = 0 : i64, is_permute_quantize, minimumHardwareExecutionCost = 17144 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
                input(%arg2 : memref<1x56x256x56xf16, #NHWC, @CMX_NN>)
                weights(%arg2 : memref<1x56x256x56xf16, #NHWC, @CMX_NN>)
                parent_input(%arg2 : memref<1x56x256x56xf16, #NHWC, @CMX_NN>)
                parent_output(%arg3 : memref<1x56x256x56xf16, #NWCH, @CMX_NN>)
                outputs(%arg3 : memref<1x56x256x56xf16, #NWCH, @CMX_NN>)
                    -> memref<1x56x256x56xf16, #NWCH, @CMX_NN> variants : {
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [27, 255, 55], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [27, 255, 55], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
              PPETask <ADD> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01]}
            }
    }

    %6 = VPUIP.NCEClusterTiling
        inputs(%in as %arg2: memref<1x256x56x56xf16, #NHWC>)
        outputs(%buf_in_1 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !EltwiseDistributedType {
            %internal_6 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %7 = VPUIP.ViewOp %5 : !PermuteOutDistributedType to !EltwiseDistributedType

    %8 = VPUIP.NCEClusterTiling
    inputs(
        %6 as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>,
        %7 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    outputs(
        %elt_in_place_out_buf as %arg4: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    -> !EltwiseDistributedType {
        %internal_7 = VPUIP.NCEClusterTask
            {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 31170 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
                -> memref<1x256x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
            }
    }

    return %2, %8 : !EltwiseDistributedType, !EltwiseDistributedType

    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<2.000000e+00> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare memref<1x56x256x56xf16, #NHWC> = dense<1.000000e+00> : tensor<1x56x256x56xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[BUF_0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[PERM_QUANT_IN:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x56x256x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK:       [[PERM_QUANT_OUT:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>

    // CHECK: [[VIEW0:%.*]] = VPUIP.ViewOp [[PERM_QUANT_OUT]] :
    // CHECK-SAME: !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME: to !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[BUF_IN1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_IN2:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_BUF:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-NOT:       VPURT.AllocDistributed

    // CHECK:       [[COPY_IN0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[CST]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC>)
    // CHECK-SAME:       outputs([[BUF_0]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[COPY_IN1:%.*]]  = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[ARG0]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC>)
    // CHECK-SAME:        outputs([[BUF_IN1]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[ELTW_RES:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[COPY_IN0]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC, @CMX_NN>, [[COPY_IN1]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[OUTPUT_BUF]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.NCEClusterTask

    // CHECK:        [[COPY_IN_PERM:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[CST_0]] as {{[^:]+}}: memref<1x56x256x56xf16, #NHWC>)
    // CHECK-SAME:        outputs([[PERM_QUANT_IN]] as {{[^:]+}}: memref<1x56x256x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x56x256x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[PERM_RES:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[COPY_IN_PERM]] as {{[^:]+}}: memref<1x56x256x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[PERM_QUANT_OUT]] as {{[^:]+}}: memref<1x56x256x56xf16, #NWCH, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:              is_permute_quantize

    // CHECK:        [[COPY_IN3:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[ARG0]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC>)
    // CHECK-SAME:        outputs([[BUF_IN2]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:       [[VIEW1:%.*]] = VPUIP.ViewOp [[PERM_RES]] :
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x56x256x56xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:        [[ELTW_RES_IN_PLACE:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[COPY_IN3]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC, @CMX_NN>, [[VIEW1]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[VIEW0]] as {{[^:]+}}: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          is_inplace = true

    // CHECK:        return [[ELTW_RES]], [[ELTW_RES_IN_PLACE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x128x52x104xf16, #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x128x104x208xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint: @VPU.SW::@runtime stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK:    func @InplaceEltwiseSubViewInterp(%arg0: memref<1x128x104x104xf16, #NHWC>, %arg1: memref<1x128x104x104xf16, #NHWC>)
func.func @InplaceEltwiseSubViewInterp(%in1: memref<1x128x104x104xf16, #NHWC>, %in2: memref<1x128x104x104xf16, #NHWC>) -> (!DistributedType2, !DistributedType2, !DistributedType1) {
    %buf_0  = VPURT.AllocDistributed -> !DistributedType1
    %buf_2  = VPURT.AllocDistributed -> !DistributedType1
    %buf_in = VPURT.AllocDistributed -> !DistributedType1
    %buf_in_1 = VPURT.AllocDistributed -> !DistributedType1
    %buf_in_2 = VPURT.AllocDistributed -> !DistributedType1
    %output_buf = VPURT.AllocDistributed -> !DistributedType2
    %output_buf_1 = VPURT.AllocDistributed -> !DistributedType2
    %output_buf_2 = VPURT.AllocDistributed -> !DistributedType1

    %0 = VPUIP.SubView %in1 [0, 0, 0, 0] [1, 128, 52, 104]
        : memref<1x128x104x104xf16, #NHWC> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>

    %1 = VPUIP.SubView %in1 [0, 0, 0, 0] [1, 128, 52, 104]
        : memref<1x128x104x104xf16, #NHWC> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>


    %3 = VPUIP.NCEClusterTiling
        inputs(%0 as %arg2: memref<1x128x52x104xf16, #NHWC>)
        outputs(%buf_in as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        -> !DistributedType1 {
            %internal_1 = VPUIP.Copy inputs(%arg2 : memref<1x128x52x104xf16, #NHWC>) outputs(%arg3 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>) -> memref<1x128x52x104xf16, #NHWC, @CMX_NN>
    }

    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg4: memref<1x128x52x104xf16, #NHWC, @CMX_NN>) outputs(%output_buf as %arg5: memref<1x128x104x208xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg4 as %arg6: memref<1x128x52x104xf16, #NHWC, @CMX_NN>) outputs(%arg5 as %arg7: memref<1x128x104x208xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x128x104x208xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [128, 52, 104, 1], [128, 104, 208, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg6, %arg7) : memref<1x128x52x104xf16, #NHWC, @CMX_NN>, memref<1x128x104x208xf16, #NHWC, @CMX_NN>
        }
    }

    %6 = VPUIP.SubView %in2 [0, 0, 0, 0] [1, 128, 52, 104]
        : memref<1x128x104x104xf16, #NHWC> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>

    %7 = VPUIP.SubView %in2 [0, 0, 52, 0] [1, 128, 52, 104]
        : memref<1x128x104x104xf16, #NHWC> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>

    %9 = VPUIP.NCEClusterTiling
        inputs(%7 as %arg2: memref<1x128x52x104xf16, #NHWC>)
        outputs(%buf_in_1 as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        -> !DistributedType1 {
            %internal_5 = VPUIP.Copy inputs(%arg2 : memref<1x128x52x104xf16, #NHWC>) outputs(%arg3 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>) -> memref<1x128x52x104xf16, #NHWC, @CMX_NN>
    }

    %10 = VPUIP.NCEClusterTiling inputs(%9 as %arg4: memref<1x128x52x104xf16, #NHWC, @CMX_NN>) outputs(%output_buf_1 as %arg5: memref<1x128x104x208xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg4 as %arg6: memref<1x128x52x104xf16, #NHWC, @CMX_NN>) outputs(%arg5 as %arg7: memref<1x128x104x208xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x128x104x208xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [128, 52, 104, 1], [128, 104, 208, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg6, %arg7) : memref<1x128x52x104xf16, #NHWC, @CMX_NN>, memref<1x128x104x208xf16, #NHWC, @CMX_NN>
        }
    }

    %12 = VPUIP.NCEClusterTiling
        inputs(%1 as %arg2: memref<1x128x52x104xf16, #NHWC>)
        outputs(%buf_2 as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        -> !DistributedType1 {
            %internal_8 = VPUIP.Copy inputs(%arg2 : memref<1x128x52x104xf16, #NHWC>) outputs(%arg3 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>) -> memref<1x128x52x104xf16, #NHWC, @CMX_NN>
    }

    %13 = VPUIP.NCEClusterTiling
        inputs(%6 as %arg2: memref<1x128x52x104xf16, #NHWC>)
        outputs(%buf_in_2 as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        -> !DistributedType1 {
            %internal_9 = VPUIP.Copy inputs(%arg2 : memref<1x128x52x104xf16, #NHWC>) outputs(%arg3 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>) -> memref<1x128x52x104xf16, #NHWC, @CMX_NN>
    }

    %14 = VPUIP.NCEClusterTiling
    inputs(
        %12 as %arg2: memref<1x128x52x104xf16, #NHWC, @CMX_NN>,
        %13 as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    outputs(
        %output_buf_2 as %arg4: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %internal_10 = VPUIP.NCEClusterTask
            {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 31170 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg2 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
                -> memref<1x128x52x104xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 51, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 103, 127], outStart = [0, 52, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
            }
    }
    return %4, %10, %14 : !DistributedType2, !DistributedType2, !DistributedType1

    // CHECK:         [[BUF_2:%.*]]  = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN0:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN2:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[OUTPUT_BUF:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:         [[OUTPUT_BUF1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-NOT:         VPURT.AllocDistributed

    // CHECK:        [[SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 128, 52, 104]
    // CHECK-SAME:        : memref<1x128x104x104xf16, #NHWC> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>

    // CHECK:        [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 128, 52, 104]
    // CHECK-SAME:        : memref<1x128x104x104xf16, #NHWC> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>

    // CHECK:        [[COPY_IN1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[SUBVIEW0]] as %arg2: memref<1x128x52x104xf16, #NHWC>)
    // CHECK-SAME:        outputs([[BUF_IN0]] as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[INTERP_0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[COPY_IN1]] as %arg2: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[OUTPUT_BUF]] as %arg3: memref<1x128x104x208xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.SW.Kernel

    // CHECK:        [[SUBVIEW2:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 128, 52, 104]
    // CHECK-SAME:        : memref<1x128x104x104xf16, #NHWC> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>
    // CHECK:        [[SUBVIEW3:%.*]] = VPUIP.SubView %arg1 [0, 0, 52, 0] [1, 128, 52, 104]
    // CHECK-SAME:        : memref<1x128x104x104xf16, #NHWC> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}>

    // CHECK:        [[COPY_IN3:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[SUBVIEW3]] as %arg2: memref<1x128x52x104xf16, #NHWC>)
    // CHECK-SAME:        outputs([[BUF_IN1]] as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[INTERP_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[COPY_IN3]] as %arg2: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[OUTPUT_BUF1]] as %arg3: memref<1x128x104x208xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x104x208xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.SW.Kernel

    // CHECK:        [[COPY_IN4:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[SUBVIEW1]] as %arg2: memref<1x128x52x104xf16, #NHWC>)
    // CHECK-SAME:        outputs([[BUF_2]] as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[COPY_IN5:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[SUBVIEW2]] as %arg2: memref<1x128x52x104xf16, #NHWC>)
    // CHECK-SAME:        outputs([[BUF_IN2]] as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.Copy

    // CHECK:        [[ELTW_RES2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[COPY_IN4]] as %arg2: memref<1x128x52x104xf16, #NHWC, @CMX_NN>, [[COPY_IN5]] as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[BUF_IN2]] as %arg4: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:             [[INNER:%.*]] = VPUIP.NCEClusterTask

    // CHECK:        return [[INTERP_0]], [[INTERP_1]], [[ELTW_RES2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType1 = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType2 = !quant.uniform<u8:f16, 2.0000000000000000E-1>


!DistributedType1 = !VPUIP.DistributedBuffer<
    1x64x26x52x!qElemType2,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x64x13x26x!qElemType2,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

// CHECK:    func @InplaceEltwisePermQuantSubView(%arg0: memref<1x64x52x52x!qElemType>, %arg1: memref<1x64x52x52x!qElemType>)
func.func @InplaceEltwisePermQuantSubView(%in1: memref<1x64x52x52x!qElemType1>, %in2: memref<1x64x52x52x!qElemType1>) -> (!DistributedType2, !DistributedType2, !DistributedType1) {
    %wt = const.Declare memref<64x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<64x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>
    %buf  = VPURT.AllocDistributed -> !DistributedType1
    %buf_in = VPURT.AllocDistributed -> !DistributedType1
    %buf_in_1 = VPURT.AllocDistributed -> !DistributedType1
    %buf_in_2 = VPURT.AllocDistributed -> !DistributedType1
    %output_buf = VPURT.AllocDistributed -> !DistributedType2
    %output_buf_1 = VPURT.AllocDistributed -> !DistributedType2
    %output_buf_2 = VPURT.AllocDistributed -> !DistributedType1

    %0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
        inputs(%in1 : memref<1x64x52x52x!qElemType1>)
        -> memref<1x64x52x52x!qElemType1, #NHWC>

    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x64x52x52x!qElemType1, #NHWC>) -> memref<1x64x52x52x!qElemType2, #NHWC>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 26, 52] :
        memref<1x64x52x52x!qElemType2, #NHWC> to memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 26, 52] :
        memref<1x64x52x52x!qElemType2, #NHWC> to memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    %4 = VPUIP.NCEClusterTiling
        inputs(%2 as %arg2: memref<1x64x26x52x!qElemType2, #NHWC>)
        outputs(%buf_in as %arg3: memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
        -> !DistributedType1 {
            %inner = VPUIP.Copy inputs(%arg2 : memref<1x64x26x52x!qElemType2, #NHWC>) outputs(%arg3 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>) -> memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>
    }

    %6 = VPUIP.NCEClusterTiling
        inputs(
            %4 as %arg2: memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>,
            %wt as %arg3: memref<64x1x1x4xsi32, @CMX_NN>,
            %act_win as %arg4: memref<1x1x1x16xui8, @CMX_NN>)
        outputs(
            %output_buf as %arg5: memref<1x64x13x26x!qElemType2, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType2, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            %inner= VPUIP.NCEClusterTask {activation_window_channel_length = 64 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], minimumHardwareExecutionCost = 10325 : i64, task_type = #VPUIP.nce_task_type<MAXPOOL>}
                input(%arg2 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
                weight_table(%arg3 : memref<64x1x1x4xsi32, @CMX_NN>)
                activation_window(%arg4 : memref<1x1x1x16xui8, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x26x52x!qElemType2, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                parent_output(%arg5 : memref<1x64x13x26x!qElemType2, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                outputs(%arg5 : memref<1x64x13x26x!qElemType2, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) -> memref<1x64x13x26x!qElemType2, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>
            variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 12, 25], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 12, 25], outStart = [0, 13, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            PPETask <NOOP> {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
    }

    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
        inputs(%in2 : memref<1x64x52x52x!qElemType1>) -> memref<1x64x52x52x!qElemType1, #NHWC>

    %8 = VPUIP.QuantizeCast inputs(%7 : memref<1x64x52x52x!qElemType1, #NHWC>) -> memref<1x64x52x52x!qElemType2, #NHWC>

    %10 = VPUIP.SubView %8 [0, 0, 0, 0] [1, 64, 26, 52] :
        memref<1x64x52x52x!qElemType2, #NHWC> to memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    %11 = VPUIP.SubView %8 [0, 0, 26, 0] [1, 64, 26, 52] :
        memref<1x64x52x52x!qElemType2, #NHWC> to memref<1x64x26x52x!qElemType2, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    %12 = VPUIP.NCEClusterTiling
        inputs(%10 as %arg2: memref<1x64x26x52x!qElemType2, #NHWC>)
        outputs(%buf_in_1 as %arg3: memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
        -> !DistributedType1 {
            %inner = VPUIP.Copy inputs(%arg2 : memref<1x64x26x52x!qElemType2, #NHWC>) outputs(%arg3 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>) -> memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>
    }

    %13 = VPUIP.NCEClusterTiling
        inputs(
            %12 as %arg2: memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>,
            %wt as %arg3: memref<64x1x1x4xsi32, @CMX_NN>,
            %act_win as %arg4: memref<1x1x1x16xui8, @CMX_NN>)
        outputs(
            %output_buf_1 as %arg5: memref<1x64x13x26x!qElemType2, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType2, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            %inner= VPUIP.NCEClusterTask {activation_window_channel_length = 64 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], minimumHardwareExecutionCost = 10325 : i64, task_type = #VPUIP.nce_task_type<MAXPOOL>}
                input(%arg2 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
                weight_table(%arg3 : memref<64x1x1x4xsi32, @CMX_NN>)
                activation_window(%arg4 : memref<1x1x1x16xui8, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x26x52x!qElemType2, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                parent_output(%arg5 : memref<1x64x13x26x!qElemType2, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x64x13x26x!qElemType2, #NHWC, @CMX_NN>) -> memref<1x64x13x26x!qElemType2, #NHWC, @CMX_NN>
            variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 12, 25], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 12, 25], outStart = [0, 13, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            PPETask <NOOP> {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
    }

    %14 = VPUIP.NCEClusterTiling
        inputs(%3 as %arg2: memref<1x64x26x52x!qElemType2, #NHWC>)
        outputs(%buf as %arg3: memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
        -> !DistributedType1 {
            %inner = VPUIP.Copy inputs(%arg2 : memref<1x64x26x52x!qElemType2, #NHWC>) outputs(%arg3 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>) -> memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>
    }

    %15 = VPUIP.NCEClusterTiling
        inputs(%11 as %arg2: memref<1x64x26x52x!qElemType2, #NHWC>)
        outputs(%buf_in_2 as %arg3: memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
        -> !DistributedType1 {
            %inner = VPUIP.Copy inputs(%arg2 : memref<1x64x26x52x!qElemType2, #NHWC>) outputs(%arg3 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>) -> memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>
    }

    %16 = VPUIP.NCEClusterTiling
    inputs(
        %14 as %arg2: memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>,
        %15 as %arg3: memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
    outputs(
        %output_buf_2 as %arg4: memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
    -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %inner = VPUIP.NCEClusterTask
            {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 31170 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg2 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN>)
                -> memref<1x64x26x52x!qElemType2, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 51, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 103, 127], outStart = [0, 52, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
            }
    }

    return %6, %13, %16 : !DistributedType2, !DistributedType2, !DistributedType1

    // CHECK-DAG:         [[ACT_WIN:%.*]] = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>
    // CHECK-DAG:         [[WT:%.*]] = const.Declare memref<64x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:         [[BUF:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_IN_2:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_OUT:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         [[BUF_OUT_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-NOT:       VPURT.AllocDistributed

    // CHECK:         [[PERM_CAST:%.*]] = VPUIP.PermuteCast
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:       inputs(%arg0 : memref<1x64x52x52x!qElemType>)
    // CHECK-SAME:       -> memref<1x64x52x52x!qElemType, #NHWC>

    // CHECK:         [[QUANT_CAST:%.*]] = VPUIP.QuantizeCast
    // CHECK-SAME:       inputs([[PERM_CAST]] : memref<1x64x52x52x!qElemType, #NHWC>)
    // CHECK-SAME:       -> memref<1x64x52x52x!qElemType1, #NHWC>

    // CHECK:         [[SUB_VIEW:%.*]] = VPUIP.SubView [[QUANT_CAST]]
    // CHECK-SAME:       [0, 0, 0, 0] [1, 64, 26, 52]
    // CHECK-SAME:       : memref<1x64x52x52x!qElemType1, #NHWC>
    // CHECK-SAME:       to memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    // CHECK:         [[SUB_VIEW_1:%.*]] = VPUIP.SubView [[QUANT_CAST]]
    // CHECK-SAME:       [0, 0, 0, 0] [1, 64, 26, 52]
    // CHECK-SAME:       : memref<1x64x52x52x!qElemType1, #NHWC>
    // CHECK-SAME:       to memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    // CHECK:         [[COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[SUB_VIEW]] as %arg2: memref<1x64x26x52x!qElemType1, #NHWC>)
    // CHECK-SAME:       outputs([[BUF_IN]] as %arg3: memref<1x64x26x52x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            [[INNER:%.*]] = VPUIP.Copy

    // CHECK:         [[MAXPOOL:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[COPY]] as %arg2: memref<1x64x26x52x!qElemType1, #NHWC, @CMX_NN>, [[WT]] as %arg3: memref<64x1x1x4xsi32, @CMX_NN>, [[ACT_WIN]] as %arg4: memref<1x1x1x16xui8, @CMX_NN>)
    // CHECK-SAME:       outputs([[BUF_OUT:%.*]] as %arg5: memref<1x64x13x26x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            [[INNER:%.*]] = VPUIP.NCEClusterTask

    // CHECK:         [[PERM_CAST_1:%.*]] = VPUIP.PermuteCast
    // CHECK-SAME:       {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:       inputs(%arg1 : memref<1x64x52x52x!qElemType>)
    // CHECK-SAME:       -> memref<1x64x52x52x!qElemType, #NHWC>

    // CHECK:         [[QUANT_CAST_1:%.*]] = VPUIP.QuantizeCast
    // CHECK-SAME:       inputs([[PERM_CAST_1]] : memref<1x64x52x52x!qElemType, #NHWC>)
    // CHECK-SAME:       -> memref<1x64x52x52x!qElemType1, #NHWC>

    // CHECK:         [[SUB_VIEW_2:%.*]] = VPUIP.SubView [[QUANT_CAST_1]]
    // CHECK-SAME:       [0, 0, 0, 0] [1, 64, 26, 52]
    // CHECK-SAME:       : memref<1x64x52x52x!qElemType1, #NHWC>
    // CHECK-SAME:       to memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    // CHECK:         [[SUB_VIEW_3:%.*]] = VPUIP.SubView [[QUANT_CAST_1]]
    // CHECK-SAME:       [0, 0, 26, 0] [1, 64, 26, 52]
    // CHECK-SAME:       : memref<1x64x52x52x!qElemType1, #NHWC>
    // CHECK-SAME:       to memref<1x64x26x52x!qElemType1, {order = #NHWC, strides = [173056, 1, 3328, 64]}>

    // CHECK:         [[COPY_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[SUB_VIEW_2]] as %arg2: memref<1x64x26x52x!qElemType1, #NHWC>)
    // CHECK-SAME:       outputs([[BUF_IN_1]] as %arg3: memref<1x64x26x52x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            [[INNER:%.*]] = VPUIP.Copy

    // CHECK:         [[MAXPOOL_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[COPY_1:%.*]] as %arg2: memref<1x64x26x52x!qElemType1, #NHWC, @CMX_NN>, [[WT]] as %arg3: memref<64x1x1x4xsi32, @CMX_NN>, [[ACT_WIN]] as %arg4: memref<1x1x1x16xui8, @CMX_NN>)
    // CHECK-SAME:       outputs([[BUF_OUT_1:%.*]] as %arg5: memref<1x64x13x26x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x13x26x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    /// CHECK:           [[INNER:%.*]] = VPUIP.NCEClusterTask

    // CHECK:         [[COPY_2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[SUB_VIEW_1]] as %arg2: memref<1x64x26x52x!qElemType1, #NHWC>)
    // CHECK-SAME:       outputs([[BUF]] as %arg3: memref<1x64x26x52x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:         [[INNER:%.*]] = VPUIP.Copy

    // CHECK:         [[COPY_3:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[SUB_VIEW_3]] as %arg2: memref<1x64x26x52x!qElemType1, #NHWC>)
    // CHECK-SAME:       outputs([[BUF_IN_2]] as %arg3: memref<1x64x26x52x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            [[INNER:%.*]] = VPUIP.Copy

    // CHECK:         [[INPLACE_ELTWISE:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[COPY_2]] as %arg2: memref<1x64x26x52x!qElemType1, #NHWC, @CMX_NN>, [[COPY_3:%.*]] as %arg3: memref<1x64x26x52x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[BUF_IN_2]] as %arg4: memref<1x64x26x52x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x26x52x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:            [[INNER:%.*]] = VPUIP.NCEClusterTask

    // CHECK:         return [[MAXPOOL]], [[MAXPOOL_1]], [[INPLACE_ELTWISE]]
}
