//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --swizzling %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1470000 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
}

func.func @DoNotSwizzleDueToAlignmentMemIncrease(%in : memref<1x16x149x150xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>,
                        %weights_0 : memref<16x16x3x3xf16, #NHWC, @CMX_NN>)
                        -> memref<1x16x149x150xf16, #NHWC, @DDR> {

    %buf0 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    %buf4 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x149x150xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x149x150xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%0 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>) -> memref<1x16x149x150xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        weights(%weights_0 : memref<16x16x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>) -> memref<1x16x149x150xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }


    %3 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%2 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%2 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        parent_output(%buf3 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        outputs(%buf3 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>) -> memref<1x16x149x150xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }
    %4 = VPUIP.Copy
            inputs(%3 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
            outputs(%buf4 : memref<1x16x149x150xf16, #NHWC, @DDR>)
             -> memref<1x16x149x150xf16, #NHWC, @DDR>

    return %4 : memref<1x16x149x150xf16, #NHWC, @DDR>

    // Verify that swizzling is only assigned to the output buffer of the first conv
    // The second conv doesn't have output swizzling because of the memory exceeds CMX size when the input is swizzled
    // CHECK:      [[BUF0:%.+]] = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF1:%.+]] = VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> memref<1x16x149x150xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
    // CHECK:      [[BUF2:%.+]] = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF3:%.+]] = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF4:%.+]] = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @DDR>

    // CHECK:      [[CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x149x150xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK:      [[CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[CONV0]] : memref<1x16x149x150xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
    // CHECK:      [[CONV2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[CONV1]] : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUF3]] : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x148x148xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x148x148xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x148x148xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, #NHWC, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x148x148xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x148x148xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x148x148xf16, #NHWC, [@CMX_NN, 0]>

func.func @SetSwizzlingForConstantButNotActivationDueToCmxSizeLimit(%input : !Input_DDR)
                        -> !Output_DDR {

    %weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed

    %4 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%1 : !InputDistributed)
        parent_output(%output_buff_1_cmx : !OutputDistributed)
        outputs(%output_buff_1_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %5 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
    input(%1 : !InputDistributed)
    weights(%3 : !WeightsDistributed)
    weight_table(%2 : !WeightsTableDistributed)
    parent_input(%1 : !InputDistributed)
    parent_output(%output_buff_2_cmx : !OutputDistributed)
    outputs(%output_buff_2_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %6 = VPUIP.Copy { out_mem_space = @DDR } inputs(%5: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

    return %6 : !Output_DDR

    // Check that swizzling was enabled only for constants but not on activation due to CMX limitation

    // CHECK:      [[CSR_WT:%.+]] = const.Declare memref<16x1x1x4xsi32
    // CHECK-SAME:    swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}
    // CHECK:      [[CSR_W:%.+]] = const.Declare memref<16x16x1x1xf16
    // CHECK-SAME:    swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}

    // CHECK:      [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_W:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<16x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x148x148xf16, #NHWC, @DDR>

    // CHECK:      [[NCE_CT_COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[NCE_CST_COPY_WT:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[NCE_CST_COPY_W:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<16x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[NCE_CT_NCE_TASK_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> variants

    // CHECK:      [[NCE_CT_NCE_TASK_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> variants

    // CHECK:      [[NCE_CT_COPY_OUTPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> memref<1x16x148x148xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1470000 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
}

func.func @CannotSwizzledDueToMultiUserWhichCannotSwizzled(%arg0 : memref<1x256x5x80xf16, #NHWC, @DDR>,
                        %arg1 : memref<1x256x7x80xf16, #NHWC, @DDR>,
                        %weight_table_0 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weight_table_1 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weights : memref<256x256x3x3xf16, #NHWC, @CMX_NN>)
                        -> (memref<1x256x2x40xf16, #NHWC, @DDR>, memref<1x256x3x40xf16, #NHWC, @DDR>) {

    %buf0 = memref.alloc() : memref<1x256x5x80xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x256x7x80xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x256x2x40xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x256x3x40xf16, #NHWC, @CMX_NN>
    %buf4 = memref.alloc() : memref<1x256x2x40xf16, #NHWC, @DDR>
    %buf5 = memref.alloc() : memref<1x256x3x40xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%arg0 : memref<1x256x5x80xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x256x5x80xf16, #NHWC, @CMX_NN>)
             -> memref<1x256x5x80xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.Copy
            inputs(%arg1 : memref<1x256x7x80xf16, #NHWC, @DDR>)
            outputs(%buf1 : memref<1x256x7x80xf16, #NHWC, @CMX_NN>)
             -> memref<1x256x7x80xf16, #NHWC, @CMX_NN>

    %2 = VPUIP.NCEClusterTask
        {
            is_superdense,
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [2, 2],
            minimumHardwareExecutionCost = 99787 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%0 : memref<1x256x5x80xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<256x256x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table_0 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x256x5x80xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x256x2x40xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x256x2x40xf16, #NHWC, @CMX_NN>) -> memref<1x256x2x40xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    cluster_id = 0 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            DPUTask
                {
                    cluster_id = 1 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
        }
        PPE :
        {
            PPETask <NOOP>
            {
                clamp_high = 2147483647 : i64,
                clamp_low = -2147483648 : i64,
                fp_prelu_alpha = 1.000000e+00 : f64,
                lrelu_mult = 1 : i64,
                lrelu_shift = 0 : i64
            }
        }

    %3 = VPUIP.NCEClusterTask
        {
            is_superdense,
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3], kernel_strides = [2, 2],
            minimumHardwareExecutionCost = 99787 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : memref<1x256x7x80xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<256x256x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table_1 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x256x7x80xf16, #NHWC, @CMX_NN>)
        parent_output(%buf3 : memref<1x256x3x40xf16, #NHWC, @CMX_NN>)
        outputs(%buf3 : memref<1x256x3x40xf16, #NHWC, @CMX_NN>) -> memref<1x256x3x40xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    cluster_id = 0 : i64, inEnd = [79, 4, 255], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 1, 255], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            DPUTask
                {
                    cluster_id = 1 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
        }
        PPE :
        {
            PPETask <NOOP>
            {
                clamp_high = 2147483647 : i64,
                clamp_low = -2147483648 : i64,
                fp_prelu_alpha = 1.000000e+00 : f64,
                lrelu_mult = 1 : i64,
                lrelu_shift = 0 : i64
            }
        }

    %4 = VPUIP.Copy
            inputs(%2 : memref<1x256x2x40xf16, #NHWC, @CMX_NN>)
            outputs(%buf4 : memref<1x256x2x40xf16, #NHWC, @DDR>)
             -> memref<1x256x2x40xf16, #NHWC, @DDR>

    %5 = VPUIP.Copy
            inputs(%3 : memref<1x256x3x40xf16, #NHWC, @CMX_NN>)
            outputs(%buf5 : memref<1x256x3x40xf16, #NHWC, @DDR>)
             -> memref<1x256x3x40xf16, #NHWC, @DDR>

    return %4, %5 : memref<1x256x2x40xf16, #NHWC, @DDR>, memref<1x256x3x40xf16, #NHWC, @DDR>

    // CHECK:      [[INPUT_1:%.+]] = memref.alloc() : memref<1x256x5x80xf16, #NHWC, @CMX_NN>
    // CHECK:      [[INPUT_2:%.+]] = memref.alloc() : memref<1x256x7x80xf16, #NHWC, @CMX_NN>
    // CHECK:      [[CONV_OUT_1:%.+]] = memref.alloc() : memref<1x256x2x40xf16, #NHWC, @CMX_NN>
    // CHECK:      [[CONV_OUT_2:%.+]] = memref.alloc() : memref<1x256x3x40xf16, #NHWC, @CMX_NN>
    // CHECK:      [[OUTPUT_1:%.+]] = memref.alloc() : memref<1x256x2x40xf16, #NHWC, @DDR>
    // CHECK:      [[OUTPUT_2:%.+]] = memref.alloc() : memref<1x256x3x40xf16, #NHWC, @DDR>

    // CHECK:      [[COPY_IN_1:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x256x5x80xf16, #NHWC, @DDR>) outputs([[INPUT_1]] : memref<1x256x5x80xf16, #NHWC, @CMX_NN>) -> memref<1x256x5x80xf16, #NHWC, @CMX_NN>
    // CHECK:      [[COPY_IN_2:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x256x7x80xf16, #NHWC, @DDR>) outputs([[INPUT_2]] : memref<1x256x7x80xf16, #NHWC, @CMX_NN>) -> memref<1x256x7x80xf16, #NHWC, @CMX_NN>
    // CHECK:      [[CONV_1:%.+]] = VPUIP.NCEClusterTask {is_superdense, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 99787 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:   input([[COPY_IN_1]] : memref<1x256x5x80xf16, #NHWC, @CMX_NN>) weights(%arg4 : memref<256x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg2 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:   parent_input([[COPY_IN_1]] : memref<1x256x5x80xf16, #NHWC, @CMX_NN>) parent_output([[CONV_OUT_1]] : memref<1x256x2x40xf16, #NHWC, @CMX_NN>) outputs([[CONV_OUT_1]] : memref<1x256x2x40xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:   -> memref<1x256x2x40xf16, #NHWC, @CMX_NN> variants : {
    // CHECK-NEXT:        DPUTask {cluster_id = 0 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT:        DPUTask {cluster_id = 1 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT:      } PPE : {
    // CHECK-NEXT:        PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK-NEXT:      }
    // CHECK:      [[CONV_2:%.+]] = VPUIP.NCEClusterTask {is_superdense, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 99787 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:   input([[COPY_IN_2]] : memref<1x256x7x80xf16, #NHWC, @CMX_NN>) weights(%arg4 : memref<256x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:   parent_input([[COPY_IN_2]] : memref<1x256x7x80xf16, #NHWC, @CMX_NN>) parent_output([[CONV_OUT_2]] : memref<1x256x3x40xf16, #NHWC, @CMX_NN>) outputs([[CONV_OUT_2]] : memref<1x256x3x40xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:   -> memref<1x256x3x40xf16, #NHWC, @CMX_NN> variants : {
    // CHECK-NEXT:        DPUTask {cluster_id = 0 : i64, inEnd = [79, 4, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT:        DPUTask {cluster_id = 1 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT:      } PPE : {
    // CHECK-NEXT:        PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK-NEXT:      }
    // CHECK:      [[OUT_1:%.+]] = VPUIP.Copy inputs([[CONV_1]] : memref<1x256x2x40xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_1]] : memref<1x256x2x40xf16, #NHWC, @DDR>) -> memref<1x256x2x40xf16, #NHWC, @DDR>
    // CHECK:      [[OUT_2:%.+]] = VPUIP.Copy inputs([[CONV_2]] : memref<1x256x3x40xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_2]] : memref<1x256x3x40xf16, #NHWC, @DDR>) -> memref<1x256x3x40xf16, #NHWC, @DDR>
}

// -----

// CHECK-LABEL: @DoNotSwizzle5dTensors

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1470000 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
}

func.func @DoNotSwizzle5dTensors(
    %IN: !VPUIP.DistributedBuffer<25x1x64x2x4xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>,
    %OUT: !VPUIP.DistributedBuffer<25x1x16x2x4xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>
) -> !VPUIP.DistributedBuffer<25x1x16x2x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
    memory_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
}> {
    %WEIGHTS_CST = const.Declare memref<25x16x64x1x1xf16, #GNHWC> = dense<1.0> : tensor<1x25x16x64xf16>,
        [#const.Reshape<[25, 16, 64, 1, 1]>, #const.Reorder<#GNHWC>]

    %WEIGHT_TABLE_CST = const.Declare memref<25x16x1x1x4xsi32> = dense<1> : tensor<25x16x1x1x4xsi32>

    %ALLOC_WEIGHTS = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<25x16x64x1x1xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]}>

    %WEIGHTS = VPUIP.Copy
        inputs(%WEIGHTS_CST : memref<25x16x64x1x1xf16, #GNHWC>)
        outputs(%ALLOC_WEIGHTS : !VPUIP.DistributedBuffer<25x16x64x1x1xf16, #GNHWC, @CMX_NN, {
            mode = "SEGMENTED",
            num_tiles = [4, 1, 1, 1, 1],
            num_clusters = 4 : i64,
            uniform_distributed_segments,
            compute_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
            compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
            memory_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
            memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
        }>) -> !VPUIP.DistributedBuffer<25x16x64x1x1xf16, #GNHWC, @CMX_NN, {
            mode = "SEGMENTED",
            num_tiles = [4, 1, 1, 1, 1],
            num_clusters = 4 : i64,
            uniform_distributed_segments,
            compute_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
            compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
            memory_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
            memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
        }>

    %ALLOC_WEIGHT_TABLE = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<25x16x1x1x4xsi32, #NCDHW, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>

    %WEIGHT_TABLE = VPUIP.Copy
        inputs(%WEIGHT_TABLE_CST : memref<25x16x1x1x4xsi32>)
        outputs(%ALLOC_WEIGHT_TABLE : !VPUIP.DistributedBuffer<25x16x1x1x4xsi32, #NCDHW, @CMX_NN, {
            mode = "SEGMENTED",
            num_tiles = [4, 1, 1, 1, 1],
            num_clusters = 4 : i64,
            uniform_distributed_segments,
            compute_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
            compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
            memory_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
            memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
        }>) -> !VPUIP.DistributedBuffer<25x16x1x1x4xsi32, #NCDHW, @CMX_NN, {
            mode = "SEGMENTED",
            num_tiles = [4, 1, 1, 1, 1],
            num_clusters = 4 : i64,
            uniform_distributed_segments,
            compute_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
            compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
            memory_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
            memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
        }>

    %MATMUL = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
    }
    input(%IN : !VPUIP.DistributedBuffer<25x1x64x2x4xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>)
    weights(%WEIGHTS : !VPUIP.DistributedBuffer<25x16x64x1x1xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>)
    weight_table(%WEIGHT_TABLE : !VPUIP.DistributedBuffer<25x16x1x1x4xsi32, #NCDHW, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>)
    parent_input(%IN : !VPUIP.DistributedBuffer<25x1x64x2x4xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>)
    parent_output(%OUT : !VPUIP.DistributedBuffer<25x1x16x2x4xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>)
    outputs(%OUT : !VPUIP.DistributedBuffer<25x1x16x2x4xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>) -> !VPUIP.DistributedBuffer<25x1x16x2x4xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>
    variants : {
        DPUTask {
            cluster_id = 0 : i64,
            inEnd = [3, 1, 63],
            inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            outEnd = [3, 1, 15],
            outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
            cluster_id = 1 : i64,
            inEnd = [3, 1, 63],
            inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            outEnd = [3, 1, 15],
            outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
            cluster_id = 2 : i64,
            inEnd = [3, 1, 63],
            inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            outEnd = [3, 1, 15],
            outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
            cluster_id = 3 : i64,
            inEnd = [3, 1, 63],
            inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            outEnd = [3, 1, 15],
            outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
    } PPE : {
        PPETask <NOOP> {
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64
        }
    }

    return %OUT : !VPUIP.DistributedBuffer<25x1x16x2x4xf16, #GNHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [4, 1, 1, 1, 1],
        num_clusters = 4 : i64,
        uniform_distributed_segments,
        compute_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
        memory_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
        memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
    }>

    // CHECK-NOT: swizzlingScheme = #VPUIP.SwizzlingSchemeAttr
}
