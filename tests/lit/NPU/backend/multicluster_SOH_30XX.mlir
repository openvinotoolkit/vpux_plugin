//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-VPUX30XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
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

module @TestMultiClusterSOH {

module @UsedMemory {
    IE.MemoryResource 2048 bytes of @DDR
}

IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @UsedMemory {
        IE.MemoryResource 1048576 bytes of @CMX_NN
    }
}
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x32x32xf16>
    }
    outputsInfo : {
        DataInfo "conv" : tensor<1x16x32x32xf16>
    }

func.func @main(%arg0: memref<1x16x32x32xf16, #NHWC, @DDR>, %arg1: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR> {

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // DDR input buffers for SOH tiling
    %input1_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    %input2_ddr = VPURT.DeclareBuffer <DDR> <16384> -> memref<1x16x16x32xf16, #NHWC, @DDR>

    // DDR output buffers for SOH tiling
    %output1_ddr = VPURT.DeclareBuffer <DDR> <32768> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    %output2_ddr = VPURT.DeclareBuffer <DDR> <49152> -> memref<1x16x16x32xf16, #NHWC, @DDR>

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %input1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>
    %input2 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <16384> -> !OutputDistributed
    %output1 = VPURT.DeclareBuffer <CMX_NN> [0] <16384> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>
    %output2 = VPURT.DeclareBuffer <CMX_NN> [1] <16384> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <32768> -> !WeightsDistributed
    %weights_0 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights_1 = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>

    %weight_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33280> -> !WeightsTableDistributed
    %weight_table_0 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %weight_table_1 = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>


    // Upload weights and weights table

    VPURT.Task updates(%bar10: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%weights_cst: memref<16x16x1x1xf16, #NHWC>)
            outputs(%weights: !WeightsDistributed)
            -> !WeightsDistributed
    }

    VPURT.Task updates(%bar10: !VPURT.Barrier)  {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%weights_table_cst: memref<16x1x1x4xsi32>)
            outputs(%weight_table: !WeightsTableDistributed)
            -> !WeightsTableDistributed
    }

    // Upload 1st input tile

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%input1_ddr: memref<1x16x16x32xf16, #NHWC, @DDR>)
            outputs(%input1: memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>
    }

    // Upload 2nd input tile

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%input2_ddr: memref<1x16x16x32xf16, #NHWC, @DDR>)
            outputs(%input2: memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
            -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    }

    // 1st tile

    VPURT.Task waits(%bar1, %bar10: !VPURT.Barrier, !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>,
                is_segmented
            }
            input(%input1: memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%weights_0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%weight_table_0: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%parent_input_cmx: !InputDistributed)
            parent_output(%parent_out_cmx: !OutputDistributed)
            outputs(%output1: memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [31, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>
                }
            } PPE : {
            }
    }

    // 2nd tile

    VPURT.Task waits(%bar1, %bar10: !VPURT.Barrier, !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>,
                is_segmented
            }
            input(%input2: memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
            weights(%weights_1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            weight_table(%weight_table_1: memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
            parent_input(%parent_input_cmx: !InputDistributed)
            parent_output(%parent_out_cmx: !OutputDistributed)
            outputs(%output2: memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
            -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
            variants : {
                DPUTask {
                    outStart = [0, 16, 0],
                    outEnd = [31, 31, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>
                }
            } PPE : {
            }
    }

    // Copyback 1st result tile

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%output1: memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%output1_ddr: memref<1x16x16x32xf16, #NHWC, @DDR>)
            -> memref<1x16x16x32xf16, #NHWC, @DDR>
    }

    // Copyback 2nd result tile

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%output2: memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
            outputs(%output2_ddr: memref<1x16x16x32xf16, #NHWC, @DDR>)
            -> memref<1x16x16x32xf16, #NHWC, @DDR>
    }

    return %arg1: memref<1x16x32x32xf16, #NHWC, @DDR>
}

}

// CHECK:   identifier: "TestMultiClusterSOH",
// CHECK:         task_type: "NNDMATask",
// CHECK:         task: {
// CHECK:           src: {
// CHECK:             dimensions: [
// CHECK:               16,
// CHECK:               16,
// CHECK:               1,
// CHECK:               1
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               32.0,
// CHECK:               2.0,
// CHECK:               32.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:               data_index: 0
// CHECK:             locale: "GraphFile",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:           },
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               16,
// CHECK:               16,
// CHECK:               1,
// CHECK:               1
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               32.0,
// CHECK:               2.0,
// CHECK:               32.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 32768
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0,
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:         task_type: "NNDMATask",
// CHECK:             dimensions: [
// CHECK:               16,
// CHECK:               1,
// CHECK:               1,
// CHECK:               4
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               4.0,
// CHECK:               16.0,
// CHECK:               16.0,
// CHECK:               16.0,
// CHECK:               4.0
// CHECK:             ],
// CHECK:               data_index: 0
// CHECK:             locale: "GraphFile",
// CHECK:             locale_index: [
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "I32",
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               16,
// CHECK:               1,
// CHECK:               1,
// CHECK:               4
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               4.0,
// CHECK:               16.0,
// CHECK:               16.0,
// CHECK:               16.0,
// CHECK:               4.0
// CHECK:             ],
// CHECK:               data_index: 33280
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0,
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "I32",
// CHECK:         task_type: "NCE2Task",
// CHECK:         task: {
// CHECK:           invariant: {
// CHECK:             parent_input_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 0
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             parent_output_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 16384
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             input_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 16384.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 0
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             output_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 16384.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 16384
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:             weights_data: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32.0,
// CHECK:                 2.0,
// CHECK:                 32.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 32768
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             weights_table: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 4
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 4.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 4.0
// CHECK:               ],
// CHECK:                 data_index: 33280
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:             is_segmented: true
// CHECK:           variant: [
// CHECK:               mpe_mode: "VECTOR_FP16",
// CHECK:               workload_end_X: 31,
// CHECK:               workload_end_Y: 15,
// CHECK:               workload_end_Z: 15
// CHECK:         task_type: "NCE2Task",
// CHECK:         task: {
// CHECK:           invariant: {
// CHECK:             parent_input_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 0
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             },
// CHECK:             parent_output_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 16384
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             input_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 16384.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 0
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             output_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 16384.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 16384
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             weights_data: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32.0,
// CHECK:                 2.0,
// CHECK:                 32.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 32768
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             weights_table: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 4
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 4.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 4.0
// CHECK:               ],
// CHECK:                 data_index: 33280
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:             is_segmented: true
// CHECK:           variant: [
// CHECK:               mpe_mode: "VECTOR_FP16",
// CHECK:               workload_start_Y: 16,
// CHECK:               workload_end_X: 31,
// CHECK:               workload_end_Y: 31,
// CHECK:               workload_end_Z: 15
