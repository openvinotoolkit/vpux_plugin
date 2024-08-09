//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-NPU37XX

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

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, {
        order = #NHWC,
        strides = [32768, 1, 1024, 32]
    },
    @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

module @TestMultiClusterSOK {

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
        DataInfo "conv" : tensor<1x32x32x32xf16>
    }

func.func @main(%arg0: memref<1x16x32x32xf16, #NHWC, @DDR>, %arg1: memref<1x32x32x32xf16, #NHWC, @DDR>) -> memref<1x32x32x32xf16, #NHWC, @DDR> {

    %weights_cst1 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_cst2 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cst2 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // Barriers
    %bar0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %bar2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %bar3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
    %bar10 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !ParentInputDistributed
    %input_1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    %input_2 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>

    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> [0, 1] <32768> -> !ParentOutputDistributed
    %parent_out_cmx_compact = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    %output1 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <32768> -> !OutputDistributed
    %output2 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <32768> -> !OutputDistributed

    %weights1 = VPURT.DeclareBuffer <CMX_NN> [0] <98304> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights2 = VPURT.DeclareBuffer <CMX_NN> [1] <98304> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weight_table1 = VPURT.DeclareBuffer <CMX_NN> [0] <98816> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %weight_table2 = VPURT.DeclareBuffer <CMX_NN> [1] <98816> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>


    // Upload weights and weights table

    VPURT.Task updates(%bar10: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%weights_cst1: memref<16x16x1x1xf16, #NHWC>)
            outputs(%weights1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar10: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%weights_cst2: memref<16x16x1x1xf16, #NHWC>)
            outputs(%weights2: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    }

    VPURT.Task updates(%bar10: !VPURT.Barrier)  {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%weights_table_cst1: memref<16x1x1x4xsi32>)
            outputs(%weight_table1: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar10: !VPURT.Barrier)  {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%weights_table_cst2: memref<16x1x1x4xsi32>)
            outputs(%weight_table2: memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
            -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    }

    // Broadcast input

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%arg0: memref<1x16x32x32xf16, #NHWC, @DDR>)
            outputs(%parent_input_cmx: !ParentInputDistributed)
            -> !ParentInputDistributed
    }

    // 1st tile

    VPURT.Task waits(%bar1, %bar10: !VPURT.Barrier, !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>,
                out_channel_offset = 0
            }
            input(%input_1: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%weights1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%weight_table1: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%parent_input_cmx: !ParentInputDistributed)
            parent_output(%parent_out_cmx: !ParentOutputDistributed)
            outputs(%output1: !OutputDistributed)
            -> !OutputDistributed
            variants : {
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [31, 31, 15],
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
                out_channel_offset = 16
            }
            input(%input_2: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
            weights(%weights2: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            weight_table(%weight_table2: memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
            parent_input(%parent_input_cmx: !ParentInputDistributed)
            parent_output(%parent_out_cmx: !ParentOutputDistributed)
            outputs(%output2: !OutputDistributed)
            -> !OutputDistributed
            variants : {
                DPUTask {
                    outStart = [0, 0, 16],
                    outEnd = [31, 31, 31],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>
                }
            } PPE : {
            }
    }

    // Copyback result

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%parent_out_cmx_compact: memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%arg1: memref<1x32x32x32xf16, #NHWC, @DDR>)
            -> memref<1x32x32x32xf16, #NHWC, @DDR>
    }

    return %arg1: memref<1x32x32x32xf16, #NHWC, @DDR>
}

}

// CHECK:   identifier: "TestMultiClusterSOK",
// CHECK:         task_type: "NNDMATask",
// CHECK:         task: {
// CHECK:           src: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               32,
// CHECK:               32
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 0
// CHECK:             },
// CHECK:             locale: "ProgrammableInput",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:             bit_strides: [
// CHECK:               16,
// CHECK:               262144,
// CHECK:               16,
// CHECK:               8192,
// CHECK:               256
// CHECK:             ]
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               32,
// CHECK:               32
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 0
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0,
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:             bit_strides: [
// CHECK:               16,
// CHECK:               262144,
// CHECK:               16,
// CHECK:               8192,
// CHECK:               256
// CHECK:             ]
// CHECK:         task_type: "NNDMATask",
// CHECK:         task: {
// CHECK:           src: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               32,
// CHECK:               32,
// CHECK:               32
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 32768
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:             bit_strides: [
// CHECK:               16,
// CHECK:               524288,
// CHECK:               16,
// CHECK:               16384,
// CHECK:               512
// CHECK:             ]
// CHECK:           },
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               32,
// CHECK:               32,
// CHECK:               32
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 0
// CHECK:             },
// CHECK:             locale: "ProgrammableOutput",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:             bit_strides: [
// CHECK:               16,
// CHECK:               524288,
// CHECK:               16,
// CHECK:               16384,
// CHECK:               512
// CHECK:             ]
// CHECK:         task_type: "NCE2Task",
// CHECK:             mpe_frequent_mode: "VECTOR_FP16",
// CHECK:             parent_input_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 262144,
// CHECK:                 16,
// CHECK:                 8192,
// CHECK:                 256
// CHECK:               ]
// CHECK:             parent_output_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 32,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 32768
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 524288,
// CHECK:                 16,
// CHECK:                 16384,
// CHECK:                 512
// CHECK:               ]
// CHECK:             input_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 262144,
// CHECK:                 16,
// CHECK:                 8192,
// CHECK:                 256
// CHECK:               ]
// CHECK:             output_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 32768
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 524288,
// CHECK:                 16,
// CHECK:                 16384,
// CHECK:                 512
// CHECK:               ]
// CHECK:             weights_data: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 98304
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 256,
// CHECK:                 16,
// CHECK:                 256,
// CHECK:                 256
// CHECK:               ]
// CHECK:             weights_table: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 4
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 98816
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:               bit_strides: [
// CHECK:                 32,
// CHECK:                 128,
// CHECK:                 128,
// CHECK:                 128,
// CHECK:                 32
// CHECK:               ]
// CHECK:           variant: [
// CHECK:               mpe_mode: "VECTOR_FP16",
// CHECK:               workload_end_X: 31,
// CHECK:               workload_end_Y: 31,
// CHECK:               workload_end_Z: 15
// CHECK:         task_type: "NCE2Task",
// CHECK:             parent_input_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 262144,
// CHECK:                 16,
// CHECK:                 8192,
// CHECK:                 256
// CHECK:               ]
// CHECK:             parent_output_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 32,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 32768
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 524288,
// CHECK:                 16,
// CHECK:                 16384,
// CHECK:                 512
// CHECK:               ]
// CHECK:             input_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 262144,
// CHECK:                 16,
// CHECK:                 8192,
// CHECK:                 256
// CHECK:               ]
// CHECK:             output_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 32768
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 524288,
// CHECK:                 16,
// CHECK:                 16384,
// CHECK:                 512
// CHECK:               ]
// CHECK:             weights_data: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 98304
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               bit_strides: [
// CHECK:                 16,
// CHECK:                 256,
// CHECK:                 16,
// CHECK:                 256,
// CHECK:                 256
// CHECK:               ]
// CHECK:             weights_table: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 4
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 98816
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:               bit_strides: [
// CHECK:                 32,
// CHECK:                 128,
// CHECK:                 128,
// CHECK:                 128,
// CHECK:                 32
// CHECK:               ]
// CHECK:             out_channel_offset: 16
// CHECK:           variant: [
// CHECK:               mpe_mode: "VECTOR_FP16",
// CHECK:               workload_start_Z: 16,
// CHECK:               workload_end_X: 31,
// CHECK:               workload_end_Y: 31,
// CHECK:               workload_end_Z: 31
