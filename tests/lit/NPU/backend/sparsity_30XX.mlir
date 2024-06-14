//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(n, c, h, w) -> (n, h, w, c)>

module @Test {

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
        DataInfo "output" : tensor<1x16x32x32xf16>
    }

func.func @main(%arg0: memref<1x16x32x32xf16, #NHWC>, %arg1: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> {
    %cst_wt = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %cst_wt_table = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    %cst_sm = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]>
    %cst_se = VPURT.DeclareBuffer <CMX_NN> [0] <49920> -> memref<1x1x32x32xi32, #NHWC, [@CMX_NN, 0]>

    %bar = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    VPURT.Task updates(%bar : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x32x32xf16, #NHWC>) outputs(%in : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar : !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
                input(%in : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
                input_sparsity_map(%cst_sm : memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]>)
                input_storage_element_table(%cst_se : memref<1x1x32x32xi32, #NHWC, [@CMX_NN, 0]>)
                weights(%cst_wt : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
                weight_table(%cst_wt_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
                parent_input(%in : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
                parent_input_sparsity_map(%cst_sm : memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]>)
                parent_input_storage_element_table(%cst_se : memref<1x1x32x32xi32, #NHWC, [@CMX_NN, 0]>)
                parent_output(%arg1 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%arg1 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {outEnd = [31, 5, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0], inStart = [0, 0, 0], inEnd = [31, 5, 15]}
        DPUTask {outEnd = [31, 11, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 6, 0], inStart = [0, 6, 0], inEnd = [31, 11, 15]}
        DPUTask {outEnd = [31, 17, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 12, 0], inStart = [0, 12, 0], inEnd = [31, 17, 15]}
        DPUTask {outEnd = [31, 23, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 18, 0], inStart = [0, 18, 0], inEnd = [31, 23, 15]}
        DPUTask {outEnd = [31, 31, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 24, 0], inStart = [0, 24, 0], inEnd = [31, 31, 15]}
        } PPE :  {
        }
    }
    return %arg1: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
}

}

// CHECK:   identifier: "Test"

// CHECK:   task_count: 3,

// CHECK:   task_type: "NCE2Task",

// CHECK:   input_data: {
// CHECK:   name: "temp-2",
// CHECK:   dimensions: [
// CHECK:       1,
// CHECK:       16,
// CHECK:       32,
// CHECK:       32
// CHECK:   ],
// CHECK:   strides: [
// CHECK:       2.0,
// CHECK:       32768.0,
// CHECK:       2.0,
// CHECK:       1024.0,
// CHECK:       32.0
// CHECK:   ],
// CHECK:   data: {
// CHECK:       data_index: 768,
// CHECK:       sparsity_index: 33536,
// CHECK:       storage_element_index: 49920
// CHECK:   },
// CHECK:   locale: "VPU_CMX_NN",
// CHECK:   locale_index: [
// CHECK:       0
// CHECK:   ],
// CHECK:   data_dtype: "FP16",

// CHECK:   output_data: {
// CHECK:   name: "output",
// CHECK:   dimensions: [
// CHECK:       1,
// CHECK:       16,
// CHECK:       32,
// CHECK:       32
// CHECK:   ],
// CHECK:   strides: [
// CHECK:       2.0,
// CHECK:       32768.0,
// CHECK:       2.0,
// CHECK:       1024.0,
// CHECK:       32.0
// CHECK:   ],
// CHECK:   data: {
// CHECK:       data_index: 0
// CHECK:   },
// CHECK:   locale: "ProgrammableOutput",
// CHECK:   locale_index: [
// CHECK:       0
// CHECK:   ],
// CHECK:   data_dtype: "FP16",
