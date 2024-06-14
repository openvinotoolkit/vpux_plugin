//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-VPUX40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputITI1 = !VPUIP.ITIBuffer<
    1x16x18x32xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1],
    inwardHaloRegions = [], outwardHaloRegions = []
>

!OutputITI0 = !VPUIP.ITIBuffer<
    1x16x17x32xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0],
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0>],
    outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 1, 32], offset = [0, 0, 15, 0], cluster_id = 0,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1>]>]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x16x18x32xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1],
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 17, 0], cluster_id = 1>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 1, 32], offset = [0, 0, 1, 0], cluster_id = 1,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 1, 32], offset = [0, 0, 16, 0], cluster_id = 1,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 0, 0], cluster_id = 2>]>,
    ]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x16x17x32xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 2],
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 0, 0], cluster_id = 2>],
    outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 1, 32], offset = [0, 0, 1, 0], cluster_id = 2,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 17, 0], cluster_id = 1>]>]>

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x18x32xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x16x18x32xf16>
    }

func.func @main(%arg0: memref<1x16x18x32xf16, #NHWC, [@CMX_NN, 1]>, %arg1:  memref<1x16x18x32xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x16x18x32xf16, #NHWC, [@CMX_NN, 1]> {
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !InputITI1
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> !OutputITI1
    %weights = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<16x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [1] <39424> -> memref<16x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
    %parent_out_cmx_tile0 = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> !OutputITI0
    %parent_out_cmx_tile2 = VPURT.DeclareBuffer <CMX_NN> [2] <17408> -> !OutputITI2

    %input = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !InputITI1
    %output = VPURT.DeclareBuffer <CMX_NN> [1] <17408> ->  !OutputITI1

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input: !InputITI1)
            weights(%weights: memref<16x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>)
            weight_table(%weights_table: memref<16x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>)
            parent_input(%parent_input_cmx: !InputITI1)
            parent_output(%parent_out_cmx: !OutputITI1)
            output_ITI_buff(%parent_out_cmx_tile0, %parent_out_cmx_tile2: !OutputITI0, !OutputITI2)
            outputs(%output: !OutputITI1)
            -> !OutputITI1
            variants : {
                DPUTask {
                    outStart = [0, 1, 0],
                    outEnd = [31, 16, 15],
                    inStart = [0, 0, 0],
                    inEnd = [31, 17, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1,
                    haloRegions = [
                        #VPUIP.DPUHaloRegionAttr<xStart = 0, xEnd = 31, yStart = 1, yEnd = 1, zStart = 0, zEnd = 15, targetOffset = 7680, targetClusters = [0], sparsityOffset = 960, targetWidth = 32>,
                        #VPUIP.DPUHaloRegionAttr<xStart = 0, xEnd = 31, yStart = 16, yEnd = 16, zStart = 0, zEnd = 15, targetOffset = 8192, targetClusters = [2], sparsityOffset = 1024, targetWidth = 32>
                    ]
                }
            } PPE : {
            }
    }

    return %arg1:  memref<1x16x18x32xf16, #NHWC, [@CMX_NN, 1]>
}

}


// CHECK:   task_type: "NCE2Task"

// CHECK:   parent_input_tensor: {
// CHECK:     name: "temp-0",
// CHECK:     dimensions: [
// CHECK:       1,
// CHECK:       16,
// CHECK:       18,
// CHECK:       32
// CHECK:     ],
// CHECK:     data: {
// CHECK:       data_index: 0
// CHECK:     },
// CHECK:     locale: "VPU_CMX_NN",
// CHECK:     locale_index: [
// CHECK:       1
// CHECK:     ],
// CHECK:     bit_strides: [
// CHECK:       16,
// CHECK:       147456,
// CHECK:       16,
// CHECK:       8192,
// CHECK:       256
// CHECK:     ]

// CHECK:   parent_output_tensor: {
// CHECK:     name: "temp-1",
// CHECK:     dimensions: [
// CHECK:       1,
// CHECK:       16,
// CHECK:       18,
// CHECK:       32
// CHECK:     ],
// CHECK:     data: {
// CHECK:       data_index: 17408
// CHECK:     },
// CHECK:     locale: "VPU_CMX_NN",
// CHECK:     locale_index: [
// CHECK:       1
// CHECK:     ],
// CHECK:     bit_strides: [
// CHECK:       16,
// CHECK:       147456,
// CHECK:       16,
// CHECK:       8192,
// CHECK:       256
// CHECK:     ]

// CHECK:   input_data: {
// CHECK:     name: "temp-6",
// CHECK:     dimensions: [
// CHECK:       1,
// CHECK:       16,
// CHECK:       18,
// CHECK:       32
// CHECK:     ],
// CHECK:     data: {
// CHECK:       data_index: 0
// CHECK:     },
// CHECK:     locale: "VPU_CMX_NN",
// CHECK:     locale_index: [
// CHECK:       1
// CHECK:     ],
// CHECK:     bit_strides: [
// CHECK:       16,
// CHECK:       147456,
// CHECK:       16,
// CHECK:       8192,
// CHECK:       256
// CHECK:     ]

// CHECK:   output_data: {
// CHECK:     name: "temp-7",
// CHECK:     dimensions: [
// CHECK:       1,
// CHECK:       16,
// CHECK:       18,
// CHECK:       32
// CHECK:     ],
// CHECK:     data: {
// CHECK:       data_index: 17408
// CHECK:     },
// CHECK:     locale: "VPU_CMX_NN",
// CHECK:     locale_index: [
// CHECK:       1
// CHECK:     ],
// CHECK:     bit_strides: [
// CHECK:       16,
// CHECK:       147456,
// CHECK:       16,
// CHECK:       8192,
// CHECK:       256
// CHECK:     ]

//CHECK:   variant: [
//CHECK:     {
//CHECK:       mpe_mode: "CUBOID_16x16",
//CHECK:       workload_start_Y: 1,
//CHECK:       workload_end_X: 31,
//CHECK:       workload_end_Y: 16,
//CHECK:       workload_end_Z: 15,
//CHECK:       halo_regions: [
//CHECK:         {
//CHECK:           x_end: 31,
//CHECK:           y_start: 1,
//CHECK:           y_end: 1,
//CHECK:           z_end: 15,
//CHECK:           target_offset: 7680,
//CHECK:           target_clusters: [
//CHECK:             0
//CHECK:           ],
//CHECK:           sparsity_offset: 960,
//CHECK:           target_width: 32
//CHECK:         },
//CHECK:         {
//CHECK:           x_end: 31,
//CHECK:           y_start: 16,
//CHECK:           y_end: 16,
//CHECK:           z_end: 15,
//CHECK:           target_offset: 8192,
//CHECK:           target_clusters: [
//CHECK:             2
//CHECK:           ],
//CHECK:           sparsity_offset: 1024,
//CHECK:           target_width: 32
//CHECK:         }
//CHECK:       ],
//CHECK:       idu_workload_size_x: 32,
//CHECK:       idu_workload_size_y: 18,
//CHECK:       idu_workload_size_z: 16
//CHECK:     }
//CHECK:   ]
