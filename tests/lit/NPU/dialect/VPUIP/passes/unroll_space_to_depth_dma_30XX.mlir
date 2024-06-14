//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-space-to-depth-dma  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UnrollSpaceToDepthDMABlockFirstNHWC(%input: memref<1x2x4x6xf16, #NHWC>, %output: memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%input : memref<1x2x4x6xf16, #NHWC>) outputs(%inBuffer : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%inBuffer : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%outBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%output :  memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    }

    return %output: memref<1x8x2x3xf16, #NHWC>
    //CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs(%arg0 : memref<1x2x4x6xf16, #NHWC>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 48 : i64, srcWidth = 24 : i64, srcStride = 48 : i64, srcPlaneStride = 24 : i64, dstWidth = 8 : i64, dstStride = 16 : i64, dstPlaneStride = 8 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_0]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs(%arg1 : memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    //CHECK:    }

    //CHECK:    return %arg1 : memref<1x8x2x3xf16, #NHWC>
}

func.func @UnrollSpaceToDepthDMABlockFirstNCHW(%input: memref<1x2x4x6xf16>, %output: memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%input : memref<1x2x4x6xf16>) outputs(%inBuffer : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%inBuffer : memref<1x2x4x6xf16, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x8x2x3xf16, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%outBuffer : memref<1x8x2x3xf16, [@CMX_NN, 0]>) outputs(%output :  memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16>
    }

    return %output: memref<1x8x2x3xf16>
    //CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <146> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <140> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <134> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs(%arg0 : memref<1x2x4x6xf16>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 6 : i64, dstStride = 48 : i64, dstPlaneStride = 24 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_0]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 6 : i64, dstStride = 48 : i64, dstPlaneStride = 24 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_1]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_1]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 6 : i64, dstStride = 48 : i64, dstPlaneStride = 24 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_2]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_2]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 6 : i64, dstStride = 48 : i64, dstPlaneStride = 24 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_3]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_3]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x8x2x3xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs(%arg1 : memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16>
    //CHECK:    }

    //CHECK:    return %arg1 : memref<1x8x2x3xf16>
}

func.func @UnrollSpaceToDepthDMADepthFirstNHWC(%input: memref<1x2x4x6xf16, #NHWC>, %output: memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%input : memref<1x2x4x6xf16, #NHWC>) outputs(%inBuffer : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%inBuffer : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%outBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%output :  memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    }

    return %output: memref<1x8x2x3xf16, #NHWC>
    //CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <180> -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <176> -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <132> -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>


    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs(%arg0 : memref<1x2x4x6xf16, #NHWC>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 4 : i64, dstStride = 16 : i64, dstPlaneStride = 8 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_0]] : memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 4 : i64, dstStride = 16 : i64, dstPlaneStride = 8 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_1]] : memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_1]] : memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 4 : i64, dstStride = 16 : i64, dstPlaneStride = 8 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_2]] : memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_2]] : memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 4 : i64, dstStride = 16 : i64, dstPlaneStride = 8 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_3]] : memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_3]] : memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x1x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs(%arg1 : memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    //CHECK:    }

    //CHECK:    return %arg1 : memref<1x8x2x3xf16, #NHWC>
}

func.func @UnrollSpaceToDepthDMADepthFirstNCHW(%input: memref<1x2x4x6xf16>, %output: memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%input : memref<1x2x4x6xf16>) outputs(%inBuffer : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%inBuffer : memref<1x2x4x6xf16, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x8x2x3xf16, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%outBuffer : memref<1x8x2x3xf16, [@CMX_NN, 0]>) outputs(%output :  memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16>
    }

    return %output: memref<1x8x2x3xf16>
    //CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <182> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <176> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <134> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs(%arg0 : memref<1x2x4x6xf16>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 6 : i64, dstStride = 24 : i64, dstPlaneStride = 12 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_0]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 6 : i64, dstStride = 24 : i64, dstPlaneStride = 12 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_1]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_1]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 6 : i64, dstStride = 24 : i64, dstPlaneStride = 12 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_2]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_2]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 12 : i64, srcWidth = 2 : i64, srcStride = 4 : i64, srcPlaneStride = 2 : i64, dstWidth = 6 : i64, dstStride = 24 : i64, dstPlaneStride = 12 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_3]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_3]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x8x2x3xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs(%arg1 : memref<1x8x2x3xf16>) -> memref<1x8x2x3xf16>
    //CHECK:    }

    //CHECK:    return %arg1 : memref<1x8x2x3xf16>
}

func.func @UnrollSpaceToDepthDMABlockFirstNCHWToNHWC(%input: memref<1x2x4x6xf16>, %output: memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%input : memref<1x2x4x6xf16>) outputs(%inBuffer : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%inBuffer : memref<1x2x4x6xf16, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%outBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%output :  memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    }

    return %output: memref<1x8x2x3xf16, #NHWC>
    //CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x2x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <178> -> memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <130> -> memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <176> -> memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>


    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs(%arg0 : memref<1x2x4x6xf16>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3 : i64, len = 8 : i64, srcWidth = 4 : i64, srcStride = 12 : i64, srcPlaneStride = 4 : i64, dstWidth = 2 : i64, dstStride = 4 : i64, dstPlaneStride = 16 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_0]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3 : i64, len = 8 : i64, srcWidth = 4 : i64, srcStride = 12 : i64, srcPlaneStride = 4 : i64, dstWidth = 2 : i64, dstStride = 4 : i64, dstPlaneStride = 16 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_1]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_1]] : memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3 : i64, len = 8 : i64, srcWidth = 4 : i64, srcStride = 12 : i64, srcPlaneStride = 4 : i64, dstWidth = 2 : i64, dstStride = 4 : i64, dstPlaneStride = 16 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_2]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_2]] : memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:           dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3 : i64, len = 8 : i64, srcWidth = 4 : i64, srcStride = 12 : i64, srcPlaneStride = 4 : i64, dstWidth = 2 : i64, dstStride = 4 : i64, dstPlaneStride = 16 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_3]] : memref<1x1x2x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_3]] : memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x2x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs(%arg1 : memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    //CHECK:    }

    //CHECK:    return %arg1 : memref<1x8x2x3xf16, #NHWC>
}

func.func @UnrollSpaceToDepthDMADepthFirstNCHWToNHWC(%input: memref<1x2x4x6xf16>, %output: memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA inputs(%input : memref<1x2x4x6xf16>) outputs(%inBuffer : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>, output_channel = 2 : i64, output_width = 6 : i64}
                inputs(%inBuffer : memref<1x2x4x6xf16, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%outBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%output :  memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    }

    return %output: memref<1x8x2x3xf16, #NHWC>
    //CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1x1x4x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x4x6xf16, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <136> -> memref<1x1x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x1x4x6xf16, #NHWC, [@CMX_NN, 0]>


    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs(%arg0 : memref<1x2x4x6xf16>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x2x4x6xf16, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:            dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 24 : i64, srcWidth = 12 : i64, srcStride = 24 : i64, srcPlaneStride = 12 : i64, dstWidth = 4 : i64, dstStride = 16 : i64, dstPlaneStride = 24 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_0]] : memref<1x1x4x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x1x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.SpaceToDepthDMA {
    //CHECK-SAME:           block_size = 2 : i64,
    //CHECK-SAME:            dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 2 : i64, len = 24 : i64, srcWidth = 12 : i64, srcStride = 24 : i64, srcPlaneStride = 12 : i64, dstWidth = 4 : i64, dstStride = 16 : i64, dstPlaneStride = 24 : i64>,
    //CHECK-SAME:           mode = #IE.space_to_depth_mode<DEPTH_FIRST>
    //CHECK-SAME:           }
    //CHECK:            inputs([[INPUT_1]] : memref<1x1x4x6xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_1]] : memref<1x1x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs(%arg1 : memref<1x8x2x3xf16, #NHWC>) -> memref<1x8x2x3xf16, #NHWC>
    //CHECK:    }

    //CHECK:    return %arg1 : memref<1x8x2x3xf16, #NHWC>
}
