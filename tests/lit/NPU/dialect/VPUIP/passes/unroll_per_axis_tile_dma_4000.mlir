//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-per-axis-tile-dma %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x12x171x512xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 12, 43, 512], [1, 12, 43, 512], [1, 12, 43, 512], [1, 12, 42, 512]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0]],
    memory_shapes = [[1, 12, 43, 512], [1, 12, 43, 512], [1, 12, 43, 512], [1, 12, 42, 512]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0]]
}>

// CHECK-LABEL: @UnrollPerAxisTileDMA
func.func @UnrollPerAxisTileDMA() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x171x512xf16, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 12 : i64}
            inputs(%0 : memref<1x1x171x512xf16, @DDR>)
            outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }

    return %1 : !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_DDR_0:%.*]] = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x22016xf16, @DDR>
    //CHECK:    [[INPUT_DDR_1:%.*]] = VPURT.DeclareBuffer <DDR> <44096> -> memref<1x1x22016xf16, @DDR>
    //CHECK:    [[INPUT_DDR_2:%.*]] = VPURT.DeclareBuffer <DDR> <88128> -> memref<1x1x22016xf16, @DDR>
    //CHECK:    [[INPUT_DDR_3:%.*]] = VPURT.DeclareBuffer <DDR> <132160> -> memref<1x1x21504xf16, @DDR>

    //CHECK:    [[OUTPUT_CMX_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> ->
    //CHECK-SAME:           !VPUIP.DistributedBuffer<
    //CHECK-SAME:               1x12x171x512xf16, #NCHW, @CMX_NN, {
    //CHECK-SAME:               mode = "OVERLAPPED",
    //CHECK-SAME:               num_tiles = [1, 1, 4, 1],
    //CHECK-SAME:               num_clusters = 4 : i64,
    //CHECK-SAME:               uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:      compute_shapes = [[1, 12, 43, 512], [1, 12, 43, 512], [1, 12, 43, 512], [1, 12, 42, 512]],
    //CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0]],
    //CHECK-SAME{LITERAL}:      memory_shapes = [[1, 12, 43, 512], [1, 12, 43, 512], [1, 12, 43, 512], [1, 12, 42, 512]],
    //CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0]]}>

    //CHECK:    [[OUTPUT_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x12x22016xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x12x22016xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_CMX_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x12x22016xf16, [@CMX_NN, 2]>
    //CHECK:    [[OUTPUT_CMX_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> memref<1x12x21504xf16, [@CMX_NN, 3]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:           numPlanes = 1 : i64, len = 528384 : i64,
    //CHECK-SAME:           srcWidth = 44032 : i64, srcStride = 0 : i64, srcPlaneStride = 44032 : i64,
    //CHECK-SAME:           dstWidth = 528384 : i64, dstStride = 528384 : i64, dstPlaneStride = 528384 : i64>, port = 0 : i64}
    //CHECK-SAME:       inputs([[INPUT_DDR_0]] : memref<1x1x22016xf16, @DDR>)
    //CHECK-SAME:       outputs([[OUTPUT_CMX_0]] : memref<1x12x22016xf16, [@CMX_NN, 0]>) -> memref<1x12x22016xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:           numPlanes = 1 : i64, len = 528384 : i64,
    //CHECK-SAME:           srcWidth = 44032 : i64, srcStride = 0 : i64, srcPlaneStride = 44032 : i64,
    //CHECK-SAME:           dstWidth = 528384 : i64, dstStride = 528384 : i64, dstPlaneStride = 528384 : i64>, port = 1 : i64}
    //CHECK-SAME:       inputs([[INPUT_DDR_1]] : memref<1x1x22016xf16, @DDR>)
    //CHECK-SAME:       outputs([[OUTPUT_CMX_1]] : memref<1x12x22016xf16, [@CMX_NN, 1]>) -> memref<1x12x22016xf16, [@CMX_NN, 1]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:           numPlanes = 1 : i64, len = 528384 : i64,
    //CHECK-SAME:           srcWidth = 44032 : i64, srcStride = 0 : i64, srcPlaneStride = 44032 : i64,
    //CHECK-SAME:           dstWidth = 528384 : i64, dstStride = 528384 : i64, dstPlaneStride = 528384 : i64>, port = 0 : i64}
    //CHECK-SAME:       inputs([[INPUT_DDR_2]] : memref<1x1x22016xf16, @DDR>)
    //CHECK-SAME:       outputs([[OUTPUT_CMX_2]] : memref<1x12x22016xf16, [@CMX_NN, 2]>) -> memref<1x12x22016xf16, [@CMX_NN, 2]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PerAxisTileDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:           numPlanes = 1 : i64, len = 516096 : i64,
    //CHECK-SAME:           srcWidth = 43008 : i64, srcStride = 0 : i64, srcPlaneStride = 43008 : i64,
    //CHECK-SAME:           dstWidth = 516096 : i64, dstStride = 516096 : i64, dstPlaneStride = 516096 : i64>, port = 1 : i64}
    //CHECK-SAME:       inputs([[INPUT_DDR_3]] : memref<1x1x21504xf16, @DDR>)
    //CHECK-SAME:       outputs([[OUTPUT_CMX_3]] : memref<1x12x21504xf16, [@CMX_NN, 3]>) -> memref<1x12x21504xf16, [@CMX_NN, 3]>
    //CHECK:    }

    //CHECK:    return [[OUTPUT_CMX_BUF]] : !VPUIP.DistributedBuffer
}
