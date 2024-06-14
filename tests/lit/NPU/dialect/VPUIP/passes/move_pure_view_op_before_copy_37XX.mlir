//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-pure-view-op-before-copy %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x239x18xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @MoveShapeCastWithAlignmentBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x16x478x9xf16, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x239x18xf16, #NHWC, @DDR>
    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x239x18xf16, #NHWC, @CMX_NN>) outputs(%out as %arg2: memref<1x16x239x18xf16, #NHWC>) -> memref<1x16x239x18xf16, #NHWC, @DDR> {
        VPUIP.Copy inputs(%arg1: memref<1x16x239x18xf16, #NHWC, @CMX_NN>) outputs(%arg2: memref<1x16x239x18xf16, #NHWC>) -> memref<1x16x239x18xf16, #NHWC>
    }
    %1 = VPUIP.ShapeCast {shape = [1, 16, 478, 9]} inputs(%0 : memref<1x16x239x18xf16, #NHWC, @DDR>) -> memref<1x16x478x9xf16, #NHWC, @DDR>

    return %1 : memref<1x16x478x9xf16, #NHWC, @DDR>
    //CHECK:    [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 478, 9]} inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x239x18xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x478x9xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    //CHECK:    [[OUTBUFF:%.*]] = memref.alloc() : memref<1x16x478x9xf16, #NHWC, @DDR>
    //CHECK:    [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[SHAPECAST]] as %arg1: memref<1x16x478x9xf16, #NHWC, @CMX_NN>) outputs([[OUTBUFF]] as %arg2: memref<1x16x478x9xf16, #NHWC, @DDR>) -> memref<1x16x478x9xf16, #NHWC, @DDR> {
    //CHECK:        VPUIP.Copy inputs(%arg1 : memref<1x16x478x9xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x478x9xf16, #NHWC, @DDR>) -> memref<1x16x478x9xf16, #NHWC, @DDR>
    //CHECK:    }
    //CHECK:    return [[COPY]] : memref<1x16x478x9xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x16x8xf16, #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
    }
>

func.func @DoNotMoveShapeCastBeforeTilingCopySegmentedDueToAlignment(%arg0: !InputDistributed) -> memref<1x1024x2x1xf16, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x16x8xf16, #NHWC, @DDR>
    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x16x8xf16, #NHWC, @CMX_NN>) outputs(%out as %arg2: memref<1x16x16x8xf16, #NHWC>) -> memref<1x16x16x8xf16, #NHWC, @DDR> {
        VPUIP.Copy inputs(%arg1 : memref<1x16x16x8xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x16x8xf16, #NHWC>) -> memref<1x16x16x8xf16, #NHWC>
    }
    %1 = VPUIP.ShapeCast {shape = [1, 1024, 2, 1]} inputs(%0 : memref<1x16x16x8xf16, #NHWC, @DDR>) -> memref<1x1024x2x1xf16, #NHWC, @DDR>

    return %1 : memref<1x1024x2x1xf16, #NHWC, @DDR>
    //CHECK:    [[OUTBUFF:%.*]] = memref.alloc() : memref<1x16x16x8xf16, #NHWC, @DDR>
    //CHECK:    [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x16x8xf16, #NHWC, @CMX_NN>) outputs([[OUTBUFF]] as %arg2: memref<1x16x16x8xf16, #NHWC>) -> memref<1x16x16x8xf16, #NHWC, @DDR> {
    //CHECK:        VPUIP.Copy inputs(%arg1 : memref<1x16x16x8xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x16x8xf16, #NHWC>) -> memref<1x16x16x8xf16, #NHWC>
    //CHECK:    }
    //CHECK:    [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 1024, 2, 1]} inputs([[COPY]] : memref<1x16x16x8xf16, #NHWC, @DDR>) -> memref<1x1024x2x1xf16, #NHWC, @DDR>
    //CHECK:    return [[SHAPECAST]] : memref<1x1024x2x1xf16, #NHWC, @DDR>
}
