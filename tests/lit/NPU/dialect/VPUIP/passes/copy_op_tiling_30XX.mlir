//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tile-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @SplitDoubleStrideCopyByChannels(
        %arg0: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
        %arg1: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
        -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    return %0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:      memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @SplitOutputDoubleStrideCopyByChannels(
        %arg0: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>,
        %arg1: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
        -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    return %0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:      memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @LegalizeCopy(
        %arg0: memref<1x64x512x512xf16, #NCHW>,
        %arg1: memref<1x64x512x512xf16, #NCHW>)
        -> memref<1x64x512x512xf16, #NCHW> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x64x512x512xf16, #NCHW>)
                   outputs(%arg1 : memref<1x64x512x512xf16, #NCHW>)
                   -> memref<1x64x512x512xf16, #NCHW>

    return %0 : memref<1x64x512x512xf16, #NCHW>

    // Currently, large Copy nodes are tiled C-wise

    // Cut first tile:
    // CHECK: [[SUBVIEW_SRC_1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_1:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the second tile:
    // CHECK: [[SUBVIEW_SRC_2:%.*]] = VPUIP.SubView %arg0 [0, 31, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_2:%.*]] = VPUIP.SubView %arg1 [0, 31, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // CHECK: [[SUBVIEW_SRC_3:%.*]] = VPUIP.SubView %arg0 [0, 62, 0, 0] [1, 2, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_3:%.*]] = VPUIP.SubView %arg1 [0, 62, 0, 0] [1, 2, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_3]] : memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_3]] : memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:          memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Concatenate the resulting output tiles:
    // CHECK: [[VAR6:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_RET_1]], [[COPY_RET_2]], [[COPY_RET_3]] :
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x64x512x512xf16>)
    // CHECK-SAME:        -> memref<1x64x512x512xf16>
    // CHECK: return [[VAR6]] : memref<1x64x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @LegalizeStridedCopy(
        %arg0: memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>,
        %arg1: memref<1x64x512x512xf16, #NCHW>)
        -> memref<1x64x512x512xf16, #NCHW> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
                   outputs(%arg1 : memref<1x64x512x512xf16, #NCHW>)
                   -> memref<1x64x512x512xf16, #NCHW>

    return %0 : memref<1x64x512x512xf16, #NCHW>

    // Currently, large Copy nodes are tiled C-wise
    // If the Copy is strided, the strides should be preserved

    // Cut the first tile:
    // CHECK: [[SUBVIEW_SRC_1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_1:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // The Copy-tile preserves the original strides:
    // CHECK: [[COPY_RET_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the second tile:
    // CHECK: [[SUBVIEW_SRC_2:%.*]] = VPUIP.SubView %arg0 [0, 31, 0, 0] [1, 31, 512, 512]
    // CHECK-SAME:      memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_2:%.*]] = VPUIP.SubView %arg1 [0, 31, 0, 0] [1, 31, 512, 512]
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // The Copy-tile preserves the original strides:
    // CHECK: [[COPY_RET_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the third tile:
    // CHECK: [[SUBVIEW_SRC_3:%.*]] = VPUIP.SubView %arg0 [0, 62, 0, 0] [1, 2, 512, 512]
    // CHECK-SAME:          memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:          memref<1x2x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_3:%.*]] = VPUIP.SubView %arg1 [0, 62, 0, 0] [1, 2, 512, 512]
    // CHECK-SAME:          memref<1x64x512x512xf16>
    // CHECK-SAME:          memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // CHECK: [[COPY_RET_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[SUBVIEW_SRC_3]] : memref<1x2x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:          outputs([[SUBVIEW_DST_3]] : memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:            memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Concatenate the resulting output tiles:
    // CHECK: [[RESULT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_RET_1]], [[COPY_RET_2]], [[COPY_RET_3]] :
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x64x512x512xf16>)
    // CHECK-SAME:        memref<1x64x512x512xf16>

    // CHECK: return [[RESULT]] : memref<1x64x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @DoNotLegalizeCopy(
        %arg0: memref<1x315x221x241xi8, #NCHW>,
        %arg1: memref<1x315x221x241xi8, #NCHW>)
        -> memref<1x315x221x241xi8, #NCHW> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x315x221x241xi8, #NCHW>)
                   outputs(%arg1 : memref<1x315x221x241xi8, #NCHW>)
                   -> memref<1x315x221x241xi8, #NCHW>

    return %0 : memref<1x315x221x241xi8, #NCHW>

    // Small enough Copy nodes (those with transaction volume less than (16MB - 1Byte)) should not be affected by the pass

    // CHECK: [[VAR0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x315x221x241xi8>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x315x221x241xi8>)
    // CHECK-SAME:        -> memref<1x315x221x241xi8>
    // CHECK: return [[VAR0]] : memref<1x315x221x241xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func  @SplitDoubleStrideCopy(
        %arg0: memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>,
        %arg1: memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
        -> memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
                   outputs(%arg1 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
                   -> memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    return %0 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 255, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 255, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0] [1, 1, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0] [1, 1, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>,
    // CHECK-SAME:      memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @SplitByHeight3D(
        %arg0: memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>,
        %arg1: memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
        -> memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
                   outputs(%arg1 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
                   -> memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    return %0 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0] [1, 131, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0] [1, 131, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  -> memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 131, 0] [1, 129, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 131, 0] [1, 129, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  -> memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>,
    // CHECK-SAME:      memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  -> memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @SplitByChannels5D(
        %arg0: memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>,
        %arg1: memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
        -> memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
                   -> memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    return %0 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0, 0] [1, 255, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0, 0] [1, 255, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  -> memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0, 0] [1, 65, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0, 0] [1, 65, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  -> memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>,
    // CHECK-SAME:      memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @SplitByDepth5D(
        %arg0: memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>,
        %arg1: memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
        -> memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
                   outputs(%arg1 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
                   -> memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    return %0 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0, 0] [1, 1, 255, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0, 0] [1, 1, 255, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  -> memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 0, 255, 0, 0] [1, 1, 65, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 0, 255, 0, 0] [1, 1, 65, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  -> memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>,
    // CHECK-SAME:      memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  -> memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
}

//
// -----
//


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputType = memref<3x32x1080x1920x!quant.uniform<u8:f16, 1.2608227898092832:124>, #NHWC, @DDR>
!OutputType = memref<3x32x1080x1920x!quant.uniform<u8:f16, 1.2608227898092832:124>, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>

func.func @RecursiveSplitLargeCopy(
        %arg0: !InputType,
        %arg1: !OutputType)
        -> !OutputType {

    // CHECK-LABEL: @RecursiveSplitLargeCopy
    %0 = VPUIP.Copy inputs(%arg0 : !InputType)
                   outputs(%arg1 : !OutputType)
                   -> !OutputType

    return %0 : !OutputType
    // First split across N axis
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>

    // Next 5 splits across H axis
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_3]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_5]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_6:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_7:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_6]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_7]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_8:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_9:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_8]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_9]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_10:%.+]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 1020, 0] [1, 32, 60, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_11:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 1020, 0] [1, 32, 60, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_10]] : memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_11]] : memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>

    // Concat of splits across H axis for first N split
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[SUBVIEW_1]]0, [[SUBVIEW_1]]3, [[SUBVIEW_1]]6 : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>


    // CHECK:       [[SUBVIEW_12:%.+]] = VPUIP.SubView %arg0 [1, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_13:%.+]] = VPUIP.SubView %arg1 [1, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_14:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_15:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy inputs([[SUBVIEW_14]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_15]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_16:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_17:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_6:%.+]] = VPUIP.Copy inputs([[SUBVIEW_16]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_17]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_18:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_19:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_7:%.+]] = VPUIP.Copy inputs([[SUBVIEW_18]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_19]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_20:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_21:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_8:%.+]] = VPUIP.Copy inputs([[SUBVIEW_20]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_21]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_22:%.+]] = VPUIP.SubView [[SUBVIEW_12]] [0, 0, 1020, 0] [1, 32, 60, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_23:%.+]] = VPUIP.SubView [[SUBVIEW_13]] [0, 0, 1020, 0] [1, 32, 60, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_9:%.+]] = VPUIP.Copy inputs([[SUBVIEW_22]] : memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_23]] : memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_5]], [[COPY_6]], [[COPY_7]], [[COPY_8]], [[COPY_9]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_13]] : memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_24:%.+]] = VPUIP.SubView %arg0 [2, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_25:%.+]] = VPUIP.SubView %arg1 [2, 0, 0, 0] [1, 32, 1080, 1920]
    // CHECK-SAME:         memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_26:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_27:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 0, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_10:%.+]] = VPUIP.Copy inputs([[SUBVIEW_26]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_27]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_28:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_29:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 255, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_11:%.+]] = VPUIP.Copy inputs([[SUBVIEW_28]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_29]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_30:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_31:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 510, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_12:%.+]] = VPUIP.Copy inputs([[SUBVIEW_30]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_31]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_32:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_33:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 765, 0] [1, 32, 255, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_13:%.+]] = VPUIP.Copy inputs([[SUBVIEW_32]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_33]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_34:%.+]] = VPUIP.SubView [[SUBVIEW_24]] [0, 0, 1020, 0] [1, 32, 60, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>
    // CHECK:       [[SUBVIEW_35:%.+]] = VPUIP.SubView [[SUBVIEW_25]] [0, 0, 1020, 0] [1, 32, 60, 1920]
    // CHECK-SAME:         memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK-SAME:         to memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[COPY_14:%.+]] = VPUIP.Copy inputs([[SUBVIEW_34]] : memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [66355200, 1, 61440, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_35]] : memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       [[CONCATVIEW_2:%.+]] = VPUIP.ConcatView inputs([[COPY_10]], [[COPY_11]], [[COPY_12]], [[COPY_13]], [[COPY_14]] : memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x255x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x60x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_25]] : memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>

    // Resulting concat
    // CHECK:       [[CONCATVIEW_3:%.+]] = VPUIP.ConcatView inputs([[CONCATVIEW_0]], [[CONCATVIEW_1]], [[CONCATVIEW_2]] : memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>, memref<1x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:         outputs(%arg1 : memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<3x32x1080x1920x!qElemType, {order = #NHWC, strides = [265420800, 1, 245760, 32]}, @DDR>
    // CHECK:       return [[CONCATVIEW_3]]
}
