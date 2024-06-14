//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-concat-view-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @AvoidConcatExtraChannel(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x3x110x512xf16, #NHWC, @DDR>,
        %arg3: memref<1x3x4x512xf16, #NHWC, @DDR>)
         -> (memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg5: memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 57, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg5: memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x57x512xf16, {order = #NHWC}, @DDR>, memref<1x16x57x512xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x16x114x512xf16, #NHWC, @DDR>) -> memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview2 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy0 = VPUIP.Copy
        inputs(%subview2 : memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>)
        -> memref<1x3x110x512xf16, #NHWC, @DDR>
    %subview3 = VPUIP.SubView %concat [0, 0, 110, 0] [1, 3, 4, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy1 = VPUIP.Copy
        inputs(%subview3 : memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>)
        -> memref<1x3x4x512xf16, #NHWC, @DDR>
    return %copy0, %copy1 : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK: [[NEW_BUFFER:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView
    // CHECK-SAME:  [0, 0, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[NEW_BUFFER]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW0]] as %arg4: memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs([[SUBVIEW1]] as %arg5: memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg4 : memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs(%arg5 : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView
    // CHECK-SAME:   [0, 0, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[NEW_BUFFER]] [0, 0, 57, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW2]] as %arg4: memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg5: memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR> {
    // CHECK: VPUIP.Copy inputs(%arg4 : memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs(%arg5 : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs([[NEW_BUFFER]] : memref<1x3x114x512xf16, #NHWC, @DDR>) -> memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY0:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>) -> memref<1x3x110x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 110, 0] [1, 3, 4, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>) -> memref<1x3x4x512xf16, #NHWC, @DDR>
    // CHECK: return [[LAST_COPY0]], [[LAST_COPY1]] : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x16x46x240xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>
!DistributedType2 = !VPUIP.DistributedBuffer<
    1x16x45x240xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

func.func @DoNotAvoidConcatExtraChannel(%arg0 : memref<1x1x136x240xf16, @DDR>) -> memref<1x1x136x240xf16, @DDR> {
    %0 = VPURT.AllocDistributed -> !DistributedType1
    %alloc = memref.alloc() : memref<1x16x136x240xf16, @DDR>
    %1 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 46, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg1: memref<1x16x46x240xf16, @CMX_NN>) outputs(%1 as %arg2: memref<1x16x46x240xf16>) -> memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR> {
      %12 = VPUIP.Copy inputs(%arg1 : memref<1x16x46x240xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x46x240xf16>) -> memref<1x16x46x240xf16>
    }
    %3 = VPURT.AllocDistributed -> !DistributedType2
    %4 = VPUIP.SubView %alloc [0, 0, 46, 0] [1, 16, 45, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x16x45x240xf16, @CMX_NN>) outputs(%4 as %arg2: memref<1x16x45x240xf16>) -> memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR> {
      %12 = VPUIP.Copy inputs(%arg1 : memref<1x16x45x240xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x45x240xf16>) -> memref<1x16x45x240xf16>
    }
    %6 = VPURT.AllocDistributed -> !DistributedType2
    %7 = VPUIP.SubView %alloc [0, 0, 91, 0] [1, 16, 45, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %8 = VPUIP.NCEClusterTiling inputs(%6 as %arg1: memref<1x16x45x240xf16, @CMX_NN>) outputs(%7 as %arg2: memref<1x16x45x240xf16>) -> memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR> {
      %12 = VPUIP.Copy inputs(%arg1 : memref<1x16x45x240xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x45x240xf16>) -> memref<1x16x45x240xf16>
    }
    %9 = VPUIP.ConcatView inputs(%2, %5, %8 : memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>, memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>,
        memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>) outputs(%alloc : memref<1x16x136x240xf16, @DDR>) -> memref<1x16x136x240xf16, @DDR>
    %10 = VPUIP.SubView %9 [0, 1, 0, 0] [1, 1, 136, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %11 = VPUIP.Copy inputs(%10 : memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>) outputs(%arg0 : memref<1x1x136x240xf16, @DDR>) -> memref<1x1x136x240xf16, @DDR>

    return %11 : memref<1x1x136x240xf16, @DDR>

    // CHECK:   [[ALLOCDISTRIBUTED0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x46x240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x16x136x240xf16, @DDR>
    // CHECK:   [[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 16, 46, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:   [[CLUSTERTILLING0:%.+]] = VPUIP.NCEClusterTiling inputs([[ALLOCDISTRIBUTED0]] as %arg1: memref<1x16x46x240xf16, @CMX_NN>)
    // CHECK-SAME:  outputs([[SUBVIEW0]] as %arg2: memref<1x16x46x240xf16>) -> memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR> {
    // CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x16x46x240xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x46x240xf16>) -> memref<1x16x46x240xf16>
    // CHECK:   }
    // CHECK:   [[ALLOCDISTRIBUTED1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x45x240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 46, 0] [1, 16, 45, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:   [[CLUSTERTILLING1:%.+]] = VPUIP.NCEClusterTiling inputs([[ALLOCDISTRIBUTED1]] as %arg1: memref<1x16x45x240xf16, @CMX_NN>)
    // CHECK-SAME:  outputs([[SUBVIEW1]] as %arg2: memref<1x16x45x240xf16>) -> memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR> {
    // CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x16x45x240xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x45x240xf16>) -> memref<1x16x45x240xf16>
    // CHECK:   }
    // CHECK:   [[ALLOCDISTRIBUTED2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x45x240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 91, 0] [1, 16, 45, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:   [[CLUSTERTILLING2:%.+]] = VPUIP.NCEClusterTiling inputs([[ALLOCDISTRIBUTED2]] as %arg1: memref<1x16x45x240xf16, @CMX_NN>)
    // CHECK-SAME:  outputs([[SUBVIEW2]] as %arg2: memref<1x16x45x240xf16>) -> memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR> {
    // CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x16x45x240xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x45x240xf16>) -> memref<1x16x45x240xf16>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CLUSTERTILLING0]], [[CLUSTERTILLING1]], [[CLUSTERTILLING2]] : memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>,
    // CHECK-SAME:  memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>, memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>)
    // CHECK-SAME:  outputs(%alloc : memref<1x16x136x240xf16, @DDR>) -> memref<1x16x136x240xf16, @DDR>
    // CHECK:   [[SUBVIEW3:%.+]] = VPUIP.SubView [[CONCAT]] [0, 1, 0, 0] [1, 1, 136, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>)
    // CHECK-SAME:  outputs({{[^:]+}} : memref<1x1x136x240xf16, @DDR>) -> memref<1x1x136x240xf16, @DDR>

    // CHECK:   return [[COPY]] : memref<1x1x136x240xf16, @DDR>

  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @AvoidConcatExtraChannelAndChannelOffsetNotEqualZero(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x3x110x512xf16, #NHWC, @DDR>,
        %arg3: memref<1x3x4x512xf16, #NHWC, @DDR>)
         -> (memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg5: memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 57, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg5: memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x57x512xf16, {order = #NHWC}, @DDR>, memref<1x16x57x512xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x16x114x512xf16, #NHWC, @DDR>) -> memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview2 = VPUIP.SubView %concat [0, 3, 0, 0] [1, 3, 110, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy0 = VPUIP.Copy
        inputs(%subview2 : memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>)
        -> memref<1x3x110x512xf16, #NHWC, @DDR>
    %subview3 = VPUIP.SubView %concat [0, 3, 110, 0] [1, 3, 4, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy1 = VPUIP.Copy
        inputs(%subview3 : memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>)
        -> memref<1x3x4x512xf16, #NHWC, @DDR>
    return %copy0, %copy1 : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK: [[NEW_BUFFER:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView
    // CHECK-SAME:  [0, 3, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[NEW_BUFFER]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW0]] as %arg4: memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs([[SUBVIEW1]] as %arg5: memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg4 : memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs(%arg5 : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView
    // CHECK-SAME:  [0, 3, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[NEW_BUFFER]]
    // CHECK-SAME:  [0, 0, 57, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW2]] as %arg4: memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg5: memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR> {
    // CHECK: VPUIP.Copy inputs(%arg4 : memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs(%arg5 : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs([[NEW_BUFFER]] : memref<1x3x114x512xf16, #NHWC, @DDR>) -> memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY0:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>) -> memref<1x3x110x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 110, 0] [1, 3, 4, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>) -> memref<1x3x4x512xf16, #NHWC, @DDR>
    // CHECK: return [[LAST_COPY0]], [[LAST_COPY1]] : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRData0 = memref<1x16x57x512xf16, {order = #NHWC}, @DDR>
!IODDRSM0 = memref<1x16x57x512xi1, {order = #NHWC}, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<
    data=!IODDRData0,
    sparsity_map=!IODDRSM0
>

!IODDRSparse1 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>
>
!IODistrCMXSparse0 = !VPUIP.SparseBuffer<

    data=!VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>,
    sparsity_map=!VPUIP.DistributedBuffer<
    1x16x57x512xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>
>

!IODDRData2 = memref<1x16x57x512xf16, #NHWC>
!IODDRSM2 = memref<1x16x57x512xi1, #NHWC>
!IODDRSparse2 = !VPUIP.SparseBuffer<
    data=!IODDRData2,
    sparsity_map=!IODDRSM2
>

!IODDRSparse3 = !VPUIP.SparseBuffer<
    data=memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x16x57x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IOCMXData0 = memref<1x16x57x512xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x16x57x512xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<
    data=!IOCMXData0,
    sparsity_map=!IOCMXSM0
>

!IODDRData4 = memref<1x16x114x512xf16, #NHWC, @DDR>
!IODDRSM4 = memref<1x16x114x512xi1, #NHWC, @DDR>
!IODDRSparse4 = !VPUIP.SparseBuffer<
    data=!IODDRData4,
    sparsity_map=!IODDRSM4
>

!IODDRSparse5 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse6 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse7 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>
>

// CHECK-LABEL: @AvoidConcatExtraChannelSparse
func.func @AvoidConcatExtraChannelSparse(%arg0: !IODistrCMXSparse0, %arg1: !IODistrCMXSparse0, %arg2: !IODDRSparse1, %arg3: !IODDRSparse7) -> (!IODDRSparse1, !IODDRSparse7) {
    %0 = memref.alloc() : !IODDRData4
    %1 = memref.alloc() : !IODDRSM4
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODDRSparse4

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %4 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: !IOCMXSparse0) outputs(%3 as %arg5: !IODDRSparse2) -> !IODDRSparse0 {
      %12 = VPUIP.Copy inputs(%arg4 : !IOCMXSparse0) outputs(%arg5 : !IODDRSparse2) -> !IODDRSparse2
    }
    %5 = VPUIP.SubView %2 [0, 0, 57, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %6 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: !IOCMXSparse0) outputs(%5 as %arg5: !IODDRSparse2) -> !IODDRSparse0 {
      %12 = VPUIP.Copy inputs(%arg4 : !IOCMXSparse0) outputs(%arg5 : !IODDRSparse2) -> !IODDRSparse2
    }
    %7 = VPUIP.ConcatView inputs(%4, %6 : !IODDRSparse0, !IODDRSparse0) outputs(%2 : !IODDRSparse4) -> !IODDRSparse4
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 3, 110, 512] : !IODDRSparse4 to !IODDRSparse6
    %9 = VPUIP.Copy inputs(%8 : !IODDRSparse6) outputs(%arg2 : !IODDRSparse1) -> !IODDRSparse1
    %10 = VPUIP.SubView %7 [0, 0, 110, 0] [1, 3, 4, 512] : !IODDRSparse4 to !IODDRSparse5
    %11 = VPUIP.Copy inputs(%10 : !IODDRSparse5) outputs(%arg3 : !IODDRSparse7) -> !IODDRSparse7
    return %9, %11 : !IODDRSparse1, !IODDRSparse7

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xi1, #NHWC, @DDR>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x3x114x512xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs([[SUBVIEW_1]] as %arg5: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy inputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs(%arg5 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 57, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs([[SUBVIEW_3]] as %arg5: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>> {
    // CHECK:       [[inner_1:%.+]] = VPUIP.Copy inputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs(%arg5 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 0, 0] [1, 3, 110, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 110, 0] [1, 3, 4, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_5]] : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>

    // CHECK:       return [[COPY_2]], [[COPY_3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRData0 = memref<1x16x57x512xf16, {order = #NHWC}, @DDR>
!IODDRSM0 = memref<1x16x57x512xi1, {order = #NHWC}, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<
    data=!IODDRData0,
    sparsity_map=!IODDRSM0
>

!IODDRSparse1 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>
>
!IODistrCMXSparse0 = !VPUIP.SparseBuffer<

    data=!VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>,
    sparsity_map=!VPUIP.DistributedBuffer<
    1x16x57x512xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>
>

!IODDRData2 = memref<1x16x57x512xf16, #NHWC>
!IODDRSM2 = memref<1x16x57x512xi1, #NHWC>
!IODDRSparse2 = !VPUIP.SparseBuffer<
    data=!IODDRData2,
    sparsity_map=!IODDRSM2
>

!IODDRSparse3 = !VPUIP.SparseBuffer<
    data=memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x16x57x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IOCMXData0 = memref<1x16x57x512xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x16x57x512xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<
    data=!IOCMXData0,
    sparsity_map=!IOCMXSM0
>

!IODDRData4 = memref<1x16x114x512xf16, #NHWC, @DDR>
!IODDRSM4 = memref<1x16x114x512xi1, #NHWC, @DDR>
!IODDRSparse4 = !VPUIP.SparseBuffer<
    data=!IODDRData4,
    sparsity_map=!IODDRSM4
>

!IODDRSparse5 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse6 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse7 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>
>

// CHECK-LABEL: @AvoidConcatExtraChannelSparseAndChannelOffsetNotEqualZero
func.func @AvoidConcatExtraChannelSparseAndChannelOffsetNotEqualZero(%arg0: !IODistrCMXSparse0, %arg1: !IODistrCMXSparse0, %arg2: !IODDRSparse1, %arg3: !IODDRSparse7) -> (!IODDRSparse1, !IODDRSparse7) {
    %0 = memref.alloc() : !IODDRData4
    %1 = memref.alloc() : !IODDRSM4
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODDRSparse4

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %4 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: !IOCMXSparse0) outputs(%3 as %arg5: !IODDRSparse2) -> !IODDRSparse0 {
      %12 = VPUIP.Copy inputs(%arg4 : !IOCMXSparse0) outputs(%arg5 : !IODDRSparse2) -> !IODDRSparse2
    }
    %5 = VPUIP.SubView %2 [0, 0, 57, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %6 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: !IOCMXSparse0) outputs(%5 as %arg5: !IODDRSparse2) -> !IODDRSparse0 {
      %12 = VPUIP.Copy inputs(%arg4 : !IOCMXSparse0) outputs(%arg5 : !IODDRSparse2) -> !IODDRSparse2
    }
    %7 = VPUIP.ConcatView inputs(%4, %6 : !IODDRSparse0, !IODDRSparse0) outputs(%2 : !IODDRSparse4) -> !IODDRSparse4
    %8 = VPUIP.SubView %7 [0, 3, 0, 0] [1, 3, 110, 512] : !IODDRSparse4 to !IODDRSparse6
    %9 = VPUIP.Copy inputs(%8 : !IODDRSparse6) outputs(%arg2 : !IODDRSparse1) -> !IODDRSparse1
    %10 = VPUIP.SubView %7 [0, 3, 110, 0] [1, 3, 4, 512] : !IODDRSparse4 to !IODDRSparse5
    %11 = VPUIP.Copy inputs(%10 : !IODDRSparse5) outputs(%arg3 : !IODDRSparse7) -> !IODDRSparse7
    return %9, %11 : !IODDRSparse1, !IODDRSparse7

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xi1, #NHWC, @DDR>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x3x114x512xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 3, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs([[SUBVIEW_1]] as %arg5: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy inputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs(%arg5 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg1 [0, 3, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 57, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs([[SUBVIEW_3]] as %arg5: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>> {
    // CHECK:       [[inner_1:%.+]] = VPUIP.Copy inputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs(%arg5 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 0, 0] [1, 3, 110, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 110, 0] [1, 3, 4, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_5]] : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>

    // CHECK:       return [[COPY_2]], [[COPY_3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @FuseConcatViewOps(
        %arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
         -> memref<1x24x144x256xf16, #NHWC, @DDR> {
    %input0 = VPURT.AllocDistributed -> !InputDistributed
    %input1 = VPURT.AllocDistributed -> !InputDistributed

    %0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %2 = VPUIP.NCEClusterTiling inputs(%input0 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
        VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
    }
    %3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %4 = VPUIP.NCEClusterTiling inputs(%input1 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
        VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
    }
    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x24x144x256xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 16, 144, 256] : memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %8 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%7 : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %9 = VPUIP.SubView %6 [0, 16, 0, 0] [1, 8, 144, 256] : memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %10 = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs(%9 : memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %11 = VPUIP.ConcatView inputs(%8, %10 : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) outputs(%6 : memref<1x24x144x256xf16, #NHWC, @DDR>) -> memref<1x24x144x256xf16, #NHWC, @DDR>

    return %11 : memref<1x24x144x256xf16, #NHWC, @DDR>


    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x24x144x256xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 16, 72, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_0]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR> {
    // CHECK:       [[COPY_0_INNER:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 72, 0] [1, 16, 72, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_1]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_1]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR> {
    // CHECK:       [[COPY_1_INNER:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 16, 0, 0] [1, 8, 144, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_2]] : memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[COPY_2]]
    // CHECK-SAME:          memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK-SAME:          memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK-SAME:          memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          outputs([[OUTPUT_BUFF]] : memref<1x24x144x256xf16, #NHWC, @DDR>) -> memref<1x24x144x256xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCATVIEW]] : memref<1x24x144x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x96x336xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @NotFuseConcatViewOpsWithStrideLevelIs3( ) -> memref<1x32x384x672xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !InputDistributed
    %1 = VPURT.AllocDistributed -> !InputDistributed
    %2 = VPURT.AllocDistributed -> !InputDistributed
    %3 = VPURT.AllocDistributed -> !InputDistributed

    %4 = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %6 = VPUIP.NCEClusterTiling
            inputs(%0 as %arg2: memref<1x32x96x336xf16, #NHWC, @CMX_NN>)
            outputs(%5 as %arg3: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR> {
            VPUIP.Copy inputs(%arg2 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    }

    %7 = VPUIP.SubView %4 [0, 0, 96, 0] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %8 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg2: memref<1x32x96x336xf16, #NHWC, @CMX_NN>)
            outputs(%7 as %arg3: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR> {
            VPUIP.Copy inputs(%arg2 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    }

    %9 = VPUIP.SubView %4 [0, 0, 0, 1] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %10 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg2: memref<1x32x96x336xf16, #NHWC, @CMX_NN>)
            outputs(%9 as %arg3: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR> {
            VPUIP.Copy inputs(%arg2 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    }

    %11 = VPUIP.SubView %4 [0, 0, 96, 1] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %12 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg2: memref<1x32x96x336xf16, #NHWC, @CMX_NN>)
            outputs(%11 as %arg3: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR> {
            VPUIP.Copy inputs(%arg2 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    }

    %13 = VPUIP.ConcatView inputs(%6, %8, %10, %12 :
                memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>,
                memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>,
                memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>,
                memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
                outputs(%4 : memref<1x32x192x672xf16, #NHWC, @DDR>)
                    -> memref<1x32x192x672xf16, #NHWC, @DDR>

    %14 = memref.alloc() : memref<1x32x384x672xf16, #NHWC, @DDR>
    %15 = VPUIP.SubView %14 [0, 0, 0, 0] [1, 32, 192, 672] [1, 1, 2, 1]
            : memref<1x32x384x672xf16, #NHWC, @DDR> to memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>
    %16 = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    %17 = VPUIP.Copy inputs(%16 : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs(%15 : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>) -> memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>

    %18 = VPUIP.SubView %14 [0, 0, 1, 0] [1, 32, 192, 672] [1, 1, 2, 1]
            : memref<1x32x384x672xf16, #NHWC, @DDR> to memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>
    %19 = VPUIP.Copy inputs(%13 : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs(%18 : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>) -> memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>

    %20 = VPUIP.ConcatView inputs(%17, %19 :
                memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>,
                memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>)
                outputs(%14 : memref<1x32x384x672xf16, #NHWC, @DDR>) -> memref<1x32x384x672xf16, #NHWC, @DDR>

    return %20 : memref<1x32x384x672xf16, #NHWC, @DDR>


    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_0]] as %arg0: memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg1: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
    // CHECK:       [[COPY_0_INNER:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 96, 0] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_1]] as %arg0: memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg1: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
    // CHECK:       [[COPY_1_INNER:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 1] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:       [[COPY_2:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_2]] as %arg0: memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_2]] as %arg1: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
    // CHECK:       [[COPY_2_INNER:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 96, 1] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:       [[COPY_3:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_3]] as %arg0: memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_3]] as %arg1: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
    // CHECK:       [[COPY_3_INNER:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)

    // CHECK:       [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[COPY_2]], [[COPY_3]]

    // CHECK:       [[OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x32x384x672xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 0, 0, 0] [1, 32, 192, 672] [1, 1, 2, 1]
    // CHECK:       [[INPUT_4:%.+]] = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[INPUT_4]] : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs([[SUBVIEW_4]] : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>)

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 0, 1, 0] [1, 32, 192, 672] [1, 1, 2, 1]
    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>)

    // CHECK:       [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_4]], [[COPY_5]]

    // CHECK:       return [[CONCAT_1]] : memref<1x32x384x672xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @NotFuseWhenMoreThanOneCopyBetweenConcatView(
        %arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
         -> memref<1x40x144x256xf16, #NHWC, @DDR> {
    %input0 = VPURT.AllocDistributed -> !InputDistributed
    %input1 = VPURT.AllocDistributed -> !InputDistributed

    %0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %2 = VPUIP.NCEClusterTiling inputs(%input0 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
        VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
    }
    %3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %4 = VPUIP.NCEClusterTiling inputs(%input1 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
        VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
    }
    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x40x144x256xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 16, 144, 256] : memref<1x40x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %8 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%7 : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>

    %9 = VPUIP.SubView %6 [0, 16, 0, 0] [1, 16, 144, 256] : memref<1x40x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %10 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%9 : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>

    %11 = VPUIP.SubView %6 [0, 32, 0, 0] [1, 8, 144, 256] : memref<1x40x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %12 = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs(%11 : memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>) -> memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %13 = VPUIP.ConcatView inputs(%8, %10, %12 :
                memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>,
                memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>,
                memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)
                outputs(%6 : memref<1x40x144x256xf16, #NHWC, @DDR>) -> memref<1x40x144x256xf16, #NHWC, @DDR>

    return %13 : memref<1x40x144x256xf16, #NHWC, @DDR>

    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 72, 256]
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_0]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
    // CHECK:       [[COPY_0_INNER:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 72, 0] [1, 16, 72, 256]
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_1]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
    // CHECK:       [[COPY_1_INNER:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

    // CHECK:       [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]

    // CHECK:       [[OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x40x144x256xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 0, 0, 0] [1, 16, 144, 256]
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[SUBVIEW_4]] : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 16, 0, 0] [1, 16, 144, 256]
    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)

    // CHECK:       [[SUBVIEW_6:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 32, 0, 0] [1, 8, 144, 256]
    // CHECK:       [[COPY_6:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs([[SUBVIEW_6]] : memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)

    // CHECK:       [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_4]], [[COPY_5]], [[COPY_6]]

    // CHECK:       return [[CONCAT_1]] : memref<1x40x144x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
	1x16x72x256xf16, #NHWC, @CMX_NN, {
	mode = "SEGMENTED",
	num_tiles = [1, 1, 2, 1],
	num_clusters = 2
}>

func.func @OneCopyAfterConcatViewHasNoUser(
		%arg0: memref<1x8x144x256xf16, #NHWC, @DDR>,
        %arg1: memref<1x16x144x256xf16, #NHWC, @DDR>)
		-> memref<1x16x144x256xf16, #NHWC, @DDR> {
	%input0 = VPURT.AllocDistributed -> !InputDistributed
	%input1 = VPURT.AllocDistributed -> !InputDistributed

	%0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	%1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%2 = VPUIP.NCEClusterTiling inputs(%input0 as %arg2: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg3: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
		VPUIP.Copy inputs(%arg2 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
	}
	%3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%4 = VPUIP.NCEClusterTiling inputs(%input1 as %arg2: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg3: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
		VPUIP.Copy inputs(%arg2 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
	}
	%5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

	%7 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

	return %arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>

	// CHECK: [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
	// CHECK: [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

	// CHECK: [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	// CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 72, 256]
	// CHECK: [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_0]] as %arg2: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg3: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
	// CHECK: [[COPY_0_INNER:%.+]] = VPUIP.Copy inputs(%arg2 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

	// CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 72, 0] [1, 16, 72, 256]
	// CHECK: [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_1]] as %arg2: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg3: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
	// CHECK: [[COPY_1_INNER:%.+]] = VPUIP.Copy inputs(%arg2 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

	// CHECK: [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
	// CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>)

	// CHECK: return %arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
	1x16x72x256xf16, #NHWC, @CMX_NN, {
	mode = "SEGMENTED",
	num_tiles = [1, 1, 2, 1],
	num_clusters = 2
}>

func.func @OneCopyAfterConcatViewHasMultiUser(
		%arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
		-> (memref<1x16x144x256xf16, #NHWC, @DDR>, memref<1x16x144x256xf16, #NHWC, @CMX_NN>) {
	%input0 = VPURT.AllocDistributed -> !InputDistributed
	%input1 = VPURT.AllocDistributed -> !InputDistributed

	%0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	%1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%2 = VPUIP.NCEClusterTiling inputs(%input0 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
		VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
	}
	%3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%4 = VPUIP.NCEClusterTiling inputs(%input1 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
		VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
	}
	%5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	%7 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%6 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

	%8 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @CMX_NN>
    %9 = VPUIP.Copy inputs(%7 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%8 : memref<1x16x144x256xf16, #NHWC, @CMX_NN>) -> memref<1x16x144x256xf16, #NHWC, @CMX_NN>

	return %7, %9 : memref<1x16x144x256xf16, #NHWC, @DDR>, memref<1x16x144x256xf16, #NHWC, @CMX_NN>

	// CHECK: [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
	// CHECK: [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

	// CHECK: [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	// CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 72, 256]
	// CHECK: [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_0]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
	// CHECK: [[COPY_0_INNER:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

	// CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 72, 0] [1, 16, 72, 256]
	// CHECK: [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_1]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
	// CHECK: [[COPY_1_INNER:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

	// CHECK: [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]

    // CHECK: [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	// CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[OUTPUT_BUFF_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>)

    // CHECK: [[OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @CMX_NN>
	// CHECK: [[COPY_5:%.+]] = VPUIP.Copy inputs([[COPY_4]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[OUTPUT_BUFF_1]] : memref<1x16x144x256xf16, #NHWC, @CMX_NN>)

	// CHECK: return [[COPY_4]], [[COPY_5]] : memref<1x16x144x256xf16, #NHWC, @DDR>, memref<1x16x144x256xf16, #NHWC, @CMX_NN>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @AvoidConcatExtraChannelToReduceDataMovement(
        %arg0: memref<1x32x360x640xf16, #NHWC, @DDR>,
        %arg1: memref<1x1x90x640xf16, #NHWC, @DDR>)
         -> memref<1x1x90x640xf16, #NHWC, @DDR>{
    %cst_0= const.Declare memref<16x32x1x1xf16, #NHWC> = dense<1.0> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg2: memref<1x32x30x640xf16, #NHWC>) outputs(%1 as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x32x30x640xf16, #NHWC>) outputs(%arg3 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x32x30x640xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<16x32x1x1xf16, #NHWC>) outputs(%3 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x32x1x1xf16, #NHWC>) outputs(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> memref<16x32x1x1xf16, #NHWC, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %6 = VPUIP.NCEClusterTiling inputs(%cst_1 as %arg2: memref<16x1x1x4xsi32>) outputs(%5 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x1x1x4xsi32>) outputs(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    }
    %7 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>, %4 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>, %6 as %arg4: memref<16x1x1x4xsi32, @CMX_NN>) outputs(%7 as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 11628 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<16x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x16x30x640xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [639, 14, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [639, 29, 15], outStart = [0, 15, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }
    %9 = memref.alloc() : memref<1x16x90x640xf16, #NHWC, @DDR>
    %10 = VPUIP.SubView %9 [0, 0, 0, 0] [1, 16, 30, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %11 = VPUIP.NCEClusterTiling inputs(%8 as %arg2: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%10 as %arg3: memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, #NHWC>
    }

    %12 = VPUIP.SubView %arg0 [0, 0, 30, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    %13 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %14 = VPUIP.NCEClusterTiling inputs(%12 as %arg2: memref<1x32x30x640xf16, #NHWC>) outputs(%13 as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x32x30x640xf16, #NHWC>) outputs(%arg3 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x32x30x640xf16, #NHWC, @CMX_NN>
    }
    %15 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %16 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<16x32x1x1xf16, #NHWC>) outputs(%15 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x32x1x1xf16, #NHWC>) outputs(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> memref<16x32x1x1xf16, #NHWC, @CMX_NN>
    }
    %17 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %18 = VPUIP.NCEClusterTiling inputs(%cst_1 as %arg2: memref<16x1x1x4xsi32>) outputs(%17 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x1x1x4xsi32>) outputs(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    }
    %19 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %20 = VPUIP.NCEClusterTiling inputs(%14 as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>, %16 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>, %18 as %arg4: memref<16x1x1x4xsi32, @CMX_NN>) outputs(%19 as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 11628 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<16x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x16x30x640xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [639, 14, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [639, 29, 15], outStart = [0, 15, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }
    %21 = VPUIP.SubView %9 [0, 0, 30, 0] [1, 16, 30, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %22 = VPUIP.NCEClusterTiling inputs(%20 as %arg2: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%21 as %arg3: memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, #NHWC>
    }

    %23 = VPUIP.SubView %arg0 [0, 0, 60, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    %24 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %25 = VPUIP.NCEClusterTiling inputs(%23 as %arg2: memref<1x32x30x640xf16, #NHWC>) outputs(%24 as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x32x30x640xf16, #NHWC>) outputs(%arg3 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x32x30x640xf16, #NHWC, @CMX_NN>
    }
    %26 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %27 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<16x32x1x1xf16, #NHWC>) outputs(%26 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x32x1x1xf16, #NHWC>) outputs(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> memref<16x32x1x1xf16, #NHWC, @CMX_NN>
    }
    %28 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %29 = VPUIP.NCEClusterTiling inputs(%cst_1 as %arg2: memref<16x1x1x4xsi32>) outputs(%28 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x1x1x4xsi32>) outputs(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    }
    %30 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %31 = VPUIP.NCEClusterTiling inputs(%25 as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>, %27 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>, %29 as %arg4: memref<16x1x1x4xsi32, @CMX_NN>) outputs(%30 as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 11628 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<16x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x16x30x640xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [639, 14, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [639, 29, 15], outStart = [0, 15, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }
    %32 = VPUIP.SubView %9 [0, 0, 60, 0] [1, 16, 30, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %33 = VPUIP.NCEClusterTiling inputs(%31 as %arg2: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%32 as %arg3: memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, #NHWC>
    }

    %34 = VPUIP.ConcatView inputs(%11, %22, %33 : memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>, memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>, memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>) outputs(%9 : memref<1x16x90x640xf16, #NHWC, @DDR>) -> memref<1x16x90x640xf16, #NHWC, @DDR>
    %35 = VPUIP.SubView %34 [0, 0, 0, 0] [1, 1, 90, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x1x90x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %37 = VPUIP.Copy inputs(%35 : memref<1x1x90x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>) outputs(%arg1 : memref<1x1x90x640xf16, #NHWC, @DDR>) -> memref<1x1x90x640xf16, #NHWC, @DDR>

    return %37 : memref<1x1x90x640xf16, #NHWC, @DDR>

    // CHECK: [[FILTER:%.+]] = const.Declare memref<16x32x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // Tile idx 0:
    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    // CHECK: [[ACTIVATION_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[ACTIVATION_COPY_IN_0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg2: memref<1x32x30x640xf16, #NHWC>)
    // CHECK:                                                       outputs([[ACTIVATION_BUF_0]] as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>)

    // CHECK: [[FILTER_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[FILTER_COPY_IN_0:%.+]] = VPUIP.NCEClusterTiling inputs([[FILTER]] as %arg2: memref<16x32x1x1xf16, #NHWC>)
    // CHECK:                                                   outputs([[FILTER_BUF_0]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>)

    // CHECK: [[TABLE_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[TABLE_COPY_IN_0:%.+]] = VPUIP.NCEClusterTiling inputs([[TABLE]] as %arg2: memref<16x1x1x4xsi32>)
    // CHECK:                                                  outputs([[TABLE_BUF_0]] as %arg3: memref<16x1x1x4xsi32, @CMX_NN>)

    // CHECK: [[CONV_RESULT_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[CONV_0:%.+]] = VPUIP.NCEClusterTiling inputs([[ACTIVATION_COPY_IN_0]] as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[FILTER_COPY_IN_0]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[TABLE_COPY_IN_0]] as %arg4: memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK:                                         outputs([[CONV_RESULT_BUF_0]] as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>)

    // Tile idx 1:
    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView %arg0 [0, 0, 30, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    // CHECK: [[ACTIVATION_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[ACTIVATION_COPY_IN_1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_1]] as %arg2: memref<1x32x30x640xf16, #NHWC>)
    // CHECK:                                                       outputs([[ACTIVATION_BUF_1]] as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>)

    // CHECK: [[FILTER_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[FILTER_COPY_IN_1:%.+]] = VPUIP.NCEClusterTiling inputs([[FILTER]] as %arg2: memref<16x32x1x1xf16, #NHWC>)
    // CHECK:                                                   outputs([[FILTER_BUF_1]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>)

    // CHECK: [[TABLE_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[TABLE_COPY_IN_1:%.+]] = VPUIP.NCEClusterTiling inputs([[TABLE]] as %arg2: memref<16x1x1x4xsi32>)
    // CHECK:                                                  outputs([[TABLE_BUF_1]] as %arg3: memref<16x1x1x4xsi32, @CMX_NN>)

    // CHECK: [[CONV_RESULT_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[CONV_1:%.+]] = VPUIP.NCEClusterTiling inputs([[ACTIVATION_COPY_IN_1]] as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[FILTER_COPY_IN_1]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[TABLE_COPY_IN_1]] as %arg4: memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK:                                         outputs([[CONV_RESULT_BUF_1]] as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>)

    // Tile idx 2:
    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg0 [0, 0, 60, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    // CHECK: [[ACTIVATION_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[ACTIVATION_COPY_IN_2:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg2: memref<1x32x30x640xf16, #NHWC>)
    // CHECK:                                                       outputs([[ACTIVATION_BUF_2]] as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>)

    // CHECK: [[FILTER_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[FILTER_COPY_IN_2:%.+]] = VPUIP.NCEClusterTiling inputs([[FILTER]] as %arg2: memref<16x32x1x1xf16, #NHWC>)
    // CHECK:                                                   outputs([[FILTER_BUF_2]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>)

    // CHECK: [[TABLE_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[TABLE_COPY_IN_2:%.+]] = VPUIP.NCEClusterTiling inputs([[TABLE]] as %arg2: memref<16x1x1x4xsi32>)
    // CHECK:                                                  outputs([[TABLE_BUF_2]] as %arg3: memref<16x1x1x4xsi32, @CMX_NN>)

    // CHECK: [[CONV_RESULT_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[CONV_2:%.+]] = VPUIP.NCEClusterTiling inputs([[ACTIVATION_COPY_IN_2]] as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[FILTER_COPY_IN_2]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[TABLE_COPY_IN_2]] as %arg4: memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK:                                         outputs([[CONV_RESULT_BUF_2]] as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>)

    // Slice Conv result at channel and concat result
    // CHECK: [[OUTPUT:%.+]] = memref.alloc() : memref<1x1x90x640xf16, #NHWC, @DDR>
    // CHECK: [[CONV_0_SLICE_CHANNEL:%.+]] = VPUIP.SubView [[CONV_0]] [0, 0, 0, 0] [1, 1, 30, 640] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUTPUT_SUB_0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 1, 30, 640] : memref<1x1x90x640xf16, #NHWC, @DDR> to memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>
    // CHECK: [[OUTPUT_COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[CONV_0_SLICE_CHANNEL]] as %arg2: memref<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN>)
    // CHECK:                                                outputs([[OUTPUT_SUB_0]] as %arg3: memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>)

    // CHECK: [[CONV_1_SLICE_CHANNEL:%.+]] = VPUIP.SubView [[CONV_1]] [0, 0, 0, 0] [1, 1, 30, 640] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUTPUT_SUB_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 30, 0] [1, 1, 30, 640] : memref<1x1x90x640xf16, #NHWC, @DDR> to memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>
    // CHECK: [[OUTPUT_COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[CONV_1_SLICE_CHANNEL]] as %arg2: memref<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN>)
    // CHECK:                                                outputs([[OUTPUT_SUB_1]] as %arg3: memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>) -> memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR> {

    // CHECK: [[CONV_2_SLICE_CHANNEL:%.+]] = VPUIP.SubView [[CONV_2]] [0, 0, 0, 0] [1, 1, 30, 640] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUTPUT_SUB_2:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 60, 0] [1, 1, 30, 640] : memref<1x1x90x640xf16, #NHWC, @DDR> to memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>
    // CHECK: [[OUTPUT_COPY_2:%.+]] = VPUIP.NCEClusterTiling inputs([[CONV_2_SLICE_CHANNEL]] as %arg2: memref<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN>)
    // CHECK:                                                outputs([[OUTPUT_SUB_2]] as %arg3: memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>) -> memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR> {

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[OUTPUT_COPY_0]], [[OUTPUT_COPY_1]], [[OUTPUT_COPY_2]]
    // CHECK:                   memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>,
    // CHECK:                   memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>,
    // CHECK:                   memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>)
    // CHECK:                   outputs([[OUTPUT]] : memref<1x1x90x640xf16, #NHWC, @DDR>) -> memref<1x1x90x640xf16, #NHWC, @DDR>

    // CHECK-NOT: VPUIP.SubView
    // CHECK: [[RESULT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x90x640xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x1x90x640xf16, #NHWC, @DDR>) -> memref<1x1x90x640xf16, #NHWC, @DDR>
	// CHECK: return [[RESULT_COPY]] : memref<1x1x90x640xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x20x40xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @RemoveDDRToDDRCopyAfterConcatThroughPureView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x256x40x40xf16, #NHWC, @DDR>)
         -> (memref<1x40x256x40xf16, #NCHW, @DDR>){
    %buffer = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%concat : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    %buffer1 = memref.alloc() : memref<1x40x256x40xf16, #NCHW, @DDR>
    %copy0 = VPUIP.Copy inputs(%permuteCast : memref<1x40x256x40xf16, #NCHW, @DDR>) outputs(%buffer1 : memref<1x40x256x40xf16, #NCHW, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    return %copy0 : memref<1x40x256x40xf16, #NCHW, @DDR>

    // CHECK: [[BUFFER0:%.+]] = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW0]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW1]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs([[BUFFER0]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[CONCAT]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x256x40x40xf16, #NCHW, @DDR>
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[PERMUTECAST]] : memref<1x40x256x40xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x20x40xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @RemoveDDRToDDRCopyAfterConcatView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x256x40x40xf16, #NHWC, @DDR>)
         -> (memref<1x256x40x40xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    %buffer1 = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %copy0 = VPUIP.Copy inputs(%concat : memref<1x256x40x40xf16,  #NHWC, @DDR>) outputs(%buffer1 : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    return %copy0 : memref<1x256x40x40xf16, #NHWC, @DDR>

    // CHECK: [[BUFFER0:%.+]] = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW0]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW1]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs([[BUFFER0]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[CONCAT]] : memref<1x256x40x40xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x20x40xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @RemoveDDRToDDRCopyAfterConcatThroughPureView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x256x40x40xf16, #NHWC, @DDR>)
         -> (memref<1x40x256x40xf16, #NCHW, @DDR>){
    %buffer = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%concat : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    %buffer1 = memref.alloc() : memref<1x40x256x40xf16, #NCHW, @DDR>
    %copy0 = VPUIP.Copy inputs(%permuteCast : memref<1x40x256x40xf16, #NCHW, @DDR>) outputs(%buffer1 : memref<1x40x256x40xf16, #NCHW, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    return %copy0 : memref<1x40x256x40xf16, #NCHW, @DDR>

    // CHECK: [[BUFFER0:%.+]] = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW0]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW1]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs([[BUFFER0]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[CONCAT]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x256x40x40xf16, #NCHW, @DDR>
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[PERMUTECAST]] : memref<1x40x256x40xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @MoveConcatViewWithClusteredCopyToCMX(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: !InputDistributed)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>) outputs(%subview0 as %arg3: memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR> {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>) outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    }

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %concat = VPUIP.ConcatView inputs(%copy, %nceTilingCopy : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.NCEClusterTiling inputs(%permuteCast as %arg2: memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%bufferCMX as %arg3: memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
        %0 = VPUIP.Copy inputs(%arg2 : memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%arg3 : memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<3584x64x1x1xf16, #NHWC, @CMX_NN>
    }

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW0]] as %arg3: memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x1x64xf16, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                          outputs([[BUFFER_DDR]] as %arg3: memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                              -> memref<1x8x1x64xf16, @DDR> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    // CHECK:                                   }

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[TILING_COPY2:%.+]] = VPUIP.NCEClusterTiling inputs([[TILING_COPY1]] as %arg2: memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW1]] as %arg3: memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY2]] : !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                          !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[DISTRIBUTEDCAST:%.+]] = VPUIP.DistributedCast inputs([[PERMUTECAST]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: return [[DISTRIBUTEDCAST]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @MoveConcatViewWithClusteredCopyToCMX_DDR2DDRCopyInputsOnly(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: memref<1x8x1x64xf16, @DDR>)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg1 : memref<1x8x1x64xf16, @DDR>) outputs(%subview0 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %concat = VPUIP.ConcatView inputs(%copy1, %copy0 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy = VPUIP.NCEClusterTiling inputs(%permuteCast as %arg2: memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%bufferCMX as %arg3: memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
        %0 = VPUIP.Copy inputs(%arg2 : memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%arg3 : memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<3584x64x1x1xf16, #NHWC, @CMX_NN>
    }

    return %nceTilingCopy : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW0]] as %arg3: memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW1]] as %arg3: memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                          !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[DISTRIBUTEDCAST:%.+]] = VPUIP.DistributedCast inputs([[PERMUTECAST]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: return [[DISTRIBUTEDCAST]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @NotMoveConcatViewWithClusteredCopyToCMXForSegmentedOutputDistribution(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: !InputDistributed)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>) outputs(%subview0 as %arg3: memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR> {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>) outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    }

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %concat = VPUIP.ConcatView inputs(%copy, %nceTilingCopy : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.NCEClusterTiling inputs(%permuteCast as %arg2: memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%bufferCMX as %arg3: memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
        %0 = VPUIP.Copy inputs(%arg2 : memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%arg3 : memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<3584x64x1x1xf16, #NHWC, @CMX_NN>
    }

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_DDR]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to
    // CHECK-SAME:                                 memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    // CHECK: [[TILING_COPY:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                          outputs([[SUBVIEW0]] as %arg3: memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                              -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    // CHECK:                                   }

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_DDR]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to
    // CHECK-SAME:                                 memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                      outputs([[SUBVIEW1]] : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY]], [[TILING_COPY]] : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>,
    // CHECK-SAME:                                                                 memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
    // CHECK-SAME:                              outputs([[BUFFER_DDR]] : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK: [[TILING_COPY2:%.+]] = VPUIP.NCEClusterTiling inputs([[PERMUTECAST]] as %arg2: memref<3584x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                                          outputs([[BUFFER_CMX]] as %arg3: memref<3584x64x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<3584x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<3584x64x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:                                   }

    // CHECK: return [[TILING_COPY2]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed0 = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!InputDistributed1 = !VPUIP.DistributedBuffer<
    1x8x447x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @NotMoveConcatViewWithClusteredCopyToCMX_NoDDR2DDRCopyInput(
        %arg0: !InputDistributed1,
        %arg1: !InputDistributed0)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>) outputs(%subview0 as %arg3: memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR> {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>) outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    }

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x8x447x64xf16, @CMX_NN>) outputs(%subview1 as %arg3: memref<1x8x447x64xf16, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR> {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x8x447x64xf16, @CMX_NN>) outputs(%arg3 : memref<1x8x447x64xf16, @DDR>) -> memref<1x8x447x64xf16, @DDR>
    }

    %concat = VPUIP.ConcatView inputs(%nceTilingCopy1, %nceTilingCopy0 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.NCEClusterTiling inputs(%permuteCast as %arg2: memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%bufferCMX as %arg3: memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
        %0 = VPUIP.Copy inputs(%arg2 : memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%arg3 : memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<3584x64x1x1xf16, #NHWC, @CMX_NN>
    }

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_DDR]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to
    // CHECK-SAME:                                 memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                          outputs([[SUBVIEW0]] as %arg3: memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                              -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    // CHECK:                                   }

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_DDR]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to
    // CHECK-SAME:                                 memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x8x447x64xf16, @CMX_NN>)
    // CHECK-SAME:                                          outputs([[SUBVIEW1]] as %arg3: memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                              -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x447x64xf16, @CMX_NN>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x447x64xf16, @DDR>) -> memref<1x8x447x64xf16, @DDR>
    // CHECK:                                   }

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY1]], [[TILING_COPY0]] : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>,
    // CHECK-SAME:                                                                 memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
    // CHECK-SAME:                              outputs([[BUFFER_DDR]] : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK: [[TILING_COPY2:%.+]] = VPUIP.NCEClusterTiling inputs([[PERMUTECAST]] as %arg2: memref<3584x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                                          outputs([[BUFFER_CMX]] as %arg3: memref<3584x64x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<3584x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<3584x64x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:                                   }

    // CHECK: return [[TILING_COPY2]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x2x49x49xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

func.func @MoveConcatViewWithClusteredCopyToCMX_ReshapeChangesShapeRank(
        %arg0: memref<1x49x49xf16, @DDR>,
        %arg1: memref<1x49x49xf16, @DDR>)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<2x49x49xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0] [1, 49, 49] : memref<2x49x49xf16, @DDR> to memref<1x49x49xf16, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg0 : memref<1x49x49xf16, @DDR>) outputs(%subview0 : memref<1x49x49xf16, @DDR>) -> memref<1x49x49xf16, @DDR>

    %subview1 = VPUIP.SubView %buffer [1, 0, 0] [1, 49, 49] : memref<2x49x49xf16, @DDR> to memref<1x49x49xf16, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg1 : memref<1x49x49xf16, @DDR>) outputs(%subview1 : memref<1x49x49xf16, @DDR>) -> memref<1x49x49xf16, @DDR>

    %concat = VPUIP.ConcatView inputs(%copy0, %copy1 : memref<1x49x49xf16, @DDR>, memref<1x49x49xf16, @DDR>) outputs(%buffer : memref<2x49x49xf16, @DDR>) -> memref<2x49x49xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<2x49x49xf16, @DDR>) -> memref<1x2x49x49xf16, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy = VPUIP.NCEClusterTiling inputs(%reshape as %arg3: memref<1x2x49x49xf16>) outputs(%bufferCMX as %arg4: memref<1x2x49x49xf16, @CMX_NN>) -> !OutputDistributed {
        %0 = VPUIP.Copy inputs(%arg3 : memref<1x2x49x49xf16>) outputs(%arg4 : memref<1x2x49x49xf16, @CMX_NN>) -> memref<1x2x49x49xf16, @CMX_NN>
    }

    return %nceTilingCopy : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0] [1, 49, 49] : !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to
    // CHECK-SAME:                          !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x49x49xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW0]] as %arg3: memref<1x49x49xf16, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x49x49xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x49x49xf16, @CMX_NN>) -> memref<1x49x49xf16, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [1, 0, 0] [1, 49, 49] : !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to
    // CHECK-SAME:                          !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x49x49xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW1]] as %arg3: memref<1x49x49xf16, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x49x49xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x49x49xf16, @CMX_NN>) -> memref<1x49x49xf16, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                          !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:                              outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x2x49x49xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: return [[RESHAPE]] : !VPUIP.DistributedBuffer<1x2x49x49xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x8x448x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

func.func @MoveConcatViewWithClusteredCopyToCMX_NoViewLikeOps(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: !InputDistributed)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>) outputs(%subview0 as %arg3: memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR> {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>) outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    }

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %concat = VPUIP.ConcatView inputs(%copy, %nceTilingCopy : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.NCEClusterTiling inputs(%concat as %arg2: memref<1x8x448x64xf16, #NCHW, @DDR>) outputs(%bufferCMX as %arg3: memref<1x8x448x64xf16, #NCHW, @CMX_NN>) -> !OutputDistributed {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x8x448x64xf16, #NCHW, @DDR>) outputs(%arg3 : memref<1x8x448x64xf16, #NCHW, @CMX_NN>) -> memref<1x8x448x64xf16, #NCHW, @CMX_NN>
    }

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW0]] as %arg3: memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x1x64xf16, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                          outputs([[BUFFER_DDR]] as %arg3: memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                              -> memref<1x8x1x64xf16, @DDR> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    // CHECK:                                   }

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[TILING_COPY2:%.+]] = VPUIP.NCEClusterTiling inputs([[TILING_COPY1]] as %arg2: memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW1]] as %arg3: memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY2]] : !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                          !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[DISTRIBUTEDCAST:%.+]] = VPUIP.DistributedCast inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: return [[DISTRIBUTEDCAST]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]],
    memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @MoveConcatViewWithClusteredCopyToCMX_ExplicitDistibution(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: !InputDistributed)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>) outputs(%subview0 as %arg3: memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR> {
        %0 = VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>) outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    }

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %concat = VPUIP.ConcatView inputs(%copy, %nceTilingCopy : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.NCEClusterTiling inputs(%permuteCast as %arg2: memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%bufferCMX as %arg3: memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
        %0 = VPUIP.Copy inputs(%arg2 : memref<3584x64x1x1xf16, #NHWC, @DDR>) outputs(%arg3 : memref<3584x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<3584x64x1x1xf16, #NHWC, @CMX_NN>
    }

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW0]] as %arg3: memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x1x64xf16, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                          outputs([[BUFFER_DDR]] as %arg3: memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                              -> memref<1x8x1x64xf16, @DDR> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @CMX_NN>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>
    // CHECK:                                   }

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[TILING_COPY2:%.+]] = VPUIP.NCEClusterTiling inputs([[TILING_COPY1]] as %arg2: memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                                          outputs([[SUBVIEW1]] as %arg3: memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:                               VPUIP.Copy inputs(%arg2 : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:                                     outputs(%arg3 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN>
    // CHECK:                                   }

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY2]] :
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:                              outputs([[BUFFER_CMX]] :
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] :
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] :
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: return [[PERMUTECAST]] :
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x8x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// This test is used for verifying the Subview of Concat used for the followed Copy changes its
// strides attr accordingly as the Subviews input to the ClusterTiling ops are not contigous

// CHECK-LABEL: func.func @AvoidConcatExtraChannelWithStridedSubView
// CHECK-SAME:    ([[INPUT_DATA0:%.+]]: !VPUIP.DistributedBuffer<1x16x8x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[INPUT_DATA1:%.+]]: !VPUIP.DistributedBuffer<1x16x8x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[INPUT_DATA2:%.+]]: memref<1x3x16x32xf16, #NHWC, @DDR>)
func.func @AvoidConcatExtraChannelWithStridedSubView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x3x16x32xf16, #NHWC, @DDR>)
         -> (memref<1x3x16x32xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x16x16x64xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 8, 64] : memref<1x16x16x64xf16, #NHWC, @DDR> to memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x16x8x64xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg5: memref<1x16x8x64xf16, #NHWC>) -> memref<1x16x8x64xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x8x64xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x8x64xf16, #NHWC>) -> memref<1x16x8x64xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 8, 0] [1, 16, 8, 64] : memref<1x16x16x64xf16, #NHWC, @DDR> to memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: memref<1x16x8x64xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg5: memref<1x16x8x64xf16, #NHWC>) -> memref<1x16x8x64xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x8x64xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x8x64xf16, #NHWC>) -> memref<1x16x8x64xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x8x64xf16, {order = #NHWC}, @DDR>, memref<1x16x8x64xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x16x16x64xf16, #NHWC, @DDR>) -> memref<1x16x16x64xf16, #NHWC, @DDR>
    %subview2 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 3, 16, 64] : memref<1x16x16x64xf16, #NHWC, @DDR> to memref<1x3x16x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>
    %subview3 = VPUIP.SubView %subview2 [0, 0, 0, 0] [1, 3, 16, 32] [1, 1, 1, 2] : memref<1x3x16x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR> to memref<1x3x16x32xf16, {order = #NHWC, strides = [16384, 1, 1024, 32]}, @DDR>
    %copy = VPUIP.Copy inputs(%subview3 : memref<1x3x16x32xf16, {order = #NHWC, strides = [16384, 1, 1024, 32]}, @DDR>) outputs(%arg2 : memref<1x3x16x32xf16, #NHWC, @DDR>) -> memref<1x3x16x32xf16, #NHWC, @DDR>
    return %copy : memref<1x3x16x32xf16, #NHWC, @DDR>

    // CHECK:    [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x3x16x64xf16, #NHWC, @DDR>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT_DATA0]] [0, 0, 0, 0] [1, 3, 8, 64] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x16x8x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 0, 0] [1, 3, 8, 64] : memref<1x3x16x64xf16, #NHWC, @DDR> to memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>
    // CHECK:    [[NCE_CLUSTER_TILING0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[SUBVIEW0]] as %arg3: memref<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[SUBVIEW1]] as %arg4: memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>) -> memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR> {
    // CHECK:      VPUIP.Copy inputs(%arg3 : memref<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg4 : memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>) -> memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>
    // CHECK:    }
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[INPUT_DATA1]] [0, 0, 0, 0] [1, 3, 8, 64] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x16x8x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 8, 0] [1, 3, 8, 64] : memref<1x3x16x64xf16, #NHWC, @DDR> to memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>
    // CHECK:    [[NCE_CLUSTER_TILING1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[SUBVIEW2]] as %arg3: memref<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[SUBVIEW3]] as %arg4: memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>) -> memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR> {
    // CHECK:      VPUIP.Copy inputs(%arg3 : memref<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg4 : memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>) -> memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCE_CLUSTER_TILING0]], [[NCE_CLUSTER_TILING1]] :
    // CHECK-SAME:    memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>, memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>)
    // CHECK-SAME:    outputs([[BUFFER_DDR]] : memref<1x3x16x64xf16, #NHWC, @DDR>) -> memref<1x3x16x64xf16, #NHWC, @DDR>
    // CHECK:    [[SUBVIEW4:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 3, 16, 32] [1, 1, 1, 2] : memref<1x3x16x64xf16, #NHWC, @DDR> to memref<1x3x16x32xf16, {order = #NHWC, strides = [3072, 1, 192, 6]}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW4]] : memref<1x3x16x32xf16, {order = #NHWC, strides = [3072, 1, 192, 6]}, @DDR>) outputs([[INPUT_DATA2]] : memref<1x3x16x32xf16, #NHWC, @DDR>) -> memref<1x3x16x32xf16, #NHWC, @DDR>
    // CHECK:    return [[COPY]] : memref<1x3x16x32xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x112x224xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>



func.func @FuseConcatViewOpsWhen1stLevelConcatHasStrides(
        %arg0: !InputDistributed, %arg1: !InputDistributed,
        %arg2: !InputDistributed, %arg3: !InputDistributed)
         -> memref<1x32x224x224xf16, #NHWC, @DDR> {
    %alloc = memref.alloc() : memref<1x32x224x224xf16, #NHWC, @DDR>

    %0 = memref.alloc() : memref<1x16x224x224xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 112, 224] [1, 1, 2, 1] : memref<1x16x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %2 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x16x112x224xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg5: memref<1x16x112x224xf16, #NHWC>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR> {
          VPUIP.Copy inputs(%arg4 : memref<1x16x112x224xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x112x224xf16, #NHWC>) -> memref<1x16x112x224xf16, #NHWC>
    }

    %3 = VPUIP.SubView %0 [0, 0, 1, 0] [1, 16, 112, 224] [1, 1, 2, 1] : memref<1x16x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %4 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: memref<1x16x112x224xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg5: memref<1x16x112x224xf16, #NHWC>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR> {
          VPUIP.Copy inputs(%arg4 : memref<1x16x112x224xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x112x224xf16, #NHWC>) -> memref<1x16x112x224xf16, #NHWC>
    }

    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>, memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>) outputs(%0 : memref<1x16x224x224xf16, #NHWC, @DDR>) -> memref<1x16x224x224xf16, #NHWC, @DDR>
    %6 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 224, 224] : memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>
    %7 = VPUIP.Copy inputs(%5 : memref<1x16x224x224xf16, #NHWC, @DDR>) outputs(%6 : memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>) -> memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>

    %8 = memref.alloc() : memref<1x16x224x224xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %8 [0, 0, 0, 0] [1, 16, 112, 224] [1, 1, 2, 1] : memref<1x16x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %10 = VPUIP.NCEClusterTiling inputs(%arg2 as %arg4: memref<1x16x112x224xf16, #NHWC, @CMX_NN>) outputs(%9 as %arg5: memref<1x16x112x224xf16, #NHWC>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR> {
          VPUIP.Copy inputs(%arg4 : memref<1x16x112x224xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x112x224xf16, #NHWC>) -> memref<1x16x112x224xf16, #NHWC>
    }

    %11 = VPUIP.SubView %8 [0, 0, 1, 0] [1, 16, 112, 224] [1, 1, 2, 1] : memref<1x16x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %12 = VPUIP.NCEClusterTiling inputs(%arg3 as %arg4: memref<1x16x112x224xf16, #NHWC, @CMX_NN>) outputs(%11 as %arg5: memref<1x16x112x224xf16, #NHWC>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR> {
          VPUIP.Copy inputs(%arg4 : memref<1x16x112x224xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x112x224xf16, #NHWC>) -> memref<1x16x112x224xf16, #NHWC>
    }

    %13 = VPUIP.ConcatView inputs(%10, %12 : memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>, memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>) outputs(%8 : memref<1x16x224x224xf16, #NHWC, @DDR>) -> memref<1x16x224x224xf16, #NHWC, @DDR>
    %14 = VPUIP.SubView %alloc [0, 16, 0, 0] [1, 16, 224, 224] : memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>
    %15 = VPUIP.Copy inputs(%13 : memref<1x16x224x224xf16, #NHWC, @DDR>) outputs(%14 : memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>) -> memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>

    %16 = VPUIP.ConcatView inputs(%7, %15 : memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>, memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>) outputs(%alloc : memref<1x32x224x224xf16, #NHWC, @DDR>) -> memref<1x32x224x224xf16, #NHWC, @DDR>

    return %16 : memref<1x32x224x224xf16, #NHWC, @DDR>


    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x32x224x224xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 16, 112, 224] [1, 1, 2, 1] :
    // CHECK-SAME:          memref<1x32x224x224xf16, #NHWC, @DDR> to
    // CHECK-SAME:          memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg4: memref<1x16x112x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] as %arg5: memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR> {
    // CHECK:           VPUIP.Copy
    // CHECK-SAME:          inputs(%arg4 : memref<1x16x112x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg5 : memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:              -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:       }

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 1, 0] [1, 16, 112, 224] [1, 1, 2, 1] :
    // CHECK-SAME:          memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg1 as %arg4: memref<1x16x112x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_1]] as %arg5: memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR> {
    // CHECK:           VPUIP.Copy
    // CHECK-SAME:          inputs(%arg4 : memref<1x16x112x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg5 : memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:              -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK        }

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 16, 0, 0] [1, 16, 112, 224] [1, 1, 2, 1] :
    // CHECK-SAME:          memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg2 as %arg4: memref<1x16x112x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_2]] as %arg5: memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR> {
    // CHECK:           VPUIP.Copy
    // CHECK-SAME:          inputs(%arg4 : memref<1x16x112x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg5 : memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:              -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK        }

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 16, 1, 0] [1, 16, 112, 224] [1, 1, 2, 1] :
    // CHECK-SAME:          memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg3 as %arg4: memref<1x16x112x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_3]] as %arg5: memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR> {
    // CHECK:           VPUIP.Copy
    // CHECK-SAME:          inputs(%arg4 : memref<1x16x112x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg5 : memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:              -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK        }

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_0]], [[COPY_1]], [[COPY_2]], [[COPY_3]] :
    // CHECK-SAME:          memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>,
    // CHECK-SAME:          memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>,
    // CHECK-SAME:          memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>,
    // CHECK-SAME:          memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x32x224x224xf16, #NHWC, @DDR>) -> memref<1x32x224x224xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x32x224x224xf16, #NHWC, @DDR>
}
