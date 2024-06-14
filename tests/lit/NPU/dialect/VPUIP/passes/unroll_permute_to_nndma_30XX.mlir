//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-permute-to-nndma  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNHWCToNCHW
func.func @PermuteToDMAWithNHWCToNCHW() -> memref<1x8x16x16xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<8x256xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 512 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNCHWToNHWC
func.func @PermuteToDMAWithNCHWToNHWC() -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<128x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<16x128xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:        dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 128 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 256 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:        }
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<128x16xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<16x128xf16, [@CMX_NN, 0]>) -> memref<16x128xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAFromTranspose
func.func @PermuteToDMAFromTranspose() -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<8x32xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 32 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<32x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<8x32xf16, [@CMX_NN, 0]>) -> memref<8x32xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithLargePlaneNumber
func.func @PermuteToDMAWithLargePlaneNumber() -> memref<1x8x32x16xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x32x16xf16, [@CMX_NN, 0]>) -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    }
    return %output: memref<1x8x32x16xf16, [@CMX_NN, 0]>


    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4608> -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<8x256xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 1024 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       }
    //CHECK:       inputs([[INPUT_BUFFER0]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER0]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 1024 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAMEL       port = 0 : i64}
    //CHECK-SAME:       inputs([[INPUT_BUFFER1]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER1]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x32x16xf16, [@CMX_NN, 0]>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @ClusterPermuteDMAWithDistributedInputAndOutput
func.func @ClusterPermuteDMAWithDistributedInputAndOutput() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <2000> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map}
            inputs(%0 : !InputDistributed)
            outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed
    // CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[INPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x4x!qElemType, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <2000> -> !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[BUFF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2000> -> !VPUIP.DistributedBuffer<4x64x!qElemType, #NC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:          VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 64 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 1 : i64, srcPlaneStride = 4 : i64, dstWidth = 1 : i64, dstStride = 64 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    // CHECK-SAME:           inputs([[INPUT]] : memref<64x4x!qElemType, [@CMX_NN, 0]>) outputs([[BUFF]] : !VPUIP.DistributedBuffer<4x64x!qElemType, #NC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<4x64x!qElemType, #NC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    }
    // CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64,
    compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]]
}>

// CHECK-LABEL: @ClusterPermuteDMAWithExpandAndPermuteExplicitSEGMENTED
func.func @ClusterPermuteDMAWithExpandAndPermuteExplicitSEGMENTED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x32x32x!qElemType, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #NHWC}
              inputs(%0 : memref<1x3x32x32x!qElemType, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<3x512x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <512> -> memref<3x512x!qElemType, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0>
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]]}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<512x16x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:   numPlanes = 3 : i64, len = 512 : i64,
    //CHECK-SAME:   srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64,
    //CHECK-SAME:   dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<3x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<512x16x!qElemType, [@CMX_NN, 0]>)
    //CHECK-SAME:   -> memref<512x16x!qElemType, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:   numPlanes = 3 : i64, len = 512 : i64,
    //CHECK-SAME:   srcWidth = 512 : i64, srcStride = 1 : i64, srcPlaneStride = 1024 : i64,
    //CHECK-SAME:   dstWidth = 1 : i64, dstStride = 16 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<3x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<512x16x!qElemType, [@CMX_NN, 1]>)
    //CHECK-SAME:   -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:                             {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:                     compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}:                     compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    //CHECK-SAME{LITERAL}:                     memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}:                     memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64,
    compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    memory_shapes = [[1, 4, 114, 224], [1, 4, 115, 224]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 109, 0]]
}>

// CHECK-LABEL: @ClusterPermuteDMAWithExpandAndPermuteExplicitOVERLAPPED
func.func @ClusterPermuteDMAWithExpandAndPermuteExplicitOVERLAPPED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x224x224x!qElemType, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map}
              inputs(%0 : memref<1x3x224x224x!qElemType, @DDR>)
              outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<3x25536x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <24416> -> memref<3x25760x!qElemType, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0>
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 114, 224], [1, 4, 115, 224]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 109, 0]]}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<25536x4x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<25760x4x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:     numPlanes = 3 : i64, len = 25536 : i64,
    //CHECK-SAME:     srcWidth = 25536 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64,
    //CHECK-SAME:     dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<3x25536x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<25536x4x!qElemType, [@CMX_NN, 0]>)
    //CHECK-SAME:     -> memref<25536x4x!qElemType, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:   dma_descriptor = #VPUIP.DMADescriptorAttr<
    //CHECK-SAME:   numPlanes = 3 : i64, len = 25760 : i64,
    //CHECK-SAME:   srcWidth = 25760 : i64, srcStride = 1 : i64, srcPlaneStride = 50176 : i64,
    //CHECK-SAME:   dstWidth = 1 : i64, dstStride = 4 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<3x25760x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<25760x4x!qElemType, [@CMX_NN, 1]>)
    //CHECK-SAME:   -> memref<25760x4x!qElemType, [@CMX_NN, 1]>


    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    //CHECK-SAME:                             {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:                     compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    //CHECK-SAME{LITERAL}:                     compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    //CHECK-SAME{LITERAL}:                     memory_shapes = [[1, 4, 114, 224], [1, 4, 115, 224]],
    //CHECK-SAME{LITERAL}:                     memory_offsets = [[0, 0, 0, 0], [0, 0, 109, 0]]}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    compute_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    compute_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @ClusterPermuteDMAWithExplicitlyDistributedInputAndOutput
func.func @ClusterPermuteDMAWithExplicitlyDistributedInputAndOutput() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <2000> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map}
            inputs(%0 : !InputDistributed)
            outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed
    // CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[INPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x4x!qElemType, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <2000>
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 4, 8, 8], [1, 4, 8, 8]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:    [[BUFF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2000>
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<4x64x!qElemType, #NC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[4, 64], [4, 64]], compute_offsets = [[0, 0], [0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[4, 64], [4, 64]], memory_offsets = [[0, 0], [0, 0]]

    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:          VPUIP.PermuteDMA {
    // CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<
    // CHECK-SAME:          numPlanes = 64 : i64, len = 4 : i64,
    // CHECK-SAME:          srcWidth = 4 : i64, srcStride = 1 : i64, srcPlaneStride = 4 : i64,
    // CHECK-SAME:          dstWidth = 1 : i64, dstStride = 64 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    // CHECK-SAME:           inputs([[INPUT]] : memref<64x4x!qElemType, [@CMX_NN, 0]>)
    // CHECK-SAME:           outputs([[BUFF]] : !VPUIP.DistributedBuffer<4x64x!qElemType, #NC, @CMX_NN,
    // CHECK-SAME:                               {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                       compute_shapes = [[4, 64], [4, 64]],
    // CHECK-SAME{LITERAL}:                       compute_offsets = [[0, 0], [0, 0]],
    // CHECK-SAME{LITERAL}:                       memory_shapes = [[4, 64], [4, 64]],
    // CHECK-SAME{LITERAL}:                       memory_offsets = [[0, 0], [0, 0]]}
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<4x64x!qElemType, #NC, @CMX_NN,
    // CHECK-SAME:              {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[4, 64], [4, 64]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0], [0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[4, 64], [4, 64]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0], [0, 0]]}>
    // CHECK:    }
    // CHECK:    return [[OUTPUT]]
}
