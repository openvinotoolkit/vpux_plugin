//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: env IE_NPU_LOG_FILTER="dump-statistics-of-task-ops" vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --dump-statistics-of-task-ops -o /dev/null %s | FileCheck %s
// REQUIRES: arch-VPUX40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e-01>

module @DumpOpsStatisticsTestTwoCompressionModes {

    func.func @CompressedWeightsAndNoUncompressedWeights(%arg0: memref<1x512x3x3x!qElemType>, %arg1: memref<1x512x3x3x!qElemType>) -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>

        %cst_1 = const.Declare memref<7600x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1.0> : tensor<7600x1x1x1xf16>
        %2 = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64, isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_1 : memref<7600x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%2 : !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        }

        %cst_2 = const.Declare memref<1408x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1> : tensor<1408x1x1x1xui8>
        %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        VPURT.Task attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64, isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_2 : memref<1408x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%3 : memref<4608x1x1x1xui8, [@CMX_NN, 0]>) -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        }
        return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
    } // func
}

// CHECK:   Input size - 4.50 KB Output size - 4.50 KB
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 2 ops
// CHECK:     VPUIP.DecompressDMAOp - 2 ops
// CHECK:       DDR2CMX - 2 ops : Size - 53.50 KB
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 2, size: 53.50 KB, compressed size: 16.21 KB, (30.32%)
// CHECK:     Compressed weights - count: 2, size: 53.50 KB, compressed size: 16.21 KB, (30.32%)
// CHECK:       F16 - count: 1, size: 49.00 KB, compressed size: 14.84 KB, (30.29%)
// CHECK:       Int8 - count: 1, size: 4.50 KB, compressed size: 1.37 KB, (30.56%)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 2, size: 16.21 KB

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e-01>

module @DumpOpsStatisticsTestF16compressionModeOnly {

    func.func @CompressedWeightsAndNoUncompressedWeights(%arg0: memref<1x512x3x3x!qElemType>, %arg1: memref<1x512x3x3x!qElemType>) -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>

        %cst_1 = const.Declare memref<7600x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1.0> : tensor<7600x1x1x1xf16>
        %2 = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64, isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_1 : memref<7600x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%2 : !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<25088x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        }

        return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
    } // func
}

// CHECK:   Input size - 4.50 KB Output size - 4.50 KB
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 1 ops
// CHECK:     VPUIP.DecompressDMAOp - 1 ops
// CHECK:       DDR2CMX - 1 ops : Size - 49.00 KB
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 1, size: 49.00 KB, compressed size: 14.84 KB, (30.29%)
// CHECK:     Compressed weights - count: 1, size: 49.00 KB, compressed size: 14.84 KB, (30.29%)
// CHECK:       F16 - count: 1, size: 49.00 KB, compressed size: 14.84 KB, (30.29%)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 1, size: 14.84 KB

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e-01>
module @DumpOpsStatisticsTestTwoM2ITasks {
  func.func @TwoM2ITasks(%arg0: memref<1x512x3x3x!qElemType>, %arg1: memref<1x512x3x3x!qElemType>) -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>

    %alloc = memref.alloc() : memref<1x768x512x1xui8, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x768x512x1xui8>
    %1 = VPUIP.Copy inputs(%alloc_0 : memref<1x768x512x1xui8>) outputs(%alloc : memref<1x768x512x1xui8, [@CMX_NN, 0]>) -> memref<1x768x512x1xui8, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<1x512x512x3xui8, [@CMX_NN, 0]>
    %2 = VPUIP.M2ITask {chroma_out_reverse_channels, do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<IL_RGB888>, scale_factor_x = 131072 : ui32, scale_factor_y = 131072 : ui32} inputs(%1 : memref<1x768x512x1xui8, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x512x512x3xui8, [@CMX_NN, 0]>) -> memref<1x512x512x3xui8, [@CMX_NN, 0]>
    %3 = VPUIP.M2ITask {chroma_out_reverse_channels, do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<IL_RGB888>, scale_factor_x = 131072 : ui32, scale_factor_y = 131072 : ui32} inputs(%1 : memref<1x768x512x1xui8, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x512x512x3xui8, [@CMX_NN, 0]>) -> memref<1x512x512x3xui8, [@CMX_NN, 0]>

    return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
  } // func
}

// CHECK:   Input size - 4.50 KB Output size - 4.50 KB
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 2 ops
// CHECK:     VPUIP.M2ITask - 2 ops
// CHECK:   Weights statistics:
// CHECK:     Total weights - count: 0, size: 0 bytes (no compression)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 0, size: 0 bytes
