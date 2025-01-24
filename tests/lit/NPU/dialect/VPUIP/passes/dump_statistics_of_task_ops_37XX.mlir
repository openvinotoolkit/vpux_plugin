//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: env IE_NPU_LOG_FILTER="dump-statistics-of-task-ops" vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --dump-statistics-of-task-ops -o /dev/null %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e-01>

module @DumpOpsStatisticsTest {

    func.func @NoUncompressedAndNoCompressedWeights(%arg0: memref<1x512x3x3x!qElemType>, %arg1: memref<1x512x3x3x!qElemType>) -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
        return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
    } // func
}

// CHECK:   Input size - 4.50 KB Output size - 4.50 KB
// CHECK:   DDR heap size - 0 bytes
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 0 ops
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 0, size: 0 bytes (no compression)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 0, size: 0 bytes

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e-01>

module @DumpOpsStatisticsTest {

    func.func @CompressedWeightsAndNoUncompressedWeights(%arg0: memref<1x512x3x3x!qElemType>, %arg1: memref<1x512x3x3x!qElemType>) -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>

        %cst_1 = const.Declare memref<15200x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1> : tensor<15200x1x1x1xui8>
        %2 = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        VPURT.Task attributes {isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_1 : memref<15200x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%2 : !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        }

        %cst_2 = const.Declare memref<1408x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1> : tensor<1408x1x1x1xui8>
        %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        VPURT.Task attributes {isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_2 : memref<1408x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%3 : memref<4608x1x1x1xui8, [@CMX_NN, 0]>) -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        }
        return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
    } // func
}

// CHECK:   Input size - 4.50 KB Output size - 4.50 KB
// CHECK:   DDR heap size - 0 bytes
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 2 ops
// CHECK:     VPUIP.DecompressDMAOp - 2 ops
// CHECK:       DDR2CMX - 2 ops : Size - 53.50 KB
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 2, size: 53.50 KB, compressed size: 16.21 KB, (30.32%)
// CHECK:     Compressed weights - count: 2, size: 53.50 KB, compressed size: 16.21 KB, (30.32%)
// CHECK:       Int8 - count: 2, size: 53.50 KB, compressed size: 16.21 KB, (30.32%)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 2, size: 16.21 KB

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e-01>

module @DumpOpsStatisticsTest {

    func.func @CompressedWeightsAndUncompressedWeights(%arg0: memref<1x512x3x3x!qElemType>, %arg1: memref<1x512x3x3x!qElemType>) -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>

        %cst_1 = const.Declare memref<15200x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1> : tensor<15200x1x1x1xui8>
        %2 = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        VPURT.Task attributes {isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_1 : memref<15200x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%2 : !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        }

        %cst_2 = const.Declare memref<1408x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<1> : tensor<1408x1x1x1xui8>
        %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        VPURT.Task attributes {isTrailingSWLayer = false} {
            %4 = VPUIP.DecompressDMAOp inputs(%cst_2 : memref<1408x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%3 : memref<4608x1x1x1xui8, [@CMX_NN, 0]>) -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        }
        %cst_3 = const.Declare memref<1x512x3x3x!qElemType> = dense<1> : tensor<1x512x3x3xui8>, [#const.CastElemType<!qElemType>]
        VPURT.Task attributes {isTrailingSWLayer = false} {
            %1 = VPUIP.NNDMA {set_crit = false, set_ord = true}
                inputs(%cst_3 : memref<1x512x3x3x!qElemType>)
                outputs(%0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>)
                -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
        }
        %cst_4 = const.Declare memref<1x256x3x3x!qElemType> = dense<1> : tensor<1x256x3x3xui8>, [#const.CastElemType<!qElemType>]
        return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
    } // func
}

// CHECK:  Input size - 4.50 KB Output size - 4.50 KB
// CHECK:  DDR heap size - 0 bytes
// CHECK:  VPUIP tasks statistics:
// CHECK:  VPUIP Tasks - 3 ops
// CHECK:    VPUIP.NNDMA - 1 ops : Size - 4.50 KB
// CHECK:      DDR2CMX - 1 ops : Size - 4.50 KB
// CHECK:    VPUIP.DecompressDMAOp - 2 ops
// CHECK:      DDR2CMX - 2 ops : Size - 53.50 KB
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 4, size: 60.25 KB, compressed size: 22.96 KB, (38.12%)
// CHECK:     Compressed weights - count: 2, size: 53.50 KB, compressed size: 16.21 KB, (30.32%)
// CHECK:       Int8 - count: 2, size: 53.50 KB, compressed size: 16.21 KB, (30.32%)
// CHECK:     Not compressed weights - count: 2, size: 6.75 KB
// CHECK:  Const swizzling statistics:
// CHECK:    Swizzled constants     - count: 0, size: 0 bytes
// CHECK:    Not swizzled constants - count: 3, size: 20.71 KB


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e-01>

module @DumpOpsStatisticsTest {

    func.func @NoCompressedWeightsAndUncompressedWeights(%arg0: memref<1x512x3x3x!qElemType>, %arg1: memref<1x512x3x3x!qElemType>) -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
        %cst_0 = const.Declare memref<1x512x3x3x!qElemType> = dense<1> : tensor<1x512x3x3xui8>, [#const.CastElemType<!qElemType>]
        VPURT.Task attributes {isTrailingSWLayer = false} {
            %1 = VPUIP.NNDMA {set_crit = false, set_ord = true}
                inputs(%cst_0 : memref<1x512x3x3x!qElemType>)
                outputs(%0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>)
                -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
        }
        %cst_1 = const.Declare memref<1x256x3x3x!qElemType> = dense<1> : tensor<1x256x3x3xui8>, [#const.CastElemType<!qElemType>]
        return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
    } // func
}

// CHECK:   Input size - 4.50 KB Output size - 4.50 KB
// CHECK:   DDR heap size - 0 bytes
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 1 ops
// CHECK:     VPUIP.NNDMA - 1 ops : Size - 4.50 KB
// CHECK:       DDR2CMX - 1 ops : Size - 4.50 KB
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 2, size: 6.75 KB (no compression)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 1, size: 4.50 KB

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module @DumpOpsStatisticsTest {

    func.func @NNDMAWithStrideOutput(%arg0: memref<1x15460x21xf16, [@CMX_NN, 0]>, %arg1: memref<1x15460x21xf16, {order = #CHW, strides = [989440, 64, 1]}, @DDR>
                                ) -> memref<1x15460x21xf16, {order = #CHW, strides = [989440, 64, 1]}, @DDR> {
        VPURT.Task attributes {isTrailingSWLayer = false} {
            %1 = VPUIP.NNDMA {set_crit = false, set_ord = true}
                inputs(%arg0 : memref<1x15460x21xf16, [@CMX_NN, 0]>)
                outputs(%arg1 : memref<1x15460x21xf16, {order = #CHW, strides = [989440, 64, 1]}, @DDR>) -> memref<1x15460x21xf16, {order = #CHW, strides = [989440, 64, 1]}, @DDR>
        }
        return %arg1 : memref<1x15460x21xf16, {order = #CHW, strides = [989440, 64, 1]}, @DDR>
    } // func
}

// CHECK:   Input size - 634.10 KB Output size - 1.88 MB
// CHECK:   DDR heap size - 0 bytes
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 1 ops
// CHECK:     VPUIP.NNDMA - 1 ops : Size - 634.10 KB
// CHECK:       CMX2DDR - 1 ops : Size - 634.10 KB
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 0, size: 0 bytes (no compression)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 0, size: 0 bytes

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @Test  {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x4x512x1xf16>
    }
    outputsInfo : {
        IE.DataInfo "mvn" : tensor<1x4x512xf16>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096,  // Size in bytes for the actSHAVE0 in the first tile.
        4096,  // Size in bytes for the actSHAVE1 in the first tile.
        4096,  // Size in bytes for the actSHAVE2 in the second tile.
        4096   // Size in bytes for the actSHAVE3 in the second tile.
    ]


// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_mvn(%input : memref<*xf16>, %output : memref<*xf16>,
    %across_channels : i64,
    %normalize: i64,
    %eps : f32
    ) attributes {
            VPU.kernel_code  = "mvn1.cpp",
            VPU.kernel_entry = "mvn1"
        }

    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_clamp(%input : memref<*xf16>, %output : memref<*xf16>, %min : f16, %max : f16)
        attributes {
            VPU.kernel_code = "activation_clamp.cpp",
            VPU.kernel_entry = "activation_clamp"
        }


    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%1: memref<1x4x512x1xf16, {order = #NCWH}>,
           %2: memref<1x4x512x1xf16, {order = #NCWH}>) -> memref<1x4x512x1xf16, {order = #NCWH}> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>
    %out_tile0_cmx0 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0] >
    %out_tile0_cmx1 = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0] >
    %out_tile0_cmx2 = VPURT.DeclareBuffer <CMX_NN> [0] <6000> -> memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0] >
    %out_tile0_cmx3 = VPURT.DeclareBuffer <CMX_NN> [0] <8000> -> memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0] >

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x4x512x1xf16, {order = #NCWH}>)
                    outputs(%in_tile0_cmx : memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>)
            -> memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_mvn            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg0: memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx0 as %arg1: memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<1x4x512x1xf16,  {order = #NCWH}, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs=[0, -1, 0.00001]} (%arg0, %arg1)
                    : memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>
                    , memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1  : !VPURT.Barrier) updates(%b2  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_mvn            // The reference to the Kernel function.
                    inputs(%out_tile0_cmx0 as %arg0: memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx1 as %arg1: memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<1x4x512x1xf16,  {order = #NCWH}, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs=[0, -1, 0.00001]} (%arg0, %arg1)
                    : memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>
                    , memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b2  : !VPURT.Barrier) updates(%b3  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_clamp            // The reference to the Kernel function.
                    inputs(%out_tile0_cmx1 as %arg0: memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx2 as %arg1: memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [0.2, 0.5]}(%arg0, %arg1)
                    : memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>
                    , memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b3 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA inputs(%out_tile0_cmx2 : memref<1x4x512x1xf16, {order = #NCWH}, [@CMX_NN, 0]>)
        outputs(%2 : memref<1x4x512x1xf16, {order = #NCWH}>) -> memref<1x4x512x1xf16, {order = #NCWH}>
    }
    return %2: memref<1x4x512x1xf16, {order = #NCWH}>

}


}

// CHECK:  Input size - 4.00 KB Output size - 4.00 KB
// CHECK:  DDR heap size - 0 bytes
// CHECK:  VPUIP tasks statistics:
// CHECK:  VPUIP Tasks - 5 ops
// CHECK:    VPUIP.NNDMA - 2 ops : Size - 8.00 KB
// CHECK:      CMX2DDR - 1 ops : Size - 4.00 KB
// CHECK:      DDR2CMX - 1 ops : Size - 4.00 KB
// CHECK:    VPUIP.SW.Kernel - 3 ops
// CHECK:      builtin_clamp - 1 ops
// CHECK:      builtin_mvn - 2 ops
// CHECK:  Barrier statistics:
// CHECK:    VPURT.ConfigureBarrierOp - 4 ops
// CHECK:  Weights statistics
// CHECK:    Total weights - count: 0, size: 0 bytes (no compression)
// CHECK:  Const swizzling statistics:
// CHECK:    Swizzled constants     - count: 0, size: 0 bytes
// CHECK:    Not swizzled constants - count: 0, size: 0 bytes

// -----

module @DumpOpsStatisticsTest {

    func.func @DDRDeclareBuffers(%arg0: memref<1x512x3x3xf16>, %arg1: memref<1x512x3x3xf16>) -> memref<1x512x3x3xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x512x3x3xf16, @DDR>
        %1 = VPURT.DeclareBuffer <DDR> <4608> -> memref<1x512x3x3xf16, @DDR>

        return %0 : memref<1x512x3x3xf16, @DDR>
    } // func
}

// CHECK:   Input size - 9.00 KB Output size - 9.00 KB
// CHECK:   DDR heap size - 13.50 KB
// CHECK:   VPUIP tasks statistics:
// CHECK:   VPUIP Tasks - 0 ops
// CHECK:   Weights statistics
// CHECK:     Total weights - count: 0, size: 0 bytes (no compression)
// CHECK:   Const swizzling statistics:
// CHECK:     Swizzled constants     - count: 0, size: 0 bytes
// CHECK:     Not swizzled constants - count: 0, size: 0 bytes
