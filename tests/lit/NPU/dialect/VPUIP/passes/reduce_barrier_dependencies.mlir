//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --reduce-barrier-dependencies %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//          Shave0_1                      Shave0_1
//            |                             |
//          Bar0                          Bar0
//        /      \                      /      \
//     DMA0_1    Shave0_2            DMA0_1    Shave0_2
//      |          |                            |
//      |         Bar2                         Bar2
//      |       /   |                        /   |
//      |    DMA0_2 Shave0_3              DMA0_2 Shave0_3
//      |       \   |                        \   |
//      |         Bar3                         Bar3
//      \          |                            |
//        \       Shave0_4                     Shave0_4
//          \     /                            /
//            Bar1                          Bar1
//             |                             |
//            Shave0_5                      Shave0_5
//
// Dependency between DMA1->Bar1 is not needed!
// because DMA2 executes on same engine after DMA1 So there is implicit dep from DMA1- > DMA2

module @DmaRedundantBar {
IE.CNNNetwork entryPoint : @main
inputsInfo : {
  DataInfo "input" : tensor<1x3x64x64xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x3x64x64xf16>
}

VPURT.SW.Runtime entryPoint: @VPU.SW::@runtime stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE }
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }


    // CHECK:   [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    // CHECK:   [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier)

    return %arg1: memref<1x3x64x64xf16, @DDR>
}
}
