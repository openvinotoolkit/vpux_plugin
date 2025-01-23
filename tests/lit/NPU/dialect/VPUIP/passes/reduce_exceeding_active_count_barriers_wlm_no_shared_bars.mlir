//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --reduce-exceeding-active-count-barriers="num-barriers=2 share-wait-and-update-barriers=0" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// Note: 'idx' added since tasks can be reordered

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {VPUIP.wlm_status = #VPUIP.wlm_status<ENABLED>} {
    module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE }
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @linearizeBarriersInParallelBranch(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

    // Graph without barriers between tasks on same FIFO.
    // Such graph could be generated by eg. disabling shareWaitAndUpdateBarriers
    // in SimplifySchedule pass.
    //
    // 0(DMA-0)    1(DMA-1)   2(SW)
    //         \         \    /
    //          \          b0
    //           \        /  \
    //            3(DMA-0)    4(DMA1)
    //                        |
    //                        b1
    //                        |
    //                        5(SW)
    //                        |
    //                        b2
    //                        |
    //                        6(DMA1)
    //                        |
    //                        b3
    //                        |
    //                        7(DMA-0)
    //
    // Linearization of tasks distributed across 3 queues with 2 barriers.
    // For WLM compilation this scenario requires linearization as
    // tasks 3 and 5 can run in parallel and cannot depend on the same
    // physical barrier. Linearization of b0 associated with parallel task (3)
    // is triggered.

    // task 0
    VPURT.Task {
            VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    // task 1
    VPURT.Task updates (%bar0:  !VPURT.Barrier) {
            VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    // task 2
    VPURT.Task updates (%bar0: !VPURT.Barrier) {
            VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
            }
    }

    // task 3
    VPURT.Task waits (%bar0:  !VPURT.Barrier) {
            VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    // task 4
    VPURT.Task waits (%bar0:  !VPURT.Barrier) updates (%bar1:  !VPURT.Barrier)  {
            VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    // task 5
    VPURT.Task waits (%bar1: !VPURT.Barrier) updates (%bar2: !VPURT.Barrier) {
            VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
            }
    }

    // task 6
    VPURT.Task waits (%bar2:  !VPURT.Barrier) updates (%bar3:  !VPURT.Barrier)  {
            VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    // task 7
    VPURT.Task waits (%bar3:  !VPURT.Barrier)  {
            VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    return %arg1: memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // 0(DMA-0)    1(DMA-1)   2(SW)
    //         \         \    /
    //          \          b0
    //           \        /
    //            3(DMA-0)
    //                    \
    //                     b1
    //                     |
    //                     4(DMA1)
    //                     |
    //                     b2
    //                     |
    //                     5(SW)
    //                     |
    //                     b3
    //                     |
    //                     6(DMA1)
    //                     |
    //                     b4
    //                     |
    //                     7(DMA-0)
    //
    // Task 0
    // CHECK: VPURT.Task
    // CHECK:   VPUIP.NNDMA {port = 0 : i64}

    // Task 1
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:   VPUIP.NNDMA {port = 1 : i64}

    // Task 2
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:   VPUIP.SW.Kernel

    // Task 3
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:   VPUIP.NNDMA {port = 0 : i64}

    // Task 4
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:   VPUIP.NNDMA {port = 1 : i64}

    // Task 5
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK:   VPUIP.SW.Kernel

    // Task 6
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
    // CHECK:   VPUIP.NNDMA {port = 1 : i64}

    // Task 7
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier)
    // CHECK:   VPUIP.NNDMA {port = 0 : i64}
}
