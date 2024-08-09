//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --assign-physical-barriers="num-barriers=4 wlm-enable=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ParallelBranches
func.func @ParallelBranches() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //       op0
    //        |
    //       bar0
    //      /    \
    //    op1    op2
    //     |      |
    //   bar1    bar2
    //     |      |
    //    op3    op4
    //     |      |
    //   bar3    bar4
    //     |      |
    //    op5    op6
    //     |      |
    //   bar5    bar6
    //     |      |
    //    op7    op8
    //      \    /
    //       bar7
    //        |
    //       op9

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) attributes {idx = 6 : i64} {
         VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) attributes {idx = 7 : i64} {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar6: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) attributes {idx = 8 : i64} {
         VPUIP.NNDMA {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar7: !VPURT.Barrier) attributes {idx = 9 : i64} {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //       op0
    //        |
    //       bar0[0]
    //      /    \
    //    op1    op2
    //     |      |
    //   bar1[1] bar2[2]
    //     |      |
    //    op3    op4
    //     |      |
    //   bar3[3] bar4[0]
    //     |      |
    //    op5    op6
    //     |      |
    //   bar5[1] bar6[2]
    //     |      |
    //    op7    op8
    //      \    /
    //       bar7[3]
    //        |
    //       op9

    // CHECK: [[BAR0:%.*]] = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK: [[BAR6:%.*]] = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    // CHECK: [[BAR7:%.*]] = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
}
