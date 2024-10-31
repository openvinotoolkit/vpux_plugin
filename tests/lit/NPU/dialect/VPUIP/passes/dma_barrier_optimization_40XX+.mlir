//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dma-barrier-optimization %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x224x224xf16, #NHWC, @DDR>

//CHECK-LABEL: @DMABarrierOptimizationSamePortAndChannel
func.func @DMABarrierOptimizationSamePortAndChannel() -> !type_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR

    %output = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output1 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output2 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output3 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output0: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output1: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output2: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output3: !type_DDR) -> !type_DDR
    }

    return %output : !type_DDR


    // CHECK-NOT:   VPURT.DeclareVirtualBarrier

    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x224x224xf16, #NHWC, @DDR>
!type_CMX = memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>

//CHECK-LABEL: @NoDMABarrierOptimizationSamePortDifferentChannel
func.func @NoDMABarrierOptimizationSamePortDifferentChannel() -> !type_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input_DDR = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR
    %buf_CMX = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !type_CMX

    %output = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output1 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output2 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output3 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input_DDR : !type_DDR) outputs(%output0: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%buf_CMX : !type_CMX) outputs(%output1: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%buf_CMX : !type_CMX) outputs(%output2: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input_DDR : !type_DDR) outputs(%output3: !type_DDR) -> !type_DDR
    }

    return %output : !type_DDR


    // CHECK:     [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BAR2]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x224x224xf16, #NHWC, @DDR>

//CHECK-LABEL: @NoDMABarrierOptimizationDifferentPortSameChannel
func.func @NoDMABarrierOptimizationDifferentPortSameChannel() -> !type_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR

    %output = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output1 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output2 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output3 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output0: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%input : !type_DDR) outputs(%output1: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output2: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%input : !type_DDR) outputs(%output3: !type_DDR) -> !type_DDR
    }

    return %output : !type_DDR


    // CHECK:     [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA {port = 1 : i64}
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA {port = 1 : i64}
    // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//          Shave0_1                      Shave0_1
//            |                             |
//          Bar0                          Bar0
//        /      \                      /      \
//     DMA0_1    Shave0_2            DMA0_1    Shave0_2
//      |          |                            |
//      |         Bar2                         Bar1
//      |       /   |                        /   |
//      |    DMA0_2 Shave0_3              DMA0_2 Shave0_3
//      |       \   |                        \   |
//      |         Bar3                         Bar2
//      \          |                            |
//        \       Shave0_4                     Shave0_4
//          \     /                            /
//            Bar1                          Bar3
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
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier)

    return %arg1: memref<1x3x64x64xf16, @DDR>
}
}
