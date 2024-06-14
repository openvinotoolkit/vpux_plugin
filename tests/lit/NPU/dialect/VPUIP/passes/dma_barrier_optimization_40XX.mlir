//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dma-barrier-optimization %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

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
