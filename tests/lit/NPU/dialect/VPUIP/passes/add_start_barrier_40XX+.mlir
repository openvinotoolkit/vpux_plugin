//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-start-barrier %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDRType = memref<1x3x224x224xf16, #NCHW, @DDR>

//CHECK-LABEL: @AddStartBarrierWithConstInput
func.func @AddStartBarrierWithConstInput() -> !DDRType {
    %0 = const.Declare !DDRType = dense<1.000000e+00> : tensor<1x3x224x224xf16>
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType

    VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64} {
      %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : !DDRType) outputs(%1 : !DDRType) -> !DDRType
    }
    return %1 : !DDRType

    // CHECK:       [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SyncDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDRType = memref<1x3x224x224xf16, #NCHW, @DDR>

//CHECK-LABEL: @AddStartBarrierWithNonConstInput
func.func @AddStartBarrierWithNonConstInput() -> !DDRType {
    %0 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %2 = VPURT.DeclareBuffer <DDR> <301056> -> !DDRType

    VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : !DDRType) outputs(%1 : !DDRType) -> !DDRType
    }
    VPURT.Task attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : !DDRType) outputs(%2 : !DDRType) -> !DDRType
    }
    return %2 : !DDRType

    // CHECK:       [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SyncDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task
    // CHECK:         VPUIP.NNDMA
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDRType = memref<1x3x224x224xf16, #NCHW, @DDR>

//CHECK-LABEL: @AddStartBarrierWithTwoSyncDMA
func.func @AddStartBarrierWithTwoSyncDMA() -> !DDRType {
    %0 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %2 = VPURT.DeclareBuffer <DDR> <301056> -> !DDRType

    VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64} {
      %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : !DDRType) outputs(%1 : !DDRType) -> !DDRType
    }
    VPURT.Task attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64} {
      %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%1 : !DDRType) outputs(%2 : !DDRType) -> !DDRType
    }
    return %2 : !DDRType

    // CHECK:       [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SyncDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SyncDMA
    // CHECK:       VPURT.Task
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task
    // CHECK:         VPUIP.NNDMA
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDRType = memref<1x3x224x224xf16, #NCHW, @DDR>

//CHECK-LABEL: @NotAddStartBarrier
func.func @NotAddStartBarrier() -> !DDRType {
    %0 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %2 = VPURT.DeclareBuffer <DDR> <301056> -> !DDRType

    %b = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b : !VPURT.Barrier) attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : !DDRType) outputs(%1 : !DDRType) -> !DDRType
    }
    VPURT.Task waits(%b : !VPURT.Barrier) attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : !DDRType) outputs(%2 : !DDRType) -> !DDRType
    }
    return %2 : !DDRType

    // CHECK:       [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:     VPUIP.SyncDMA
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
}
