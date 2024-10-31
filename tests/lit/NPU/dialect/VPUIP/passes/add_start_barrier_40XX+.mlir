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

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDRType = memref<1x3x224x224xf16, #NCHW, @DDR>

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
 func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_relu.cpp",
            VPU.kernel_entry = "activation_relu",
            VPU.task_type = @COMPUTE
        }

func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

//CHECK-LABEL: @AddStartBarrierBecauseOp
func.func @AddStartBarrierBecauseOp() -> !DDRType {
    %0 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %2 = VPURT.DeclareBuffer <DDR> <301056> -> !DDRType
    %in_ddr  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>

    %b = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b : !VPURT.Barrier) attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : !DDRType) outputs(%1 : !DDRType) -> !DDRType
    }
    VPURT.Task waits(%b : !VPURT.Barrier) attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : !DDRType) outputs(%2 : !DDRType) -> !DDRType
    }

    VPURT.Task waits(%b  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_relu
                    inputs(%in_ddr as %arg2: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr as %arg3: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    }
    return %2 : !DDRType

    // CHECK:       [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SyncDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SW.Kernel
}
