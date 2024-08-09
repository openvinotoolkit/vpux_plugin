//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-update-barrier-for-sw-kernels %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

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

// CHECK-LABEL @AddUpdateBarrierToSwKernel
func.func @AddUpdateBarrierToSwKernel(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %in_ddr  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%out_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) {
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
    return %arg1: memref<1x1x1x1000xf16, @DDR>

    // CHECK:       [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BUFF_1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[BUFF_2:%.*]] = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK-NEXT:       VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[BUFF_1]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK-NEXT:      VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[BUFF_2]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[BUFF_1]] as %arg2: memref<1x1x1x1000xf16, @DDR>) outputs([[BUFF_2]] as %arg3: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>
    // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg3) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x1x1x1000xf16, @DDR>
}

// -----

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

// CHECK-LABEL: @DontAddUpdateBarrierToSwKernel
func.func @DontAddUpdateBarrierToSwKernel(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %in_ddr  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%out_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_relu
                    inputs(%in_ddr as %in_bff: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr as %out_buff: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_bff, %out_buff)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_ddr : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }
    return %arg1: memref<1x1x1x1000xf16, @DDR>

    // CHECK:       [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:   [[BARRIER_2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BUFF_1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[BUFF_2:%.*]] = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK-NEXT:       VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[BUFF_1]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK-NEXT:      VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[BUFF_2]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[BUFF_1]] as %arg2: memref<1x1x1x1000xf16, @DDR>) outputs([[BUFF_2]] as %arg3: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg3) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:        VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:      VPUIP.NNDMA inputs([[BUFF_2]] : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x1x1x1000xf16, @DDR>
}
