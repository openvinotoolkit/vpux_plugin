//
// Copyright (C) 2021-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --group-async-execute-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module @Test {

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @DontMergeACTAndDMA
// Do not merge 2 ACT tasks or 2 DMA task due to exclusive users or different dependencies
func.func @DontMergeACTAndDMA(%arg0: memref<16xf16>, %arg1: memref<16xf16>, %arg2: memref<16xf16>)
        -> (memref<16xf16>, memref<16xf16>) {
    %buf0 = memref.alloc() : memref<16xf16>
    %buf1 = memref.alloc() : memref<16xf16>
    %buf2 = memref.alloc() : memref<16xf16>

    %t0, %f0 = async.execute
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @SHAVE_ACT} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg0 as %arg3: memref<16xf16>) outputs(%buf0 as %arg4: memref<16xf16>) on tile 0 -> memref<16xf16>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<16xf16>, memref<16xf16>
            }
        async.yield %0 : memref<16xf16>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0: !async.value<memref<16xf16>>)
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @SHAVE_ACT} {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_Sigmoid inputs(%0 as %arg3: memref<16xf16>) outputs(%buf1 as %arg4: memref<16xf16>) on tile 0 -> memref<16xf16>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<16xf16>, memref<16xf16>
            }
        async.yield %1 : memref<16xf16>
    }

    %t2, %f2 = async.execute [%t0] (%f0 as %0: !async.value<memref<16xf16>>)
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @SHAVE_ACT} {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_Sigmoid inputs(%0 as %arg3: memref<16xf16>) outputs(%buf2 as %arg4: memref<16xf16>) on tile 0 -> memref<16xf16>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<16xf16>, memref<16xf16>
            }
        async.yield %2 : memref<16xf16>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2: !async.value<memref<16xf16>>)
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @DMA_NN} {
        %3 = VPUIP.Copy inputs(%2 : memref<16xf16>) outputs(%arg1 : memref<16xf16>) -> memref<16xf16>
        async.yield %3 : memref<16xf16>
    }

    %t4, %f4 = async.execute [%t1] (%f1 as %1: !async.value<memref<16xf16>>)
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @DMA_NN} {
      %4 = VPUIP.Copy inputs(%1 : memref<16xf16>) outputs(%arg2 : memref<16xf16>) -> memref<16xf16>
      async.yield %4 : memref<16xf16>
    }

    %3 = async.await %f3 : !async.value<memref<16xf16>>
    %4 = async.await %f4 : !async.value<memref<16xf16>>
    return %3, %4 : memref<16xf16>, memref<16xf16>

    // CHECK:       [[BUF0:%.*]] = memref.alloc() : memref<16xf16>
    // CHECK:       [[BUF1:%.*]] = memref.alloc() : memref<16xf16>
    // CHECK:       [[BUF2:%.*]] = memref.alloc() : memref<16xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK:           [[VAR0:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%arg0 as [[ARG3:%.*]]: memref<16xf16>) outputs([[BUF0]] as [[ARG4:%.*]]: memref<16xf16>)
    // CHECK:           async.yield [[VAR0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          [[F0]] as [[VAR0_0:%.*]]: !async.value<memref<16xf16>>
    // CHECK:           [[VAR1:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs([[VAR0_0]] as [[ARG3:%.*]]: memref<16xf16>) outputs([[BUF1]] as [[ARG4:%.*]]: memref<16xf16>)
    // CHECK:           async.yield [[VAR1]]

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          [[F0]] as [[VAR0_1:%.*]]: !async.value<memref<16xf16>>
    // CHECK:           [[VAR2:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs([[VAR0_1]] as [[ARG3:%.*]]: memref<16xf16>) outputs([[BUF2]] as [[ARG4:%.*]]: memref<16xf16>)
    // CHECK:           async.yield [[VAR2]]

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          [[F2]] as [[VAR0_2:%.*]]: !async.value<memref<16xf16>>
    // CHECK:           [[VAR3:%.*]] = VPUIP.Copy inputs([[VAR0_2]] : memref<16xf16>) outputs(%arg1 : memref<16xf16>)
    // CHECK:           async.yield [[VAR3]]

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          [[F1]] as [[VAR0_3:%.*]]: !async.value<memref<16xf16>>
    // CHECK:           [[VAR4:%.*]] = VPUIP.Copy inputs([[VAR0_3]] : memref<16xf16>) outputs(%arg2 : memref<16xf16>)
    // CHECK:           async.yield [[VAR4]]

    // CHECK:       [[VAR5:%.*]] = async.await [[F3]]
    // CHECK:       [[VAR6:%.*]] = async.await [[F4]]
    // CHECK:       return [[VAR5]], [[VAR6]]
}

}

// -----

module @Test {

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @MergeDMAs
func.func @MergeDMAs(%arg0: memref<16xf16>, %arg1: memref<16xf16>, %arg2: memref<16xf16>)
        -> (memref<16xf16>, memref<16xf16>) {
    %buf = memref.alloc() : memref<16xf16>

    %t0, %f0 = async.execute
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @SHAVE_ACT} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp inputs(%arg0 as %arg3: memref<16xf16>) outputs(%buf as %arg4: memref<16xf16>) on tile 0 -> memref<16xf16>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<16xf16>, memref<16xf16>
            }
        async.yield %0 : memref<16xf16>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0: !async.value<memref<16xf16>>)
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1} {
        %1 = VPUIP.Copy inputs(%0 : memref<16xf16>) outputs(%arg1 : memref<16xf16>) -> memref<16xf16>
        async.yield %1 : memref<16xf16>
    }

    %t2, %f2 = async.execute [%t0] (%f0 as %0: !async.value<memref<16xf16>>)
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1} {
        %2 = VPUIP.Copy inputs(%0 : memref<16xf16>) outputs(%arg2 : memref<16xf16>) -> memref<16xf16>
        async.yield %2 : memref<16xf16>
    }

    %1 = async.await %f1 : !async.value<memref<16xf16>>
    %2 = async.await %f2 : !async.value<memref<16xf16>>
    return %1, %2 : memref<16xf16>, memref<16xf16>

    // CHECK:       [[BUF:%.*]] = memref.alloc() : memref<16xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK:           [[VAR0:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%arg0 as [[ARG3:%.*]]: memref<16xf16>) outputs([[BUF]] as [[ARG4:%.*]]: memref<16xf16>)
    // CHECK:           async.yield [[VAR0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]]:2 = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          [[F0]] as [[VAR0_0:%.*]]: !async.value<memref<16xf16>>,
    // CHECK-SAME:          [[F0]] as [[VAR0_1:%.*]]: !async.value<memref<16xf16>>
    // CHECK:           [[VAR1:%.*]] = VPUIP.Copy inputs([[VAR0_0]] : memref<16xf16>) outputs(%arg1 : memref<16xf16>
    // CHECK:           [[VAR2:%.*]] = VPUIP.Copy inputs([[VAR0_1]] : memref<16xf16>) outputs(%arg2 : memref<16xf16>)
    // CHECK:           async.yield [[VAR1]], [[VAR2]]

    // CHECK:       [[VAR1:%.*]] = async.await [[F1]]#0
    // CHECK:       [[VAR2:%.*]] = async.await [[F1]]#1
    // CHECK:       return [[VAR1]], [[VAR2]]
}

}

// -----

module @Test {

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TaskWithExclusiveUsers
func.func @TaskWithExclusiveUsers(%arg0: memref<16xf16>, %arg1: memref<16xf16>, %arg2: memref<16xf16>)
        -> (memref<16xf16>, memref<16xf16>) {
    %buf = memref.alloc() : memref<16xf16>

    %t0, %f0 = async.execute
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @SHAVE_ACT} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg3: memref<16xf16>) outputs(%buf as %arg4: memref<16xf16>) on tile 0 -> memref<16xf16>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<16xf16>, memref<16xf16>
            }
        async.yield %0 : memref<16xf16>
    }

    %t1, %f1 = async.execute
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @SHAVE_ACT} {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg3: memref<16xf16>) outputs(%arg1 as %arg4: memref<16xf16>) on tile 0 -> memref<16xf16>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<16xf16>, memref<16xf16>
            }
        async.yield %1 : memref<16xf16>
    }

    %t2, %f2 = async.execute [%t0] (%f0 as %0: !async.value<memref<16xf16>>)
            -> !async.value<memref<16xf16>>
            attributes {VPUIP.executor = @SHAVE_ACT} {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_Sigmoid inputs(%0 as %arg3: memref<16xf16>) outputs(%arg2 as %arg4: memref<16xf16>) on tile 0 -> memref<16xf16>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<16xf16>, memref<16xf16>
            }
        async.yield %2 : memref<16xf16>
    }

    %1 = async.await %f1 : !async.value<memref<16xf16>>
    %2 = async.await %f2 : !async.value<memref<16xf16>>
    return %1, %2 : memref<16xf16>, memref<16xf16>

    // CHECK:       [[BUF:%.*]] = memref.alloc() : memref<16xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK:           [[VAR0:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:          inputs(%arg0 as %arg3: memref<16xf16>)
    // CHECK-SAME:          outputs([[BUF]] as %arg4: memref<16xf16>)
    // CHECK:           async.yield [[VAR0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]]:2 = async.execute
    // CHECK-SAME:          ([[F0]] as [[VAR1:%.*]]: !async.value<memref<16xf16>>)
    // CHECK:           [[VAR2:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:          inputs(%arg0 as %arg4: memref<16xf16>)
    // CHECK-SAME:          outputs(%arg1 as %arg5: memref<16xf16>)
    // CHECK:           [[VAR3:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:          inputs([[VAR1]] as %arg4: memref<16xf16>)
    // CHECK-SAME:          outputs(%arg2 as %arg5: memref<16xf16>)
    // CHECK:           async.yield [[VAR2]], [[VAR3]]

    // CHECK:       [[VAR4:%.*]] = async.await [[F1]]#0
    // CHECK:       [[VAR5:%.*]] = async.await [[F1]]#1

    // CHECK:       return [[VAR4]], [[VAR5]]
}

}
