//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --move-wait-result-to-async-block-args %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @LinearCase
func.func @LinearCase(%arg0: memref<1x1x1x10xf16>, %arg1: memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x10xf16>
    %buf1 = memref.alloc() : memref<1x1x1x10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %1 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x10xf16>) outputs(%buf0 : memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16>
        async.yield %1 : memref<1x1x1x10xf16>
    }
    %1 = async.await %f1 : !async.value<memref<1x1x1x10xf16>>

    %t2, %f2 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %2 = VPUIP.ReLUUPA inputs(%1 : memref<1x1x1x10xf16>) outputs(%buf1 : memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16>
        async.yield %2 : memref<1x1x1x10xf16>
    }
    %2 = async.await %f2 : !async.value<memref<1x1x1x10xf16>>

    %t3, %f3 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %3 = VPUIP.ReLUUPA inputs(%2 : memref<1x1x1x10xf16>) outputs(%arg1 : memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16>
        async.yield %3 : memref<1x1x1x10xf16>
    }
    %3 = async.await %f3 : !async.value<memref<1x1x1x10xf16>>

    return %3 : memref<1x1x1x10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<1x1x1x10xf16>>)
    // CHECK:           VPUIP.ReLUUPA inputs([[VAL1]] : memref<1x1x1x10xf16>)
    // CHECK-NOT:   async.await

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-NOT:           [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F2]] as [[VAL2:%.+]]: !async.value<memref<1x1x1x10xf16>>)
    // CHECK:           VPUIP.ReLUUPA inputs([[VAL2]] : memref<1x1x1x10xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAL3]]
}

// CHECK-LABEL: @MultipleUsesInOneRegion
func.func @MultipleUsesInOneRegion(%arg0: memref<1x1x1x10xf16>, %arg1: memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %1 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x10xf16>) outputs(%buf0 : memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16>
        async.yield %1 : memref<1x1x1x10xf16>
    }
    %1 = async.await %f1 : !async.value<memref<1x1x1x10xf16>>

    %t2, %f2 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %2 = VPUIP.EltwiseUPA { task_type = #VPU.eltwise_type<ADD> } inputs(%1 : memref<1x1x1x10xf16>, %1 : memref<1x1x1x10xf16>) outputs(%arg1 : memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16>
        async.yield %2 : memref<1x1x1x10xf16>
    }
    %2 = async.await %f2 : !async.value<memref<1x1x1x10xf16>>

    return %2 : memref<1x1x1x10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<1x1x1x10xf16>>)
    // CHECK:           VPUIP.EltwiseUPA {task_type = #VPU.eltwise_type<ADD>} inputs([[VAL1]] : memref<1x1x1x10xf16>, [[VAL1]] : memref<1x1x1x10xf16>)

    // CHECK:       [[VAL2:%.+]] = async.await [[F2]]
    // CHECK:       return [[VAL2]]
}

// CHECK-LABEL: @UsesFromMultipleWaits
func.func @UsesFromMultipleWaits(%arg0: memref<1x1x1x10xf16>, %arg1: memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x10xf16>
    %buf1 = memref.alloc() : memref<1x1x1x10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %1 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x10xf16>) outputs(%buf0 : memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16>
        async.yield %1 : memref<1x1x1x10xf16>
    }
    %1 = async.await %f1 : !async.value<memref<1x1x1x10xf16>>

    %t2, %f2 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %2 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x10xf16>) outputs(%buf1 : memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16>
        async.yield %2 : memref<1x1x1x10xf16>
    }
    %2 = async.await %f2 : !async.value<memref<1x1x1x10xf16>>

    %t3, %f3 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %3 = VPUIP.EltwiseUPA { task_type = #VPU.eltwise_type<ADD> } inputs(%1 : memref<1x1x1x10xf16>, %2 : memref<1x1x1x10xf16>) outputs(%arg1 : memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16>
        async.yield %3 : memref<1x1x1x10xf16>
    }
    %3 = async.await %f3 : !async.value<memref<1x1x1x10xf16>>

    return %3 : memref<1x1x1x10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<1x1x1x10xf16>>,
    // CHECK-SAME:           [[F2]] as [[VAL2:%.+]]: !async.value<memref<1x1x1x10xf16>>)
    // CHECK:           VPUIP.EltwiseUPA {task_type = #VPU.eltwise_type<ADD>} inputs([[VAL1]] : memref<1x1x1x10xf16>, [[VAL2]] : memref<1x1x1x10xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAL3]]
}

// -----

// CHECK-LABEL: @TwoOutputs
func.func @TwoOutputs(%arg0: memref<1x1x1x2xf16>, %arg1: memref<1x1x1x2xf16>, %arg2: memref<1x1x1x2xf16>) -> (memref<1x1x1x2xf16>, memref<1x1x1x2xf16>) {
    %cst = const.Declare memref<1x1x1x2xf16> = dense<1.0> : tensor<1x1x1x2xf16>

    %buf1 = memref.alloc() : memref<1x1x1x2xf16>
    %buf2 = memref.alloc() : memref<1x1x1x2xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1x1x2xf16>> {
        %1 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x2xf16>) outputs(%buf1 : memref<1x1x1x2xf16>) -> memref<1x1x1x2xf16>
        async.yield %1 : memref<1x1x1x2xf16>
    }
    %1 = async.await %f1 : !async.value<memref<1x1x1x2xf16>>


    %t2, %f2 = async.execute -> !async.value<memref<1x1x1x2xf16>> {
        %2 = VPUIP.ReLUUPA inputs(%cst : memref<1x1x1x2xf16>) outputs(%buf2 : memref<1x1x1x2xf16>) -> memref<1x1x1x2xf16>
        async.yield %2 : memref<1x1x1x2xf16>
    }
    %2 = async.await %f2 : !async.value<memref<1x1x1x2xf16>>

    %t3, %f3 = async.execute -> !async.value<memref<1x1x1x2xf16>> {
        %3 = VPUIP.Copy inputs(%1 : memref<1x1x1x2xf16>) outputs(%arg1 : memref<1x1x1x2xf16>) -> memref<1x1x1x2xf16>
        async.yield %3 : memref<1x1x1x2xf16>
    }
    %3 = async.await %f3 : !async.value<memref<1x1x1x2xf16>>

    %t4, %f4 = async.execute -> !async.value<memref<1x1x1x2xf16>> {
        %4 = VPUIP.Copy inputs(%2 : memref<1x1x1x2xf16>) outputs(%arg2 : memref<1x1x1x2xf16>) -> memref<1x1x1x2xf16>
        async.yield %4 : memref<1x1x1x2xf16>
    }
    %4 = async.await %f4 : !async.value<memref<1x1x1x2xf16>>

    return %3, %4 : memref<1x1x1x2xf16>, memref<1x1x1x2xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare

    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<1x1x1x2xf16>
    // CHECK:       [[BUF2:%.+]] = memref.alloc() : memref<1x1x1x2xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK:           VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x2xf16>) outputs([[BUF1]] : memref<1x1x1x2xf16>)

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK:           VPUIP.ReLUUPA inputs([[CST]] : memref<1x1x1x2xf16>) outputs([[BUF2]] : memref<1x1x1x2xf16>)

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-NOT:           [[T2]]
    // CHECK-SAME:          [[F1]] as [[VAL1:%.+]]: !async.value<memref<1x1x1x2xf16>>
    // CHECK:           VPUIP.Copy inputs([[VAL1]] : memref<1x1x1x2xf16>) outputs(%arg1 : memref<1x1x1x2xf16>)

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-NOT:           [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-NOT:           [[T3]]
    // CHECK-SAME:          [[F2]] as [[VAL2:%.+]]: !async.value<memref<1x1x1x2xf16>>
    // CHECK:           VPUIP.Copy inputs([[VAL2]] : memref<1x1x1x2xf16>) outputs(%arg2 : memref<1x1x1x2xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       [[VAL4:%.+]] = async.await [[F4]]
    // CHECK:       return [[VAL3]], [[VAL4]]
}
