//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --group-async-execute-ops %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @NoMergeUPAAndDMA
// Do not merge 2 UPA task or 2 DMAs task due to exclusie users or different dependencies
func.func @NoMergeUPAAndDMA(%arg0: memref<1x1x1x16xui8>, %arg1: memref<1x1x1x16xf16>, %arg2: memref<1x1x1x16xf16>)
        -> (memref<1x1x1x16xf16>, memref<1x1x1x16xf16>) {
    %buf0 = memref.alloc() : memref<1x1x1x16xf16>
    %buf1 = memref.alloc() : memref<1x1x1x16xf16>
    %buf2 = memref.alloc() : memref<1x1x1x16xf16>

    %t0, %f0 = async.execute
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 16} {
        %0 = VPUIP.ConvertUPA inputs(%arg0 : memref<1x1x1x16xui8>) outputs(%buf0 : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %0 : memref<1x1x1x16xf16>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0: !async.value<memref<1x1x1x16xf16>>)
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 16} {
        %1 = VPUIP.ReLUUPA inputs(%0 : memref<1x1x1x16xf16>) outputs(%buf1 : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %1 : memref<1x1x1x16xf16>
    }

    %t2, %f2 = async.execute [%t0] (%f0 as %0: !async.value<memref<1x1x1x16xf16>>)
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 16} {
        %2 = VPUIP.ReLUUPA inputs(%0 : memref<1x1x1x16xf16>) outputs(%buf2 : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %2 : memref<1x1x1x16xf16>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2: !async.value<memref<1x1x1x16xf16>>)
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1} {
        %3 = VPUIP.Copy inputs(%2 : memref<1x1x1x16xf16>) outputs(%arg1 : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %3 : memref<1x1x1x16xf16>
    }

    %t4, %f4 = async.execute [%t1] (%f1 as %1: !async.value<memref<1x1x1x16xf16>>)
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1} {
      %4 = VPUIP.Copy inputs(%1 : memref<1x1x1x16xf16>) outputs(%arg2 : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
      async.yield %4 : memref<1x1x1x16xf16>
    }

    %3 = async.await %f3 : !async.value<memref<1x1x1x16xf16>>
    %4 = async.await %f4 : !async.value<memref<1x1x1x16xf16>>
    return %3, %4 : memref<1x1x1x16xf16>, memref<1x1x1x16xf16>

    // CHECK:       [[BUF0:%.*]] = memref.alloc() : memref<1x1x1x16xf16>
    // CHECK:       [[BUF1:%.*]] = memref.alloc() : memref<1x1x1x16xf16>
    // CHECK:       [[BUF2:%.*]] = memref.alloc() : memref<1x1x1x16xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK:           [[VAR0:%.*]] = VPUIP.ConvertUPA inputs(%arg0 : memref<1x1x1x16xui8>) outputs([[BUF0]] : memref<1x1x1x16xf16>)
    // CHECK:           async.yield [[VAR0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          [[F0]] as [[VAR0_0:%.*]]: !async.value<memref<1x1x1x16xf16>>
    // CHECK:           [[VAR1:%.*]] = VPUIP.ReLUUPA inputs([[VAR0_0]] : memref<1x1x1x16xf16>) outputs([[BUF1]] : memref<1x1x1x16xf16>)
    // CHECK:           async.yield [[VAR1]]

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          [[F0]] as [[VAR0_1:%.*]]: !async.value<memref<1x1x1x16xf16>>
    // CHECK:           [[VAR2:%.*]] = VPUIP.ReLUUPA inputs([[VAR0_1]] : memref<1x1x1x16xf16>) outputs([[BUF2]] : memref<1x1x1x16xf16>)
    // CHECK:           async.yield [[VAR2]]

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          [[F2]] as [[VAR0_2:%.*]]: !async.value<memref<1x1x1x16xf16>>
    // CHECK:           [[VAR3:%.*]] = VPUIP.Copy inputs([[VAR0_2]] : memref<1x1x1x16xf16>) outputs(%arg1 : memref<1x1x1x16xf16>)
    // CHECK:           async.yield [[VAR3]]

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          [[F1]] as [[VAR0_3:%.*]]: !async.value<memref<1x1x1x16xf16>>
    // CHECK:           [[VAR4:%.*]] = VPUIP.Copy inputs([[VAR0_3]] : memref<1x1x1x16xf16>) outputs(%arg2 : memref<1x1x1x16xf16>)
    // CHECK:           async.yield [[VAR4]]

    // CHECK:       [[VAR5:%.*]] = async.await [[F3]]
    // CHECK:       [[VAR6:%.*]] = async.await [[F4]]
    // CHECK:       return [[VAR5]], [[VAR6]]
}

// -----

// CHECK-LABEL: @MergeDMAs
func.func @MergeDMAs(%arg0: memref<1x1x1x16xui8>, %arg1: memref<1x1x1x16xf16>, %arg2: memref<1x1x1x16xf16>)
        -> (memref<1x1x1x16xf16>, memref<1x1x1x16xf16>) {
    %buf = memref.alloc() : memref<1x1x1x16xf16>

    %t0, %f0 = async.execute
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 16} {
        %0 = VPUIP.ConvertUPA inputs(%arg0 : memref<1x1x1x16xui8>) outputs(%buf : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %0 : memref<1x1x1x16xf16>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0: !async.value<memref<1x1x1x16xf16>>)
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1} {
        %1 = VPUIP.Copy inputs(%0 : memref<1x1x1x16xf16>) outputs(%arg1 : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %1 : memref<1x1x1x16xf16>
    }

    %t2, %f2 = async.execute [%t0] (%f0 as %0: !async.value<memref<1x1x1x16xf16>>)
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1} {
        %2 = VPUIP.Copy inputs(%0 : memref<1x1x1x16xf16>) outputs(%arg2 : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %2 : memref<1x1x1x16xf16>
    }

    %1 = async.await %f1 : !async.value<memref<1x1x1x16xf16>>
    %2 = async.await %f2 : !async.value<memref<1x1x1x16xf16>>
    return %1, %2 : memref<1x1x1x16xf16>, memref<1x1x1x16xf16>

    // CHECK:       [[BUF:%.*]] = memref.alloc() : memref<1x1x1x16xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK:           [[VAR0:%.*]] = VPUIP.ConvertUPA inputs(%arg0 : memref<1x1x1x16xui8>) outputs([[BUF]] : memref<1x1x1x16xf16>)
    // CHECK:           async.yield [[VAR0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]]:2 = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          [[F0]] as [[VAR0_0:%.*]]: !async.value<memref<1x1x1x16xf16>>,
    // CHECK-SAME:          [[F0]] as [[VAR0_1:%.*]]: !async.value<memref<1x1x1x16xf16>>
    // CHECK:           [[VAR1:%.*]] = VPUIP.Copy inputs([[VAR0_0]] : memref<1x1x1x16xf16>) outputs(%arg1 : memref<1x1x1x16xf16>
    // CHECK:           [[VAR2:%.*]] = VPUIP.Copy inputs([[VAR0_1]] : memref<1x1x1x16xf16>) outputs(%arg2 : memref<1x1x1x16xf16>)
    // CHECK:           async.yield [[VAR1]], [[VAR2]]

    // CHECK:       [[VAR1:%.*]] = async.await [[F1]]#0
    // CHECK:       [[VAR2:%.*]] = async.await [[F1]]#1
    // CHECK:       return [[VAR1]], [[VAR2]]
}

// -----

// CHECK-LABEL: @TaskWithExclusiveUsers
func.func @TaskWithExclusiveUsers(%arg0: memref<1x1x1x16xf16>, %arg1: memref<1x1x1x16xf16>, %arg2: memref<1x1x1x16xf16>)
        -> (memref<1x1x1x16xf16>, memref<1x1x1x16xf16>) {
    %buf = memref.alloc() : memref<1x1x1x16xf16>

    %t0, %f0 = async.execute
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 16} {
        %0 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x16xf16>) outputs(%buf : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %0 : memref<1x1x1x16xf16>
    }

    %t1, %f1 = async.execute
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 16} {
        %1 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x16xf16>) outputs(%arg1 : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %1 : memref<1x1x1x16xf16>
    }

    %t2, %f2 = async.execute [%t0] (%f0 as %0: !async.value<memref<1x1x1x16xf16>>)
            -> !async.value<memref<1x1x1x16xf16>>
            attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 16} {
        %2 = VPUIP.ReLUUPA inputs(%0 : memref<1x1x1x16xf16>) outputs(%arg2 : memref<1x1x1x16xf16>) -> memref<1x1x1x16xf16>
        async.yield %2 : memref<1x1x1x16xf16>
    }

    %1 = async.await %f1 : !async.value<memref<1x1x1x16xf16>>
    %2 = async.await %f2 : !async.value<memref<1x1x1x16xf16>>
    return %1, %2 : memref<1x1x1x16xf16>, memref<1x1x1x16xf16>

    // CHECK:       [[BUF:%.*]] = memref.alloc() : memref<1x1x1x16xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK:           [[VAR0:%.*]] = VPUIP.ReLUUPA
    // CHECK-SAME:          inputs(%arg0 : memref<1x1x1x16xf16>)
    // CHECK-SAME:          outputs([[BUF]] : memref<1x1x1x16xf16>)
    // CHECK:           async.yield [[VAR0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]]:2 = async.execute
    // CHECK-SAME:          ([[F0]] as [[VAR1:%.*]]: !async.value<memref<1x1x1x16xf16>>)
    // CHECK:           [[VAR2:%.*]] = VPUIP.ReLUUPA
    // CHECK-SAME:          inputs(%arg0 : memref<1x1x1x16xf16>)
    // CHECK-SAME:          outputs(%arg1 : memref<1x1x1x16xf16>)
    // CHECK:           [[VAR3:%.*]] = VPUIP.ReLUUPA
    // CHECK-SAME:          inputs([[VAR1]] : memref<1x1x1x16xf16>)
    // CHECK-SAME:          outputs(%arg2 : memref<1x1x1x16xf16>)
    // CHECK:           async.yield [[VAR2]], [[VAR3]]

    // CHECK:       [[VAR4:%.*]] = async.await [[F1]]#0
    // CHECK:       [[VAR5:%.*]] = async.await [[F1]]#1

    // CHECK:       return [[VAR4]], [[VAR5]]
}

// -----

// CHECK-LABEL: @MergeInputDMAs
func.func @MergeInputDMAs(%arg0: memref<1x3x1x1xf16>, %arg1: memref<1x3x1x1xf16>, %arg2: memref<1x3x1x1xf16>, %arg3: memref<1x3x1x1xf16>)
        -> (memref<1x3x1x1xf16>) {
    %token_0, %results_0 = async.execute
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64} {
        %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x1x1xf16>) outputs(%arg1 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
        async.yield %0 : memref<1x3x1x1xf16>
    }

    %token_1, %results_1 = async.execute
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64} {
        %1 = VPUIP.Copy inputs(%arg0 : memref<1x3x1x1xf16>) outputs(%arg2 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
        async.yield %1 : memref<1x3x1x1xf16>
    }

    %token_2, %results_2 = async.execute [%token_0, %token_1]
            (%results_0 as %0: !async.value<memref<1x3x1x1xf16>>, %results_1 as %1: !async.value<memref<1x3x1x1xf16>>)
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 16} {
        %2 = IERT.Add inputs(%0 : memref<1x3x1x1xf16>, %1 : memref<1x3x1x1xf16>) outputs(%arg3 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
        async.yield %2 : memref<1x3x1x1xf16>
    }

    %res = async.await %results_2 : !async.value<memref<1x3x1x1xf16>>
    return %res : memref<1x3x1x1xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]]:2 = async.execute
    // CHECK:           [[VAR0:%.*]] = VPUIP.Copy
    // CHECK-SAME:          outputs(%arg1 : memref<1x3x1x1xf16>)
    // CHECK:           [[VAR1:%.*]] = VPUIP.Copy
    // CHECK-SAME:          outputs(%arg2 : memref<1x3x1x1xf16>)
    // CHECK:           async.yield [[VAR0]], [[VAR1]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          ([[F0]]#0 as [[VAR2:%.*]]: !async.value<memref<1x3x1x1xf16>>,
    // CHECK-SAME:           [[F0]]#1 as [[VAR3:%.*]]: !async.value<memref<1x3x1x1xf16>>)
    // CHECK:           [[VAR4:%.*]] = IERT.Add
    // CHECK:               inputs([[VAR2]] : memref<1x3x1x1xf16>, [[VAR3]] : memref<1x3x1x1xf16>)

    // CHECK:       [[VAR5:%.*]] = async.await [[F1]]
    // CHECK:       return [[VAR5]]
}
