//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --async-scheduling %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

func.func @main(%arg0: memref<1x1x1x100xf16>, %arg1: memref<100xf16>) -> memref<100xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x100xf16>
    %buf1 = memref.alloc() : memref<1x1x1x100xf16>

    %0 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x100xf16>) outputs(%buf0 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
    %1 = VPUIP.ReLUUPA inputs(%0 : memref<1x1x1x100xf16>) outputs(%buf1 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x1x1x100xf16>) -> memref<100xf16>
    %3 = VPUIP.Copy inputs(%2 : memref<100xf16>) outputs(%arg1 : memref<100xf16>) -> memref<100xf16>

    return %3: memref<100xf16>

    // CHECK:       [[BUF0:%.+]] = memref.alloc() : memref<1x1x1x100xf16>
    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<1x1x1x100xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          VPUIP.executor = @SHAVE_UPA
    // CHECK:           [[VAR0:%.+]] = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x100xf16>) outputs([[BUF0]] : memref<1x1x1x100xf16>)
    // CHECK:           async.yield [[VAR0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          ([[F0]] as [[VAR0:%.+]]: !async.value<memref<1x1x1x100xf16>>)
    // CHECK-SAME:          VPUIP.executor = @SHAVE_UPA
    // CHECK:           [[VAR1:%.+]] = VPUIP.ReLUUPA inputs([[VAR0]] : memref<1x1x1x100xf16>) outputs([[BUF1]] : memref<1x1x1x100xf16>)
    // CHECK:           async.yield [[VAR1]]

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAR1:%.+]]: !async.value<memref<1x1x1x100xf16>>)
    // CHECK-SAME:          VPUIP.executor = @DMA_NN
    // CHECK:           [[VAR2:%.+]] = VPUIP.GenericReshape inputs([[VAR1]] : memref<1x1x1x100xf16>) -> memref<100xf16>
    // CHECK:           [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<100xf16>) outputs(%arg1 : memref<100xf16>)
    // CHECK:           async.yield [[VAR3]]

    // CHECK:       [[VAR3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAR3]]
}
