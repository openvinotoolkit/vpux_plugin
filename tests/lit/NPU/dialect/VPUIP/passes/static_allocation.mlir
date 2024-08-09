//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --static-allocation="memory-space=DDR" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @LinearGraph
module @LinearGraph {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x1x1x1000xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x1x1x1000xf16>
    }

// CHECK-LABEL: @main
func.func @main(%in: memref<1x1x1x1000xf16>, %out: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %buf1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %buf2 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    %t0, %f0 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.ReLUUPA inputs(%in : memref<1x1x1x1000xf16>) outputs(%buf0 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %0 : memref<1x1x1x1000xf16, @DDR>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.ReLUUPA inputs(%0: memref<1x1x1x1000xf16, @DDR>) outputs(%buf1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %1 : memref<1x1x1x1000xf16, @DDR>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %2 = VPUIP.ReLUUPA inputs(%1: memref<1x1x1x1000xf16, @DDR>) outputs(%buf2 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %2 : memref<1x1x1x1000xf16, @DDR>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16>>
        attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %3 = VPUIP.Copy inputs(%2 : memref<1x1x1x1000xf16, @DDR>) outputs(%out : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
        async.yield %3 : memref<1x1x1x1000xf16>
    }

    %3 = async.await %f3 : !async.value<memref<1x1x1x1000xf16>>
    return %3 : memref<1x1x1x1000xf16>

    // CHECK:   builtin.module @UsedMemory
    // CHECK:       IE.MemoryResource 4096 bytes of @DDR

    // CHECK:       [[BUF0:%.*]] = VPUIP.StaticAlloc<0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[BUF1:%.*]] = VPUIP.StaticAlloc<2048> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[BUF2:%.*]] = VPUIP.StaticAlloc<0> -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:       VPUIP.ReLUUPA
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x1x1x1000xf16, @DDR>)

    // CHECK:       VPUIP.ReLUUPA
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x1x1x1000xf16, @DDR>)

    // CHECK:       VPUIP.ReLUUPA
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x1x1x1000xf16, @DDR>)

    // CHECK:       VPUIP.Copy
}

}

// -----

// CHECK-LABEL: @LinearGraphWithReservedMem
module @LinearGraphWithReservedMem {

builtin.module @ReservedMemory {
  module @CustomReservedMemory {
    IE.MemoryResource 512 bytes of @DDR
  }
}

// CHECK:  IE.MemoryResource 512 bytes of @DDR offset 0

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x1x1x1000xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x1x1x1000xf16>
    }

// CHECK-LABEL: @main
func.func @main(%in: memref<1x1x1x1000xf16>, %out: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %buf1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %buf2 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    %t0, %f0 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.ReLUUPA inputs(%in : memref<1x1x1x1000xf16>) outputs(%buf0 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %0 : memref<1x1x1x1000xf16, @DDR>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.ReLUUPA inputs(%0: memref<1x1x1x1000xf16, @DDR>) outputs(%buf1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %1 : memref<1x1x1x1000xf16, @DDR>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_UPA, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %2 = VPUIP.ReLUUPA inputs(%1: memref<1x1x1x1000xf16, @DDR>) outputs(%buf2 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %2 : memref<1x1x1x1000xf16, @DDR>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16>>
        attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %3 = VPUIP.Copy inputs(%2 : memref<1x1x1x1000xf16, @DDR>) outputs(%out : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
        async.yield %3 : memref<1x1x1x1000xf16>
    }

    %3 = async.await %f3 : !async.value<memref<1x1x1x1000xf16>>
    return %3 : memref<1x1x1x1000xf16>

    // CHECK:   builtin.module @UsedMemory
    // CHECK:       IE.MemoryResource 4608 bytes of @DDR

    // CHECK:       [[BUF0:%.*]] = VPUIP.StaticAlloc<512> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[BUF1:%.*]] = VPUIP.StaticAlloc<2560> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[BUF2:%.*]] = VPUIP.StaticAlloc<512> -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:       VPUIP.ReLUUPA
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x1x1x1000xf16, @DDR>)

    // CHECK:       VPUIP.ReLUUPA
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x1x1x1000xf16, @DDR>)

    // CHECK:       VPUIP.ReLUUPA
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x1x1x1000xf16, @DDR>)

    // CHECK:       VPUIP.Copy
}

}
