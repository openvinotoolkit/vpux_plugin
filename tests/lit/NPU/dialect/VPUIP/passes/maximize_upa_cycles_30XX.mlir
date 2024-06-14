//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --maximize-upa-cycles %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @RecalculateUPACycle
func.func @RecalculateUPACycle(%arg0: memref<1x1x1x100xf16>, %arg1: memref<1x1x1x100xf16>) -> (memref<1x1x1x100xf16>, memref<1x1x1x100xf16>) {

    %buf0 = memref.alloc() : memref<1x1x1x100xf16>
    %buf1 = memref.alloc() : memref<1x1x1x100xf16>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x100xf16, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x1x1x100xf16>
    %buf4 = memref.alloc() : memref<1x1x1x100xf16>

    // Copy from DDR to DDR
    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 100 : i64, cycleEnd = 100 : i64} {
      %0 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x100xf16>) outputs(%buf0 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %0 : memref<1x1x1x100xf16>
    }

    // UPA task
    %t1, %r1 = async.execute [%t0] (%r0 as %0 : !async.value<memref<1x1x1x100xf16>>) -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 1 : i64, cycleBegin = 100 : i64, cycleCost = 1 : i64, cycleEnd = 101 : i64} {
      %1 = VPUIP.ReLUUPA inputs(%0 : memref<1x1x1x100xf16>) outputs(%buf1 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %1 : memref<1x1x1x100xf16>
    }
    // Copy from DDR to NNCMX
    %t2, %r2 = async.execute -> !async.value<memref<1x1x1x100xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleBegin = 100 : i64, cycleCost = 100 : i64, cycleEnd = 200 : i64} {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x100xf16>) outputs(%buf2 : memref<1x1x1x100xf16, [@CMX_NN, 0]>) -> memref<1x1x1x100xf16, [@CMX_NN, 0]>
      async.yield %2 : memref<1x1x1x100xf16, [@CMX_NN, 0]>
    }
    // Copy from NNCMX to DDR
    %t3, %r3 = async.execute [%t2] (%r2 as %0 : !async.value<memref<1x1x1x100xf16, [@CMX_NN, 0]>>)
        -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64, cycleBegin = 200 : i64, cycleCost = 100 : i64, cycleEnd = 300 : i64} {
      %3 = VPUIP.Copy inputs(%0 : memref<1x1x1x100xf16, [@CMX_NN, 0]>) outputs(%buf3 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %3 : memref<1x1x1x100xf16>
    }

    // Copy from DDR to DDR
    %t4, %r4 = async.execute [%t1] (%r1 as %0 : !async.value<memref<1x1x1x100xf16>>)  -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleBegin = 300 : i64, cycleCost = 100 : i64, cycleEnd = 400 : i64} {
      %4 = VPUIP.Copy inputs(%0 : memref<1x1x1x100xf16>) outputs(%buf4 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %4 : memref<1x1x1x100xf16>
    }
    %5 = async.await %r3 : !async.value<memref<1x1x1x100xf16>>
    %6 = async.await %r4 : !async.value<memref<1x1x1x100xf16>>
    return %5, %6 : memref<1x1x1x100xf16>, memref<1x1x1x100xf16>

    //CHECK-NOT: {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 1 : i64, cycleBegin = 100 : i64, cycleCost = 1 : i64, cycleEnd = 101 : i64}
    //CHECK:     {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 1 : i64, cycleBegin = 100 : i64, cycleCost = 200 : i64, cycleEnd = 300 : i64}
}

// -----

// CHECK-LABEL: @RecalculateUPANoDepsCycle
func.func @RecalculateUPANoDepsCycle(%arg0: memref<1x1x1x100xf16>, %arg1: memref<1x1x1x100xf16>) -> (memref<1x1x1x100xf16>, memref<1x1x1x100xf16>) {

    %buf1 = memref.alloc() : memref<1x1x1x100xf16>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x100xf16, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x1x1x100xf16>
    %buf4 = memref.alloc() : memref<1x1x1x100xf16>

    // UPA task
    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 0 : i64, cycleBegin = 100 : i64, cycleCost = 1 : i64, cycleEnd = 101 : i64} {
      %0 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x100xf16>) outputs(%buf1 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %0 : memref<1x1x1x100xf16>
    }

    // Copy from DDR to NNCMX
    %t2, %r2 = async.execute -> !async.value<memref<1x1x1x100xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleBegin = 100 : i64, cycleCost = 100 : i64, cycleEnd = 200 : i64} {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x100xf16>) outputs(%buf2 : memref<1x1x1x100xf16, [@CMX_NN, 0]>) -> memref<1x1x1x100xf16, [@CMX_NN, 0]>
      async.yield %2 : memref<1x1x1x100xf16, [@CMX_NN, 0]>
    }
    // Copy from NNCMX to DDR
    %t3, %r3 = async.execute [%t2] (%r2 as %0 : !async.value<memref<1x1x1x100xf16, [@CMX_NN, 0]>>)
        -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleBegin = 200 : i64, cycleCost = 100 : i64, cycleEnd = 300 : i64} {
      %3 = VPUIP.Copy inputs(%0 : memref<1x1x1x100xf16, [@CMX_NN, 0]>) outputs(%buf3 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %3 : memref<1x1x1x100xf16>
    }

    // Copy from DDR to DDR
    %t4, %r4 = async.execute [%t1] (%r1 as %0 : !async.value<memref<1x1x1x100xf16>>)  -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64, cycleBegin = 300 : i64, cycleCost = 100 : i64, cycleEnd = 400 : i64} {
      %4 = VPUIP.Copy inputs(%0 : memref<1x1x1x100xf16>) outputs(%buf4 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %4 : memref<1x1x1x100xf16>
    }
    %5 = async.await %r3 : !async.value<memref<1x1x1x100xf16>>
    %6 = async.await %r4 : !async.value<memref<1x1x1x100xf16>>
    return %5, %6 : memref<1x1x1x100xf16>, memref<1x1x1x100xf16>

    //CHECK-NOT: {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 0 : i64, cycleBegin = 100 : i64, cycleCost = 1 : i64, cycleEnd = 101 : i64}
    //CHECK:     {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 300 : i64, cycleEnd = 300 : i64}
}

// -----

// CHECK-LABEL: @RecalculateUPANoConsCycle
func.func @RecalculateUPANoConsCycle(%arg0: memref<1x1x1x100xf16>, %arg1: memref<1x1x1x100xf16>) -> (memref<1x1x1x100xf16>, memref<1x1x1x100xf16>) {

    %buf0 = memref.alloc() : memref<1x1x1x100xf16>
    %buf1 = memref.alloc() : memref<1x1x1x100xf16>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x100xf16, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x1x1x100xf16>
    %buf4 = memref.alloc() : memref<1x1x1x100xf16>

    // Copy from DDR to DDR
    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 100 : i64, cycleEnd = 100 : i64} {
      %0 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x100xf16>) outputs(%buf0 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %0 : memref<1x1x1x100xf16>
    }

    // UPA task
    %t1, %r1 = async.execute [%t0] (%r0 as %0 : !async.value<memref<1x1x1x100xf16>>) -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 1 : i64, cycleBegin = 100 : i64, cycleCost = 1 : i64, cycleEnd = 101 : i64} {
      %1 = VPUIP.ReLUUPA inputs(%0 : memref<1x1x1x100xf16>) outputs(%buf1 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %1 : memref<1x1x1x100xf16>
    }
    // Copy from DDR to NNCMX
    %t2, %r2 = async.execute -> !async.value<memref<1x1x1x100xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleBegin = 100 : i64, cycleCost = 100 : i64, cycleEnd = 200 : i64} {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x100xf16>) outputs(%buf2 : memref<1x1x1x100xf16, [@CMX_NN, 0]>) -> memref<1x1x1x100xf16, [@CMX_NN, 0]>
      async.yield %2 : memref<1x1x1x100xf16, [@CMX_NN, 0]>
    }
    // Copy from NNCMX to DDR
    %t3, %r3 = async.execute [%t2] (%r2 as %0 : !async.value<memref<1x1x1x100xf16, [@CMX_NN, 0]>>)
        -> !async.value<memref<1x1x1x100xf16>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64, cycleBegin = 200 : i64, cycleCost = 100 : i64, cycleEnd = 300 : i64} {
      %3 = VPUIP.Copy inputs(%0 : memref<1x1x1x100xf16, [@CMX_NN, 0]>) outputs(%buf3 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
      async.yield %3 : memref<1x1x1x100xf16>
    }

    %4 = async.await %r3 : !async.value<memref<1x1x1x100xf16>>
    %5 = async.await %r1 : !async.value<memref<1x1x1x100xf16>>
    return %4, %5 : memref<1x1x1x100xf16>, memref<1x1x1x100xf16>

    //CHECK-NOT: {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 1 : i64, cycleBegin = 100 : i64, cycleCost = 1 : i64, cycleEnd = 101 : i64}
    //CHECK:     {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 1 : i64, cycleBegin = 100 : i64, cycleCost = 200 : i64, cycleEnd = 300 : i64}
}
