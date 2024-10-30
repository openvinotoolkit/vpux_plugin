//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpuip="function-outlining='naive'" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!MemRef = memref<1x3x62x62xf16>

module @ChainCalls {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x62x62xf16>
    }

    // CHECK-NOT: func.func private @foo
    func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
        %0 = VPUIP.Copy inputs(%in: !MemRef) outputs(%out: !MemRef) -> !MemRef
        return %0 : !MemRef
    }

    // CHECK: func.func @main(
    // CHECK-SAME: {{%.+}}: memref<1x3x62x62xf16, @DDR>,
    // CHECK-SAME: [[OUT:%.+]]: memref<1x3x62x62xf16, @DDR>) -> memref<1x3x62x62xf16, @DDR>
    func.func @main(%arg0: !MemRef, %arg1: !MemRef) -> !MemRef {
        %alloc = memref.alloc() : !MemRef
        %alloc2 = memref.alloc() : !MemRef
        %0 = func.call @foo(%arg0, %alloc) : (!MemRef, !MemRef) -> !MemRef
        %1 = func.call @foo(%0, %alloc2) : (!MemRef, !MemRef) -> !MemRef
        %out = VPUIP.Copy inputs(%1: !MemRef) outputs(%arg1: !MemRef) -> !MemRef
        return %out : !MemRef

        // CHECK:      [[NET_IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[NET_OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[MAIN_ALLOC0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0:[0-9]+]]> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[MAIN_ALLOC1:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR1:[0-9]+]]> -> memref<1x3x62x62xf16, @DDR>

        // CHECK-NEXT: [[FOO_CALL0_ALLOC0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0]]> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[FOO_CALL0_ALLOC1:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR1]]> -> memref<1x3x62x62xf16, @DDR>

        // CHECK-NEXT: [[FOO_CALL1_ALLOC0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0]]> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[FOO_CALL1_ALLOC1:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR1]]> -> memref<1x3x62x62xf16, @DDR>

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[NET_IN]] {{.*}} outputs([[MAIN_ALLOC0]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[FOO_CALL0_ALLOC0]] {{.*}} outputs([[FOO_CALL0_ALLOC1]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[MAIN_ALLOC1]] {{.*}} outputs([[MAIN_ALLOC0]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[FOO_CALL1_ALLOC0]] {{.*}} outputs([[FOO_CALL1_ALLOC1]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[MAIN_ALLOC1]] {{.*}} outputs([[NET_OUT]]

        // CHECK: return [[OUT]]
    }
}
