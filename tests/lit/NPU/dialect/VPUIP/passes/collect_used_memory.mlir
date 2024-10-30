//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --collect-used-memory %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @EmptyFunction
module @EmptyFunction {
    // CHECK-NOT: module @UsedMemory

    // CHECK-LABEL: @main
    func.func @main() {
        return
    }

}

// -----

// CHECK-LABEL: @OneFunctionCMX
module @OneFunctionCMX {
    // CHECK:   IE.TileResource
    // CHECK:       builtin.module @UsedMemory
    // CHECK:           IE.MemoryResource 10 bytes of @CMX_NN

    // CHECK-LABEL: @main
    func.func @main() {
        // CHECK-NOT:   module @UsedMemory
        builtin.module @UsedMemory {
            IE.MemoryResource 10 bytes of @CMX_NN
        }

        return
    }

}

// -----

// CHECK-LABEL: @TwoFunction
module @TwoFunction {
    // CHECK:   IE.TileResource
    // CHECK:       builtin.module @UsedMemory
    // CHECK:           IE.MemoryResource 500 bytes of @CMX_NN

    // CHECK-LABEL: @foo1
    func.func @foo1() {
        // CHECK-NOT:   module @UsedMemory
        builtin.module @UsedMemory {
            IE.MemoryResource 500 bytes of @CMX_NN
        }

        return
    }

    // CHECK-LABEL: @foo2
    func.func @foo2() {
        // CHECK-NOT:   module @UsedMemory
        builtin.module @UsedMemory {
            IE.MemoryResource 10 bytes of @CMX_NN
        }

        return
    }

}
