//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --ungroup-bounded-buffers
// --canonicalize %s | FileCheck %s REQUIRES: arch-NPU40XX

// CHECK-LABEL: UngroupBoundedBuffersforConvertDMA
func.func @UngroupBoundedBuffersforConvertDMA(%arg0: memref<1x18x15xf32, @DDR>, %arg1: memref<3xsi32, @DDR>, %arg2: memref<1x18x15xf16, @DDR>, %arg3: memref<3xsi32, @DDR>)->(!VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>) {
    // CHECK-SAME: [[ARG0:%.+]]: memref<1x18x15xf32, @DDR>,
    // CHECK-SAME: [[ARG1:%.+]]: memref<3xsi32, @DDR>,
    // CHECK-SAME: [[ARG2:%.+]]: memref<1x18x15xf16, @DDR>,
    // CHECK-SAME: [[ARG3:%.+]]: memref<3xsi32, @DDR>

    %0 = VPUIP.GroupBoundedBuffer(%arg0, %arg1): memref<1x18x15xf32, @DDR>, memref<3xsi32, @DDR>->!VPUIP.BoundedBuffer<data = memref<1x18x15xf32, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>
    %1 = VPUIP.GroupBoundedBuffer(%arg2, %arg3): memref<1x18x15xf16, @DDR>, memref<3xsi32, @DDR>->!VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>
    %2 = VPUIP.ConvertDMA inputs(%0: !VPUIP.BoundedBuffer<data = memref<1x18x15xf32, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>)
                          outputs(%1: !VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>)
        ->!VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>
    return %2 : !VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>

    // CHECK: [[DATA_CONVERTDMA:%.+]] = VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x18x15xf32, @DDR>) outputs([[ARG2]] : memref<1x18x15xf16, @DDR>) -> memref<1x18x15xf16, @DDR>
    // CHECK: [[SHAPE_COPY:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<3xsi32, @DDR>) outputs([[ARG3]] : memref<3xsi32, @DDR>) -> memref<3xsi32, @DDR>
    // CHECK: [[BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[DATA_CONVERTDMA]], [[SHAPE_COPY]]) : memref<1x18x15xf16, @DDR>, memref<3xsi32, @DDR> -> !VPUIP.BoundedBuffer<data=memref<1x18x15xf16, @DDR>, dynamic_shape=memref<3xsi32, @DDR>>
    // CHECK: return [[BOUNDED_BUFFER]]
}
