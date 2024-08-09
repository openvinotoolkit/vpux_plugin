//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-last-copy %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @FuseLastCopyWithoutViewOp
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_0:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_1:%arg[0-9]]]: memref<1x2x4x4xf16>) -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>)
func.func @FuseLastCopyWithoutViewOp(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>, %arg2: memref<1x2x4x4xf16>)
                            -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = const.Declare memref<1x2x4x4xf16> = dense<1.000000e+00> : tensor<1x2x4x4xf16>
    %1 = memref.alloc() : memref<1x2x4x4xf16>
    %2 = memref.alloc() : memref<1x2x4x4xf16>

    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg3: memref<1x2x4x4xf16>) outputs(%1 as %arg4: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%0 as %arg3: memref<1x2x4x4xf16>) outputs(%2 as %arg4: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }

    %5 = VPUIP.Copy inputs(%3 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    %6 = VPUIP.Copy inputs(%4 : memref<1x2x4x4xf16>) outputs(%arg2 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %5, %6 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK-DAG:   [[VAR0:%.+]] = const.Declare memref<1x2x4x4xf16> = dense<1.000000e+00> : tensor<1x2x4x4xf16>

    // CHECK-NOT:   memref.alloc() : memref<1x2x4x4xf16>
    // CHECK-NOT:   memref.alloc() : memref<1x2x4x4xf16>

    // CHECK:   [[VAR1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:      inputs([[INPUT]] as {{[^:]+}}: memref<1x2x4x4xf16>) outputs([[OUTPUT_0]] as {{[^:]+}}: memref<1x2x4x4xf16>)
    // CHECK:   [[VAR2:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:      inputs([[VAR0]] as {{[^:]+}}: memref<1x2x4x4xf16>) outputs([[OUTPUT_1]] as {{[^:]+}}: memref<1x2x4x4xf16>)
    // CHECK:   return [[VAR1]], [[VAR2]] : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @NotFuseLastCopyChangesTypeMismatch
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x50x1x1xf16>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x50xf16>) -> memref<1x50xf16>
func.func @NotFuseLastCopyChangesTypeMismatch(%arg0: memref<1x50x1x1xf16>, %arg1: memref<1x50xf16>) -> memref<1x50xf16> {
    %0 = memref.alloc() : memref<1x50x1x1xf16>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg2: memref<1x50x1x1xf16>) outputs(%0 as %arg3: memref<1x50x1x1xf16>) on tile 0 -> memref<1x50x1x1xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x50x1x1xf16>, memref<1x50x1x1xf16>
        }
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x50x1x1xf16>) -> memref<1x50xf16>
    %3 = VPUIP.Copy inputs(%2 : memref<1x50xf16>) outputs(%arg1 : memref<1x50xf16>) -> memref<1x50xf16>
    return %3 : memref<1x50xf16>

    // CHECK:       memref.alloc()
    // CHECK:       VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_Sigmoid
    // CHECK:       VPUIP.GenericReshape
    // CHECK:       [[VAR0:%.+]] = VPUIP.Copy
    // CHECK:       return [[VAR0]]
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @NotFuseLastCopyChangesInputIsBlockArgument
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_0:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_1:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_2:%arg[0-9]]]: memref<1x2x4x4xf16>) -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>)
func.func @NotFuseLastCopyChangesInputIsBlockArgument(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>,
                                    %arg2: memref<1x2x4x4xf16>, %arg3: memref<1x2x4x4xf16>) ->
                                    (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg4: memref<1x2x4x4xf16>) outputs(%arg2 as %arg5: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg4, %arg5) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }
    %2 = VPUIP.Copy inputs(%1 : memref<1x2x4x4xf16>) outputs(%arg3 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %0, %1, %2 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK:       [[VAR0:%.+]] = VPUIP.Copy
    // CHECK:       [[VAR1:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_Sigmoid
    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy
    // CHECK:       return [[VAR0]], [[VAR1]], [[VAR2]]
}
