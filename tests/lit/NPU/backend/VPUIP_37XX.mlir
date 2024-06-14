//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// XFAIL: *
// VPU translate expectedly failed dueto unsupported DDR memory for actshave
// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" %s | vpux-translate --export-VPUIP -o %t
// REQUIRES: arch-VPUX37XX

module @Test {

module @UsedMemory {
    IE.MemoryResource 2048 bytes of @DDR
}

IE.TileResource 1 of @NCE {
    builtin.module @UsedMemory {
        IE.MemoryResource 1048576 bytes of @CMX_NN
    }
}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf32>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf32>
    }

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Softmax(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %1 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_Softmax inputs(%arg0 as %arg2: memref<1x1x1x1000xf16>) outputs(%0 as %arg3: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16, @DDR>
            }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
        %2 = VPUIP.NNDMA inputs(%0 : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }
    return %arg1: memref<1x1x1x1000xf16>
}

}

// CHECK:   should be of CMX_NN memkind, but 'DDR'

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           1,
// CHECK:           1,
// CHECK:           1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           2.0,
// CHECK:           2000.0,
// CHECK:           2000.0,
// CHECK:           2000.0,
// CHECK:           2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "softmax",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1,
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         2000.0,
// CHECK:         2000.0,
// CHECK:         2000.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 3,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   memory_sizes: [
// CHECK:     item: "DDR",
// CHECK:     number: 2048.0
// CHECK:     item: "NN_CMX",
// CHECK:     number: 1048576.0

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         4.0,
// CHECK:         4000.0,
// CHECK:         4.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP32"
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "softmax",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         4.0,
// CHECK:         4000.0,
// CHECK:         4.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP32"
// CHECK:     }
// CHECK:   ]

// CHECK:  act_kernel_runtime

// CHECK:   task_lists:
// CHECK:     task_type: "ActKernelTask"
