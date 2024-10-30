//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-copy-between-swkernels-and-network-io %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @AddNNDMAForInOut
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddNNDMAForInOut(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%arg1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
        async.yield %sw_kernel : memref<1x1x1x1000xf16, @DDR>
    }

    %3 = async.await %r1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[OUT_BUFF_DDR_1:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUT_BUFF_DDR_0:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_1:%.+]], [[RESULT_1:%.+]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN} {
    // CHECK:    [[NNDMA_1:%.+]] = VPUIP.NNDMA inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUT_BUFF_DDR_0]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[NNDMA_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_2:%.+]], [[RESULT_2:%.+]] = async.execute [[[TOKEN_1]]] ([[RESULT_1]] as [[INNER_INPUT_0:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64}
    // CHECK:     [[SW_KERNEL_1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[INNER_INPUT_0]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[OUT_BUFF_DDR_1]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[SW_KERNEL_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_3:%.+]], [[RESULT_3:%.+]] = async.execute [[[TOKEN_2]]] ([[RESULT_2]] as [[INNER_INPUT:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN}
    // CHECK:     [[NNDMA_2:%.+]] = VPUIP.NNDMA inputs([[INNER_INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[NNDMA_2]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[ASYNC_WAIT:%.+]] = async.await [[RESULT_3]] : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[ASYNC_WAIT]] : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @AddNNDMAForInOutTwoKernels
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddNNDMAForInOutTwoKernels(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
    %input_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %output_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %t1, %r1:2 = async.execute -> (!async.value<memref<1x1x1x1000xf16, @DDR>>, !async.value<memref<1x1x1x1000xf16, @DDR>>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%input_1 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>, %arg0 as %in_buff_1: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%output_1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>, %arg1 as %out_buff_1: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_1, %out_buff_1)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
        async.yield %sw_kernel#0, %sw_kernel#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    }

    %3 = async.await %r1#0 : !async.value<memref<1x1x1x1000xf16, @DDR>>
    %4 = async.await %r1#1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3, %4: memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_DDR_OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[NEW_DDR_OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[DDR_INPUT_1:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[DDR_OUTPUT_1:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_0:%.+]], [[RESULT_0:%.+]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN} {
    // CHECK:     [[NNDMA_0:%.+]] = VPUIP.NNDMA inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_DDR_OUTPUT_BUFF_0]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, @DDR>
    
    // CHECK:   [[TOKEN_1:%.+]], [[RESULT_1:%.+]]:2 = async.execute [[[TOKEN_0]]] ([[RESULT_0]] as [[EXECUTE_INNER_INPUT:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> (!async.value<memref<1x1x1x1000xf16, @DDR>>, !async.value<memref<1x1x1x1000xf16, @DDR>>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:     [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu inputs([[DDR_INPUT_1]] as [[SW_KERNEL_INPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[EXECUTE_INNER_INPUT]] as [[SW_KERNEL_INPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[DDR_OUTPUT_1]] as [[SW_KERNEL_OUTPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[NEW_DDR_OUTPUT_BUFF_1]] as [[SW_KERNEL_OUTPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_0]], [[SW_KERNEL_OUTPUT_0]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_1]], [[SW_KERNEL_OUTPUT_1]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[SW_KERNEL]]#0, [[SW_KERNEL]]#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_2:%.+]], [[RESULT_2:%.+]] = async.execute [[[TOKEN_1]]] ([[RESULT_1]]#1 as [[INNER_INPUT:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN} {
    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA inputs([[INNER_INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[NNDMA_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[AWAIT_0:%.+]] = async.await [[RESULT_1]]#0 : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   [[AWAIT_1:%.+]] = async.await [[RESULT_2]] : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[AWAIT_0]], [[AWAIT_1]] : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @SwKernelPrivateFuncSwKernelIn
// CHECK-SAME:     ([[ARG:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func private @SwKernelPrivateFuncSwKernelIn(%arg0: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %output_buff_alloc =  memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
             %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%output_buff_alloc as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    
        async.yield %sw_kernel : memref<1x1x1x1000xf16, @DDR>
    }

     %3 = async.await %r1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[OUTPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[TOKEN_1:%.+]], [[RESULT_1:%.+]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:       [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[ARG]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[SW_KERNEL]] : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[AWAIT:%.+]] = async.await [[RESULT_1]] : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[AWAIT]] : memref<1x1x1x1000xf16, @DDR>
}

// CHECK-LABEL: @AddNNDMAForPrivateFuncIn
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddNNDMAForPrivateFuncIn(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %call_op = func.call @SwKernelPrivateFuncSwKernelIn(%arg0) : (memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %call_op : memref<1x1x1x1000xf16, @DDR>
    }

    %3 = async.await %r1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_INPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_0:%.+]], [[RESULT_0:%.+]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN} {
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_INPUT_ALLOC]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_1:%.+]], [[RESULT_1:%.+]] = async.execute [[[TOKEN_0]]] ([[RESULT_0]] as [[PRIVATE_ARG:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:       [[PRIVATE_FUNC_CALL:%.+]] = func.call @SwKernelPrivateFuncSwKernelIn([[PRIVATE_ARG]]) : (memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[PRIVATE_FUNC_CALL]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[AWAIT:%.+]] = async.await [[RESULT_1]] : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[AWAIT]] : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @SwKernelPrivateFuncInOut
// CHECK-SAME:     ([[ARG_1:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_2:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func private @SwKernelPrivateFuncInOut(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
             %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%arg1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    
        async.yield %sw_kernel : memref<1x1x1x1000xf16, @DDR>
    }

     %3 = async.await %r1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_1:%.+]], [[RESULT_1:%.+]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:       [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[ARG_1]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[ARG_2]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[SW_KERNEL]] : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[AWAIT:%.+]] = async.await [[RESULT_1]] : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[AWAIT]] : memref<1x1x1x1000xf16, @DDR>
}

// CHECK-LABEL: @AddNNDMAForPrivateFuncInOut
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddNNDMAForPrivateFuncInOut(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %call_op = func.call @SwKernelPrivateFuncInOut(%arg0, %arg1) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %call_op : memref<1x1x1x1000xf16, @DDR>
    }

    %3 = async.await %r1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_OUTPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[NEW_INPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR> 

    // CHECK:   [[TOKEN_0:%.+]], [[RESULT_0:%.+]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN} {
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_INPUT_ALLOC]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_1:%.+]], [[RESULT_1:%.+]] = async.execute [[[TOKEN_0]]] ([[RESULT_0]] as [[PRIVATE_FUNC_ARG:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:       [[PRIVATE_FUNC_CALL:%.+]] = func.call @SwKernelPrivateFuncInOut([[PRIVATE_FUNC_ARG]], [[NEW_OUTPUT_ALLOC]]) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[PRIVATE_FUNC_CALL]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_2:%.+]], [[RESULT_2:%.+]] = async.execute [[[TOKEN_1]]] ([[RESULT_1]] as [[PRIVATE_ARG:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN} {
    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA inputs([[PRIVATE_ARG]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[NNDMA_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[AWAIT:%.+]] = async.await [[RESULT_2]] : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[AWAIT]] : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @SwKernelPrivateFuncOut
// CHECK-SAME:     ([[ARG:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func private @SwKernelPrivateFuncOut(%arg0: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %input_buff_alloc =  memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
             %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%input_buff_alloc as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%arg0 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
    
        async.yield %sw_kernel : memref<1x1x1x1000xf16, @DDR>
    }

     %3 = async.await %r1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[INPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[TOKEN_1:%.+]], [[RESULT_1:%.+]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:       [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[INPUT]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[ARG]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[SW_KERNEL]] : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[AWAIT:%.+]] = async.await [[RESULT_1]] : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[AWAIT]] : memref<1x1x1x1000xf16, @DDR>
}

// CHECK-LABEL: @AddNNDMAForPrivateFuncOut
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddNNDMAForPrivateFuncOut(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %call_op = func.call @SwKernelPrivateFuncOut(%arg1) : (memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %call_op : memref<1x1x1x1000xf16, @DDR>
    }

    %3 = async.await %r1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_OUTPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[TOKEN_1:%.+]], [[RESULT_1:%.+]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:       [[PRIVATE_FUNC_CALL:%.+]] = func.call @SwKernelPrivateFuncOut([[NEW_OUTPUT_ALLOC]]) : (memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[PRIVATE_FUNC_CALL]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_2:%.+]], [[RESULT_2:%.+]] = async.execute [[[TOKEN_1]]] ([[RESULT_1]] as [[PRIVATE_ARG:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN} {
    // CHECK:       [[NNDMA:%.+]] = VPUIP.NNDMA inputs([[PRIVATE_ARG]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[NNDMA]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[AWAIT:%.+]] = async.await [[RESULT_2]] : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[AWAIT]] : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @SwKernelPrivateFuncInOutTwoKernels
// CHECK-SAME:     ([[ARG_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_1:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func private @SwKernelPrivateFuncInOutTwoKernels(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
    %input_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %output_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %t1, %r1:2 = async.execute -> (!async.value<memref<1x1x1x1000xf16, @DDR>>, !async.value<memref<1x1x1x1000xf16, @DDR>>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%input_1 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>, %arg0 as %in_buff_1: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%output_1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>, %arg1 as %out_buff_1: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_1, %out_buff_1)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
        async.yield %sw_kernel#0, %sw_kernel#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    }

    %3 = async.await %r1#0 : !async.value<memref<1x1x1x1000xf16, @DDR>>
    %4 = async.await %r1#1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3, %4: memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>


    // CHECK:   [[INPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUTPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_0:%.+]], [[RESULT_0:%.+]]:2 = async.execute -> (!async.value<memref<1x1x1x1000xf16, @DDR>>, !async.value<memref<1x1x1x1000xf16, @DDR>>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu inputs([[INPUT]] as [[SW_KERNEL_INPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_0]] as [[SW_KERNEL_INPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] as [[SW_KERNEL_OUTPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_1]] as [[SW_KERNEL_OUTPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>)
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_0]], [[SW_KERNEL_OUTPUT_0]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_1]], [[SW_KERNEL_OUTPUT_1]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[SW_KERNEL]]#0, [[SW_KERNEL]]#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[AWAIT_0:%.+]] = async.await [[RESULT_0]]#0 : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   [[AWAIT_1:%.+]] = async.await [[RESULT_0]]#1 : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[AWAIT_0]], [[AWAIT_1]] : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
}

// CHECK-LABEL: @AddNNDMAForPrivateFuncInOutTwoKernels
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddNNDMAForPrivateFuncInOutTwoKernels(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
    %t1, %r1:2 = async.execute -> (!async.value<memref<1x1x1x1000xf16, @DDR>>, !async.value<memref<1x1x1x1000xf16, @DDR>>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %call_op:2 = func.call @SwKernelPrivateFuncInOutTwoKernels(%arg0, %arg1) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>)
        async.yield %call_op#0, %call_op#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    }

    %3 = async.await %r1#0 : !async.value<memref<1x1x1x1000xf16, @DDR>>
    %4 = async.await %r1#1 : !async.value<memref<1x1x1x1000xf16, @DDR>>

    return %3, %4: memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[NEW_INPUT_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_1:%.+]], [[RESULT_1:%.+]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN} {
    // CHECK:       [[NNDMA_0:%.+]] = VPUIP.NNDMA inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_INPUT_BUFF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_2:%.+]], [[RESULT_2:%.+]]:2 = async.execute [[[TOKEN_1]]] ([[RESULT_1]] as [[FUNC_ARG_0:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> (!async.value<memref<1x1x1x1000xf16, @DDR>>, !async.value<memref<1x1x1x1000xf16, @DDR>>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:       [[PRIVATE_FUNC:%.+]]:2 = func.call @SwKernelPrivateFuncInOutTwoKernels([[FUNC_ARG_0]], [[NEW_OUTPUT_BUFF]]) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>)
    // CHECK:       async.yield [[PRIVATE_FUNC]]#0, [[PRIVATE_FUNC]]#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[TOKEN_3:%.+]], [[RESULT_3:%.+]] = async.execute [[[TOKEN_2]]] ([[RESULT_2]]#1 as [[NNDMA_INPUT:%.+]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN} {
    // CHECK:       [[NNDMA_1:%.+]] = VPUIP.NNDMA inputs([[NNDMA_INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:       async.yield [[NNDMA_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[AWAIT_2:%.+]] = async.await [[RESULT_2]]#0 : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   [[AWAIT_3:%.+]] = async.await [[RESULT_3]] : !async.value<memref<1x1x1x1000xf16, @DDR>>
    // CHECK:   return [[AWAIT_2]], [[AWAIT_3]] : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
}
