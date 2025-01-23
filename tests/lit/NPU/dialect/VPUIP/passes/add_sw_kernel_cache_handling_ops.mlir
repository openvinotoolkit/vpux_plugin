//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-sw-kernel-cache-handling-ops %s | FileCheck %s
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

// CHECK-LABEL: @AddCacheHandlingSwOpOneSwKernel
func.func @AddCacheHandlingSwOpOneSwKernel(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {
    %in_ddr_0  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr_0 = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr_0 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %dma_0 : memref<1x1x1x1000xf16, @DDR>
    }

    %t1, %r1 = async.execute [%t0] (%r0 as %sw_kernel_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%sw_kernel_input as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr_0 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
        async.yield %sw_kernel : memref<1x1x1x1000xf16, @DDR>
    }

     %t2, %r2 = async.execute [%t1] (%r1 as %nndma_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>  attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
        %dma_1 = VPUIP.NNDMA inputs(%nndma_input : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %3 = async.await %r2 : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>

    return %3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[IN_BUFF:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_0:%.*]], [[R_0:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:     [[NNDMA_0:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_BUFF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_1:%.*]] = async.execute [[[T_0]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0
    // CHECK:         VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:   [[T_2:%.*]], [[R_2:%.*]] = async.execute [[[T_1]]] ([[R_0]] as [[SW_KERNEL_1_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64}
    // CHECK:     [[SW_KERNEL_1:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[SW_KERNEL_1_INPUT]] as [[INPUT:%.*]]: memref<1x1x1x1000xf16, @DDR>) outputs([[OUT_BUFF]] as [[OUTPUT:%.*]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT]], [[OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[SW_KERNEL_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_3:%.*]] = async.execute [[[T_2]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0
    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:   [[T_4:%.*]], [[R_5:%.*]] = async.execute [[[T_3]]] ([[R_2]] as [[NNDMA_1_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64}
    // CHECK:     [[NNDDMA_1:%.*]] = VPUIP.NNDMA inputs([[NNDMA_1_INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[NNDDMA_1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[ASYNC_WAIT:%.*]] = async.await [[R_5]] : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>
    // CHECK:   return [[ASYNC_WAIT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
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

// CHECK-LABEL: @AddCacheHandlingSwOpTwoSwKernels
func.func @AddCacheHandlingSwOpTwoSwKernels(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {
    %in_ddr_0  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr_0 = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr_0 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %dma_0 : memref<1x1x1x1000xf16, @DDR>
    }

    %t1, %r1 = async.execute [%t0] (%r0 as %sw_kernel_0_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel_0_output = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>
        %sw_kernel_0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%sw_kernel_0_input as %input_0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%sw_kernel_0_output as %output_0: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input_0, %output_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
        async.yield %sw_kernel_0 : memref<1x1x1x1000xf16, @DDR>
    }

    %t2, %r2 = async.execute [%t1] (%r1 as %sw_kernel_1_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 2 : i64} {
        %sw_kernel_1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%sw_kernel_1_input as %input_1: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr_0 as %output_1: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input_1, %output_1)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, @DDR>
        }
        async.yield %sw_kernel_1 : memref<1x1x1x1000xf16, @DDR>
    }

     %t3, %r3 = async.execute [%t2] (%r2 as %nndma_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>  attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
        %dma_1 = VPUIP.NNDMA inputs(%nndma_input : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %3 = async.await %r3 : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>

    return %3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[IN_BUFF:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_0:%.*]], [[R_0:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:     [[NNDMA_0:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_BUFF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_1:%.*]] = async.execute [[[T_0]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0
    // CHECK:         VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:   [[T_2:%.*]], [[R_2:%.*]] = async.execute [[[T_1]]] ([[R_0]] as [[SW_KERNEL_0_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64}
    // CHECK:     [[SW_KERNEL_0_OUTPUT:%.*]] = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     [[SW_KERNEL_0:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[SW_KERNEL_0_INPUT]] as [[INPUT_0:%.*]]: memref<1x1x1x1000xf16, @DDR>) outputs([[SW_KERNEL_0_OUTPUT]] as [[OUTPUT_0:%.*]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT_0]], [[OUTPUT_0]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[SW_KERNEL_0]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_3:%.*]], [[R_3:%.*]] = async.execute [[[T_2]]] ([[R_2]] as [[SW_KERNEL_1_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 2 : i64}
    // CHECK:     [[SW_KERNEL_1:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[SW_KERNEL_1_INPUT]] as [[INPUT_1:%.*]]: memref<1x1x1x1000xf16, @DDR>) outputs([[OUT_BUFF]] as [[OUTPUT_1:%.*]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT_1]], [[OUTPUT_1]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[SW_KERNEL_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_4:%.*]] = async.execute [[[T_3]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0
    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:   [[T_5:%.*]], [[R_5:%.*]] = async.execute [[[T_4]]] ([[R_3]] as [[NNDMA_1_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64}
    // CHECK:     [[NNDDMA_1:%.*]] = VPUIP.NNDMA inputs([[NNDMA_1_INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[NNDDMA_1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[ASYNC_WAIT:%.*]] = async.await [[R_5]] : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>
    // CHECK:   return [[ASYNC_WAIT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
}


// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input0 : memref<*xf16>, %input1 : memref<*xf16>, %output0 : memref<*xf16>, %output1 : memref<*xf16>)
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

// CHECK-LABEL: @AddCacheHandlingSwOpTwoKernels
func.func @AddCacheHandlingSwOpTwoKernels(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>) {
    %in_ddr_0  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %in_ddr_1  = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr_0 = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr_1 = VPURT.DeclareBuffer <DDR> <6000> -> memref<1x1x1x1000xf16, @DDR>
    %out_cmx_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_cmx_1 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr_0 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %dma_0 : memref<1x1x1x1000xf16, @DDR>
    }

    %t1, %r1 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr_1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %dma_0 : memref<1x1x1x1000xf16, @DDR>
    }

    %t2, %r2:2 = async.execute [%t0, %t1] (%r0 as %sw_kernel_0_input : !async.value<memref<1x1x1x1000xf16, @DDR>>, %r1 as %sw_kernel_1_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> (!async.value<memref<1x1x1x1000xf16, @DDR>>, !async.value<memref<1x1x1x1000xf16, @DDR>>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>}
                    @VPU.SW::@builtin_relu
                    inputs(%sw_kernel_0_input as %in_buff_0: memref<1x1x1x1000xf16, @DDR>, %sw_kernel_1_input as %in_buff_1: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr_0 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>, %out_ddr_1 as %out_buff_1: memref<1x1x1x1000xf16, @DDR>)
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

    %t3, %r3 = async.execute [%t2] (%r2#0 as %nndma_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>  attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
        %dma_1 = VPUIP.NNDMA inputs(%nndma_input : memref<1x1x1x1000xf16, @DDR>) outputs(%out_cmx_0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %t4, %r4 = async.execute [%t2] (%r2#1 as %nndma_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>  attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
        %dma_1 = VPUIP.NNDMA inputs(%nndma_input : memref<1x1x1x1000xf16, @DDR>) outputs(%out_cmx_1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %3 = async.await %r3 : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>
    %4 = async.await %r4 : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>

    return %3, %4 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[IN_BUFF_0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[IN_BUFF_1:%.*]] = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUT_BUFF_DDR_0:%.*]] = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUT_BUFF_DDR_1:%.*]] = VPURT.DeclareBuffer <DDR> <6000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUT_BUFF_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUT_BUFF_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_0:%.*]], [[R_0:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:     [[NNDMA_0:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_BUFF_0]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_1:%.*]], [[R_1:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:     [[NNDMA_1:%.*]] = VPUIP.NNDMA inputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_BUFF_1]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[NNDMA_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_2:%.*]] = async.execute [[[T_0]], [[T_1]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0
    // CHECK:         VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:   [[T_3:%.*]], [[R_3:%.*]]:2 = async.execute [[[T_2]]] ([[R_0]] as [[SW_KERNEL_0_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>, [[R_1]] as [[SW_KERNEL_1_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> (!async.value<memref<1x1x1x1000xf16, @DDR>>, !async.value<memref<1x1x1x1000xf16, @DDR>>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64}
    // CHECK:     [[SW_KERNEL_0:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu inputs([[SW_KERNEL_0_INPUT]] as [[INPUT_0:%.*]]: memref<1x1x1x1000xf16, @DDR>, [[SW_KERNEL_1_INPUT]] as [[INPUT_1:%.*]]: memref<1x1x1x1000xf16, @DDR>) outputs([[OUT_BUFF_DDR_0]] as [[OUTPUT_0:%.*]]: memref<1x1x1x1000xf16, @DDR>, [[OUT_BUFF_DDR_1]] as [[OUTPUT_1:%.*]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>)
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT_0]], [[OUTPUT_0]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT_1]], [[OUTPUT_1]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[SW_KERNEL_0]]#0, [[SW_KERNEL_0]]#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_4:%.*]] = async.execute [[[T_3]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0
    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:   [[T_5:%.*]], [[R_5:%.*]] = async.execute [[[T_4]]] ([[R_3]]#0 as [[NNDMA_2_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64}
    // CHECK:     [[NNDDMA_2:%.*]] = VPUIP.NNDMA inputs([[NNDMA_2_INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUT_BUFF_CMX_0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[NNDDMA_2]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_6:%.*]], [[R_6:%.*]] = async.execute [[[T_4]]] ([[R_3]]#1 as [[NNDMA_3_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64}
    // CHECK:     [[NNDDMA_3:%.*]] = VPUIP.NNDMA inputs([[NNDMA_3_INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUT_BUFF_CMX_1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[NNDDMA_3]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[ASYNC_WAIT_1:%.*]] = async.await [[R_5]] : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>
    // CHECK:   [[ASYNC_WAIT_2:%.*]] = async.await [[R_6]] : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>
    // CHECK:   return [[ASYNC_WAIT_1]], [[ASYNC_WAIT_2]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @cache_flush()
    attributes {
        VPU.task_type = @CACHE_FLUSH
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @DontAddCacheHandlingSwKernel
func.func @DontAddCacheHandlingSwKernel(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %in_ddr_0  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr_0 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %dma_0 : memref<1x1x1x1000xf16, @DDR>
    }

    %t1 = async.execute [%t0] attributes {VPUIP.executor = @SHAVE_ACT} {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0{
            VPUIP.SW.Kernel.run
        }
        async.yield
    }

    return %in_ddr_0 : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[IN_BUFF_0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_0:%.*]], [[R_0:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:     [[NNDMA_0:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_BUFF_0]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_1:%.*]] = async.execute [[[T_0]]] attributes {VPUIP.executor = @SHAVE_ACT}
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0
    // CHECK:         VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:   return [[IN_BUFF_0:%.*]] : memref<1x1x1x1000xf16, @DDR>
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

// CHECK-LABEL: @AddCacheFlush
func.func @AddCacheFlush(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {
    %in_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_ddr_0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t0] (%r0 as %sw_kernel_input : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%sw_kernel_input as %in_buff_0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_ddr_0 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, @DDR>
        }
        async.yield %sw_kernel : memref<1x1x1x1000xf16, @DDR>
    }

     %t2, %r2 = async.execute [%t1] (%r1 as %nndma_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>  attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
        %dma_1 = VPUIP.NNDMA inputs(%nndma_input : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %3 = async.await %r2 : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>

    return %3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[IN_BUFF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_0:%.*]], [[R_0:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:     [[NNDMA_0:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_BUFF]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_2:%.*]], [[R_2:%.*]] = async.execute [[[T_0]]] ([[R_0]] as [[SW_KERNEL_1_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64}
    // CHECK:     [[SW_KERNEL_1:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[SW_KERNEL_1_INPUT]] as [[INPUT:%.*]]: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[OUT_BUFF]] as [[OUTPUT:%.*]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT]], [[OUTPUT]]) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[SW_KERNEL_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_3:%.*]] = async.execute [[[T_2]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0
    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:   [[T_4:%.*]], [[R_5:%.*]] = async.execute [[[T_3]]] ([[R_2]] as [[NNDMA_1_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64}
    // CHECK:     [[NNDDMA_1:%.*]] = VPUIP.NNDMA inputs([[NNDMA_1_INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[NNDDMA_1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[ASYNC_WAIT:%.*]] = async.await [[R_5]] : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>
    // CHECK:   return [[ASYNC_WAIT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
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

// CHECK-LABEL: @AddCacheFlushInvalidate
func.func @AddCacheFlushInvalidate(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {
    %in_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %dma_0 : memref<1x1x1x1000xf16, @DDR>
    }

    %t1, %r1 = async.execute [%t0] (%r0 as %sw_kernel_input : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%sw_kernel_input as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_cmx as %out_buff_0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, @DDR>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
        async.yield %sw_kernel : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

     %t2, %r2 = async.execute [%t1] (%r1 as %nndma_input : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>  attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
        %dma_1 = VPUIP.NNDMA inputs(%nndma_input : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %3 = async.await %r2 : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>

    return %3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[IN_BUFF:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_0:%.*]], [[R_0:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:     [[NNDMA_0:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_BUFF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:     async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_1:%.*]] = async.execute [[[T_0]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0
    // CHECK:         VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:   [[T_2:%.*]], [[R_2:%.*]] = async.execute [[[T_1]]] ([[R_0]] as [[SW_KERNEL_1_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64}
    // CHECK:     [[SW_KERNEL_1:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[SW_KERNEL_1_INPUT]] as [[INPUT:%.*]]: memref<1x1x1x1000xf16, @DDR>) outputs([[OUT_BUFF]] as [[OUTPUT:%.*]]: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT]], [[OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[SW_KERNEL_1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_3:%.*]], [[R_3:%.*]] = async.execute [[[T_2]]] ([[R_2]] as [[NNDMA_1_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64}
    // CHECK:     [[NNDDMA_1:%.*]] = VPUIP.NNDMA inputs([[NNDMA_1_INPUT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[NNDDMA_1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[ASYNC_WAIT:%.*]] = async.await [[R_3]] : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>
    // CHECK:   return [[ASYNC_WAIT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
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

// CHECK-LABEL: @DontAddCacheOpsInOutInCMX
func.func @DontAddCacheOpsInOutInCMX(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {
    %in_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t0] (%r0 as %sw_kernel_input : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%sw_kernel_input as %in_buff_0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_cmx as %out_buff_0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
        async.yield %sw_kernel : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

     %t2, %r2 = async.execute [%t1] (%r1 as %nndma_input : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>  attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
        %dma_1 = VPUIP.NNDMA inputs(%nndma_input : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %3 = async.await %r2 : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>

    return %3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[INPUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_0:%.*]], [[R_0:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
    // CHECK:     [[NNDMA_0:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[INPUT_CMX]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[NNDMA_0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_1:%.*]], [[R_1:%.*]] = async.execute [[[T_0]]] ([[R_0]] as [[SW_KERNEL_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64}
    // CHECK:     [[SW_KERNEL:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[SW_KERNEL_INPUT]] as [[INPUT:%.*]]: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_CMX]] as [[OUTPUT:%.*]]: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT]], [[OUTPUT]]) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[SW_KERNEL]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_2:%.*]], [[R_2:%.*]] = async.execute [[[T_1]]] ([[R_1]] as [[INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64} {
    // CHECK:     [[NNDMA_1:%.*]] = VPUIP.NNDMA inputs([[INPUT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield %3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[AWAIT:%.*]] = async.await [[R_2]] : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>
    // CHECK:   return [[AWAIT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_Convert(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>)
    attributes {
        VPU.kernel_code = "single_shave_convert.cpp",
        VPU.kernel_entry = "single_shave_convert"
    }

func.func private @builtin_Gather(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "single_shave_gather.cpp",
        VPU.kernel_entry = "single_shave_gather",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @AddCacheFlushInvalidateSwOpForDDRInputCMXOutput
func.func @AddCacheFlushInvalidateSwOpForDDRInputCMXOutput(%arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>, %arg1: memref<1x1x1x1xf16, @DDR>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {
    %cst = const.Declare memref<51865x512xf16> = dense<1.0> : tensor<51865x512xf32>, [#const.CastElemType<f16>]

    %indices_fp16_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1457216> -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
    %indices_si32_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1457152> -> memref<1x1x1x1xsi32, [@CMX_NN, 0]>
    %indices_input_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1457152> -> memref<1x1xsi32, [@CMX_NN, 0]>

    %output_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1473024> -> memref<1x1x512xf16, [@CMX_NN, 0]>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x1xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg1 : memref<1x1x1x1xf16, @DDR>) outputs(%indices_fp16_cmx : memref<1x1x1x1xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
        async.yield %dma_0 : memref<1x1x1x1xf16, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t0] (%r0 as %sw_kernel_input : !async.value<memref<1x1x1x1xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel_1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert
                    inputs(%sw_kernel_input as %in_buff_0: memref<1x1x1x1xf16, [@CMX_NN, 0]>)
                    outputs(%indices_si32_cmx as %out_buff_0: memref<1x1x1x1xsi32, [@CMX_NN, 0]>)
                    on tile 0 -> memref<1x1x1x1xsi32, [@CMX_NN, 0]> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1xsi32, [@CMX_NN, 0]>
        }
        async.yield %sw_kernel_1 : memref<1x1x1x1xsi32, [@CMX_NN, 0]>
    }

    %t2, %r2 = async.execute [%t1] -> !async.value<memref<1x1x512xf16, [@CMX_NN, 0]>>  attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64} {
        %sw_kernel_2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
                    inputs(%cst as %arg2: memref<51865x512xf16>, %indices_input_cmx as %arg3: memref<1x1xsi32, [@CMX_NN, 0]>)
                    outputs(%output_cmx as %arg4: memref<1x1x512xf16, [@CMX_NN, 0]>)
                    on tile 0 -> memref<1x1x512xf16, [@CMX_NN, 0]>{
                VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg2, %arg3, %arg4)
                    : memref<51865x512xf16>
                    , memref<1x1xsi32, [@CMX_NN, 0]>
                    , memref<1x1x512xf16, [@CMX_NN, 0]>
        }
        async.yield %sw_kernel_2 : memref<1x1x512xf16, [@CMX_NN, 0]>
    }

    %4 = async.await %r2 : !async.value<memref<1x1x512xf16, [@CMX_NN, 0]>>

    return %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[CST:%.*]] = const.Declare memref<51865x512xf16> = dense<1.000000e+00> : tensor<51865x512xf32>, [#const.CastElemType<f16>]
    // CHECK:   [[INDICES_FP16_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1457216> -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[INDICES_SI32_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1457152> -> memref<1x1x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:   [[INDICES_INPUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1457152> -> memref<1x1xsi32, [@CMX_NN, 0]>

    // CHECK:   [[OUTPUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1473024> -> memref<1x1x512xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_0:%.*]], [[R_0:%.*]] = async.execute -> !async.value<memref<1x1x1x1xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
    // CHECK:    [[NNDMA_1:%.*]] = VPUIP.NNDMA inputs(%arg1 : memref<1x1x1x1xf16, @DDR>) outputs([[INDICES_FP16_CMX]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
    // CHECK:    async.yield [[NNDMA_1]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_1:%.*]], [[R_1:%.*]] = async.execute [[[T_0]]] ([[R_0]] as [[SW_KERNEL_INPUT:%.*]]: !async.value<memref<1x1x1x1xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:    [[SW_KERNEL_1:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[SW_KERNEL_INPUT]] as [[INPUT:%.*]]: memref<1x1x1x1xf16, [@CMX_NN, 0]>) outputs([[INDICES_SI32_CMX]] as [[OUTPUT:%.*]]: memref<1x1x1x1xsi32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1xsi32, [@CMX_NN, 0]>{
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT]], [[OUTPUT]]) : memref<1x1x1x1xf16, [@CMX_NN, 0]>, memref<1x1x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:    async.yield [[SW_KERNEL_1]] : memref<1x1x1x1xsi32, [@CMX_NN, 0]>

    // CHECK:    [[T_2:%.*]] = async.execute [[[T_1]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:     VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:    [[T_3:%.*]], [[R_3:%.*]] = async.execute [[[T_2]]] -> !async.value<memref<1x1x512xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64} {
    // CHECK:       [[SW_KERNEL_2:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs([[CST]] as [[INPUT_1:%.*]]: memref<51865x512xf16>, [[INDICES_INPUT_CMX]] as [[INPUT_2:%.*]]: memref<1x1xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_CMX]] as [[OUTPUT:%.*]]: memref<1x1x512xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x512xf16, [@CMX_NN, 0]>{
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [1, 0]}([[INPUT_1]], [[INPUT_2]], [[OUTPUT]]) : memref<51865x512xf16>, memref<1x1xsi32, [@CMX_NN, 0]>, memref<1x1x512xf16, [@CMX_NN, 0]>
    // CHECK:       async.yield [[SW_KERNEL_2]] : memref<1x1x512xf16, [@CMX_NN, 0]>

    // CHECK:   async.await [[R_3]] : !async.value<memref<1x1x512xf16, [@CMX_NN, 0]>>
    // CHECK:   return %arg0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
}

// -----

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Minimum(memref<*xf16>, memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @AddCacheFlushCacheFlushInvalidate
func.func @AddCacheFlushCacheFlushInvalidate(%arg0: memref<64x32x32x16xf16, @DDR>, %arg1: memref<64x32x32x16xf16, [@CMX_NN, 0]>, %arg2: memref<64x32x32x16xf16, [@CMX_NN, 0]>) -> (memref<64x32x32x16xf16, [@CMX_NN, 0]>, memref<64x32x32x16xf16, [@CMX_NN, 0]>) {
  %cst = const.Declare memref<64x32x32x16xf16> = dense<1.000000e+00> : tensor<64x32x32x16xf16>
  %cst_0 = const.Declare memref<64x32x32x16xf16> = dense<2.000000e+00> : tensor<64x32x32x16xf16>
  %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<64x32x32x16xf16, @DDR>
  %1 = VPURT.DeclareBuffer <DDR> <2097152> -> memref<64x32x32x16xf16, @DDR>

  %token, %bodyResults = async.execute -> !async.value<memref<64x32x32x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64} {
    %sw_kernel_1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%arg0 as %input_1: memref<64x32x32x16xf16>, %cst as %input_2: memref<64x32x32x16xf16>) outputs(%0 as %output: memref<64x32x32x16xf16>) on tile 0 -> memref<64x32x32x16xf16, @DDR>{
      VPUIP.SW.Kernel.run(%input_1, %input_2, %output) : memref<64x32x32x16xf16>, memref<64x32x32x16xf16>, memref<64x32x32x16xf16>
    }
    async.yield %sw_kernel_1 : memref<64x32x32x16xf16, @DDR>
  }

  %token_1, %bodyResults_2 = async.execute [%token] (%bodyResults as %input: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    %sw_kernel_2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%input as %input_1: memref<64x32x32x16xf16>, %cst_0 as %input_2: memref<64x32x32x16xf16>) outputs(%1 as %output: memref<64x32x32x16xf16>) on tile 0 -> memref<64x32x32x16xf16, @DDR>{
      VPUIP.SW.Kernel.run(%input_1, %input_2, %output) : memref<64x32x32x16xf16>, memref<64x32x32x16xf16>, memref<64x32x32x16xf16>
    }
    async.yield %sw_kernel_2 : memref<64x32x32x16xf16, @DDR>
  }

  %token_4, %bodyResults_5 = async.execute [%token] (%bodyResults as %input: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
    %nn_dma_1 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : memref<64x32x32x16xf16, @DDR>) outputs(%arg1 : memref<64x32x32x16xf16, [@CMX_NN, 0]>) -> memref<64x32x32x16xf16, [@CMX_NN, 0]>
    async.yield %arg1 : memref<64x32x32x16xf16, [@CMX_NN, 0]>
  }

  %token_3, %bodyResults_4 = async.execute [%token_1] (%bodyResults_2 as %input: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
    %nn_dma_2 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : memref<64x32x32x16xf16, @DDR>) outputs(%arg2 : memref<64x32x32x16xf16, [@CMX_NN, 0]>) -> memref<64x32x32x16xf16, [@CMX_NN, 0]>
    async.yield %arg2 : memref<64x32x32x16xf16, [@CMX_NN, 0]>
  }

  %2 = async.await %bodyResults_4 : !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>>
  %3 = async.await %bodyResults_5 : !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>>

  return %2, %3 : memref<64x32x32x16xf16, [@CMX_NN, 0]>, memref<64x32x32x16xf16, [@CMX_NN, 0]>

    // CHECK:   [[CST_0:%.*]] = const.Declare memref<64x32x32x16xf16> = dense<1.000000e+00> : tensor<64x32x32x16xf16>
    // CHECK:   [[CST_1:%.*]] = const.Declare memref<64x32x32x16xf16> = dense<2.000000e+00> : tensor<64x32x32x16xf16>
    // CHECK:   [[BUFF_0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<64x32x32x16xf16, @DDR>
    // CHECK:   [[BUFF_1:%.*]] = VPURT.DeclareBuffer <DDR> <2097152> -> memref<64x32x32x16xf16, @DDR>

    // CHECK:   [[TOKEN_0:%.*]] = async.execute attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:    [[TOKEN_1:%.*]], [[RESULT_1:%.*]] = async.execute [[[TOKEN_0]]] -> !async.value<memref<64x32x32x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64} {
    // CHECK:      [[SW_KERNEL_1:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%arg0 as [[INPUT_1:%.*]]: memref<64x32x32x16xf16>, [[CST_0]] as [[INPUT_2:%.*]]: memref<64x32x32x16xf16>) outputs([[BUFF_0]] as [[OUTPUT:%.*]]: memref<64x32x32x16xf16>) on tile 0 -> memref<64x32x32x16xf16, @DDR>{
    // CHECK:        VPUIP.SW.Kernel.run([[INPUT_1]], [[INPUT_2]], [[OUTPUT]]) : memref<64x32x32x16xf16>, memref<64x32x32x16xf16>, memref<64x32x32x16xf16>
    // CHECK:      async.yield [[SW_KERNEL_1]] : memref<64x32x32x16xf16, @DDR>

    // CHECK:   [[TOKEN_2:%.*]] = async.execute [[[TOKEN_1]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0{
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK:      async.yield

    // CHECK:    [[TOKEN_3:%.*]] = async.execute [[[TOKEN_2]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK:      async.yield

    // CHECK:    [[TOKEN_4:%.*]], [[RESULT_4:%.*]] = async.execute [[[TOKEN_3]]] ([[RESULT_1]] as [[INPUT:%.*]]: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:      [[SW_KERNEL_2:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs([[INPUT]] as [[INPUT_1:%.*]]: memref<64x32x32x16xf16>, [[CST_1]] as [[INPUT_2:%.*]]: memref<64x32x32x16xf16>) outputs([[BUFF_1]] as [[OUTPUT:%.*]]: memref<64x32x32x16xf16>) on tile 0 -> memref<64x32x32x16xf16, @DDR>{
    // CHECK:        VPUIP.SW.Kernel.run([[INPUT_1]], [[INPUT_2]], [[OUTPUT]]) : memref<64x32x32x16xf16>, memref<64x32x32x16xf16>, memref<64x32x32x16xf16>
    // CHECK:      async.yield [[SW_KERNEL_2]] : memref<64x32x32x16xf16, @DDR>

    // CHECK:    [[TOKEN_5:%.*]] = async.execute [[[TOKEN_4]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0{
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK:      async.yield

    // CHECK:    [[TOKEN_6:%.*]], [[RESULT_6:%.*]] = async.execute [[[TOKEN_2]]] ([[RESULT_1]] as [[INPUT_DMA_1:%.*]]: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
    // CHECK:      VPUIP.NNDMA {port = 0 : i64} inputs([[INPUT_DMA_1]] : memref<64x32x32x16xf16, @DDR>) outputs(%arg1 : memref<64x32x32x16xf16, [@CMX_NN, 0]>) -> memref<64x32x32x16xf16, [@CMX_NN, 0]>
    // CHECK:      async.yield %arg1 : memref<64x32x32x16xf16, [@CMX_NN, 0]>

    // CHECK:    [[TOKEN_7:%.*]], [[RESULT_7:%.*]] = async.execute [[[TOKEN_5]]] ([[RESULT_4]] as [[INPUT_DMA_2:%.*]]: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
    // CHECK:      VPUIP.NNDMA {port = 0 : i64} inputs([[INPUT_DMA_2]] : memref<64x32x32x16xf16, @DDR>) outputs(%arg2 : memref<64x32x32x16xf16, [@CMX_NN, 0]>) -> memref<64x32x32x16xf16, [@CMX_NN, 0]>
    // CHECK:      async.yield %arg2 : memref<64x32x32x16xf16, [@CMX_NN, 0]>

    // CHECK:    [[AWAIT_2:%.*]] = async.await [[RESULT_7]] : !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>>
    // CHECK:    [[AWAIT_1:%.*]] = async.await [[RESULT_6]] : !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>>
    // CHECK:    return [[AWAIT_2]], [[AWAIT_1]] : memref<64x32x32x16xf16, [@CMX_NN, 0]>, memref<64x32x32x16xf16, [@CMX_NN, 0]>
}

// -----

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Minimum(memref<*xf16>, memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
    IE.CNNNetwork entryPoint : @AddCacheFlushInvalidate inputsInfo : {
    DataInfo "Parameter_224" : tensor<64x32x32x16xf16>
  } outputsInfo : {
    DataInfo "Minimum_228" : tensor<64x32x32x16xf16>
  }

// CHECK-LABEL: @AddCacheFlushInvalidate
func.func @AddCacheFlushInvalidate(%arg0: memref<64x32x32x16xf16, @DDR>, %arg1: memref<64x32x32x16xf16, [@CMX_NN, 0]>) -> memref<64x32x32x16xf16, [@CMX_NN, 0]> {
  %cst = const.Declare memref<64x32x32x16xf16> = dense<1.000000e+00> : tensor<64x32x32x16xf16>
  %cst_0 = const.Declare memref<64x32x32x16xf16> = dense<2.000000e+00> : tensor<64x32x32x16xf16>
  %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<64x32x32x16xf16, @DDR>
  %1 = VPURT.DeclareBuffer <DDR> <2097152> -> memref<64x32x32x16xf16, @DDR>

  %token, %bodyResults = async.execute -> !async.value<memref<64x32x32x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64} {
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%arg0 as %arg2: memref<64x32x32x16xf16>, %cst as %arg3: memref<64x32x32x16xf16>) outputs(%0 as %arg4: memref<64x32x32x16xf16>) on tile 0 -> memref<64x32x32x16xf16, @DDR>{
      VPUIP.SW.Kernel.run(%arg2, %arg3, %arg4) : memref<64x32x32x16xf16>, memref<64x32x32x16xf16>, memref<64x32x32x16xf16>
    }
    async.yield %0 : memref<64x32x32x16xf16, @DDR>
  }

  %token_1, %bodyResults_2 = async.execute [%token] (%bodyResults as %arg2: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%arg2 as %arg3: memref<64x32x32x16xf16>, %cst_0 as %arg4: memref<64x32x32x16xf16>) outputs(%1 as %arg5: memref<64x32x32x16xf16>) on tile 0 -> memref<64x32x32x16xf16, @DDR>{
      VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<64x32x32x16xf16>, memref<64x32x32x16xf16>, memref<64x32x32x16xf16>
    }
    async.yield %1 : memref<64x32x32x16xf16, @DDR>
  }

  %token_3, %bodyResults_4 = async.execute [%token_1] (%bodyResults_2 as %arg2: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
    %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : memref<64x32x32x16xf16, @DDR>) outputs(%arg1 : memref<64x32x32x16xf16, [@CMX_NN, 0]>) -> memref<64x32x32x16xf16, [@CMX_NN, 0]>
    async.yield %arg1 : memref<64x32x32x16xf16, [@CMX_NN, 0]>
  }

  %2 = async.await %bodyResults_4 : !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>>
  return %2 : memref<64x32x32x16xf16, [@CMX_NN, 0]>

    // CHECK:   [[CST_0:%.*]] = const.Declare memref<64x32x32x16xf16> = dense<1.000000e+00> : tensor<64x32x32x16xf16>
    // CHECK:   [[CST_1:%.*]] = const.Declare memref<64x32x32x16xf16> = dense<2.000000e+00> : tensor<64x32x32x16xf16>
    // CHECK:   [[BUFF_0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<64x32x32x16xf16, @DDR>
    // CHECK:   [[BUFF_1:%.*]] = VPURT.DeclareBuffer <DDR> <2097152> -> memref<64x32x32x16xf16, @DDR>

    // CHECK:   [[TOKEN_0:%.*]] = async.execute attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK:     async.yield

    // CHECK:    [[TOKEN_1:%.*]], [[RESULT_1:%.*]] = async.execute [[[TOKEN_0]]] -> !async.value<memref<64x32x32x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64} {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%arg0 as %arg2: memref<64x32x32x16xf16>, [[CST_0]] as %arg3: memref<64x32x32x16xf16>) outputs([[BUFF_0]] as %arg4: memref<64x32x32x16xf16>) on tile 0 -> memref<64x32x32x16xf16, @DDR>{
    // CHECK:        VPUIP.SW.Kernel.run(%arg2, %arg3, %arg4) : memref<64x32x32x16xf16>, memref<64x32x32x16xf16>, memref<64x32x32x16xf16>
    // CHECK:      async.yield [[BUFF_0]] : memref<64x32x32x16xf16, @DDR>

    // CHECK:   [[TOKEN_2:%.*]] = async.execute [[[TOKEN_1]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK:      async.yield

    // CHECK:    [[TOKEN_3:%.*]], [[RESULT_3:%.*]] = async.execute [[[TOKEN_2]]] ([[RESULT_1]] as %arg2: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%arg2 as %arg3: memref<64x32x32x16xf16>, [[CST_1]] as %arg4: memref<64x32x32x16xf16>) outputs([[BUFF_1]] as %arg5: memref<64x32x32x16xf16>) on tile 0 -> memref<64x32x32x16xf16, @DDR>{
    // CHECK:        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<64x32x32x16xf16>, memref<64x32x32x16xf16>, memref<64x32x32x16xf16>
    // CHECK:      async.yield %1 : memref<64x32x32x16xf16, @DDR>

    // CHECK:    [[TOKEN_4:%.*]] = async.execute [[[TOKEN_3]]] attributes {VPUIP.executor = @SHAVE_ACT} {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0{
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK:      async.yield

    // CHECK:    [[TOKEN_5:%.*]], [[RESULT_5:%.*]] = async.execute [[[TOKEN_4]]] ([[RESULT_3]] as %arg2: !async.value<memref<64x32x32x16xf16, @DDR>>) -> !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
    // CHECK:      VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : memref<64x32x32x16xf16, @DDR>) outputs(%arg1 : memref<64x32x32x16xf16, [@CMX_NN, 0]>) -> memref<64x32x32x16xf16, [@CMX_NN, 0]>
    // CHECK:      async.yield %arg1 : memref<64x32x32x16xf16, [@CMX_NN, 0]>

    // CHECK:    [[AWAIT:%.*]] = async.await [[RESULT_5]] : !async.value<memref<64x32x32x16xf16, [@CMX_NN, 0]>>
    // CHECK:    return [[AWAIT]] : memref<64x32x32x16xf16, [@CMX_NN, 0]>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_Minimum(memref<*xf16>, memref<*xf16>, memref<*xf16>)
    attributes {
        VPU.kernel_code = "eltwise_min.cpp",
        VPU.kernel_entry = "eltwise_min",
        VPU.task_type = @COMPUTE
    }
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

// CHECK-LABEL: @AddCacheHandlingSwOpOneSwKernelMultipleDependencies
func.func @AddCacheHandlingSwOpOneSwKernelMultipleDependencies(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>, %arg2: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]> {
    %in_cmx_0  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %in_ddr_0  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr_0 = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr_1 = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x2000xf16, @DDR>

    %t0, %r0 = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_cmx_0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        async.yield %dma_0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t0] (%r0 as %sw_kernel_input : !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%sw_kernel_input as %in_buff_0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_ddr_0 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, @DDR>
        }
        async.yield %sw_kernel : memref<1x1x1x1000xf16, @DDR>
    }

    %t2, %r2 = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %dma_1 = VPUIP.NNDMA inputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr_0 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        async.yield %dma_1 : memref<1x1x1x1000xf16, @DDR>
    }


    %t3, %r3 = async.execute [%t1, %t2] (%r1 as %input_1 : !async.value<memref<1x1x1x1000xf16, @DDR>>, %r2 as %input_2 : !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x2000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %concat = VPUIP.ConcatView inputs(%input_1, %input_2 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) outputs(%out_ddr_1 : memref<1x1x1x2000xf16, @DDR>) -> memref<1x1x1x2000xf16, @DDR>
        %result = VPUIP.NNDMA inputs(%concat : memref<1x1x1x2000xf16, @DDR>) outputs(%arg2 : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
        async.yield %result : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    }

    %3 = async.await %r3 : !async.value<memref<1x1x1x2000xf16, [@CMX_NN, 0]>>
    return %3: memref<1x1x1x2000xf16, [@CMX_NN, 0]>

    // CHECK:   [[IN_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:   [[IN_DDR_0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUT_DDR_0:%.*]] = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUT_DDR_1:%.*]] = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x1x2000xf16, @DDR>

    // CHECK:   [[T_0:%.*]], [[R_0:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:     [[NN_DMA_0:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_CMX_0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:     async.yield [[NN_DMA_0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK:   [[T_1:%.*]], [[R_1:%.*]] = async.execute [[[T_0]]] ([[R_0]] as [[SW_KERNEL_INPUT:%.*]]: !async.value<memref<1x1x1x1000xf16, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64}
    // CHECK:    [[SW_KERNEL:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[SW_KERNEL_INPUT]] as [[INPUT:%.*]]: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[OUT_DDR_0]] as [[OUTPUT:%.*]]: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INPUT]], [[OUTPUT]]) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:    async.yield [[SW_KERNEL]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_2:%.*]] = async.execute [[[T_1]]] attributes {VPUIP.executor = @SHAVE_ACT}
    // CHECK:    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0
    // CHECK:      VPUIP.SW.Kernel.run
    // CHECK:    async.yield

    // CHECK:   [[T_3:%.*]], [[R_3:%.*]] = async.execute -> !async.value<memref<1x1x1x1000xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:    [[NN_DMA_1:%.*]] = VPUIP.NNDMA inputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) outputs([[IN_DDR_0]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:    async.yield [[NN_DMA_1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[T_4:%.*]], [[R_4:%.*]] = async.execute [[[T_3]], [[T_2]]] ([[R_1]] as [[INPUT_0:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>, [[R_3]] as [[INPUT_1:%.*]]: !async.value<memref<1x1x1x1000xf16, @DDR>>) -> !async.value<memref<1x1x1x2000xf16, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64}
    // CHECK:    [[CONCAT_VIEW:%.*]] = VPUIP.ConcatView inputs([[INPUT_0]], [[INPUT_1]] : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) outputs([[OUT_DDR_1]] : memref<1x1x1x2000xf16, @DDR>) -> memref<1x1x1x2000xf16, @DDR>
    // CHECK:    [[NN_DMA_2:%.*]] = VPUIP.NNDMA inputs([[CONCAT_VIEW]] : memref<1x1x1x2000xf16, @DDR>) outputs(%arg2 : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK:    async.yield [[NN_DMA_2]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>

    // CHECK:   [[AWAIT:%.*]] = async.await [[R_4]] : !async.value<memref<1x1x1x2000xf16, [@CMX_NN, 0]>>
    // CHECK:   return [[AWAIT]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
}
