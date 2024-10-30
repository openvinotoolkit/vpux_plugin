//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --link-all-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @Convolution attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @UsedMemory {
      IE.MemoryResource 21760 bytes of @CMX_NN
    }
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x16x16xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x16x16xf16, @DDR>) -> memref<1x16x16x16xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x16x16xf16, @DDR>

    %2 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%1 : memref<1x16x16x16xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%1 : memref<1x16x16x16xf16, @DDR>) previousDMA(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %4 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%1 : memref<1x16x16x16xf16, @DDR>) previousDMA(%3 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %5 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%1 : memref<1x16x16x16xf16, @DDR>) previousDMA(%4 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %6 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%1 : memref<1x16x16x16xf16, @DDR>) previousDMA(%5 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
    %7 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%1 : memref<1x16x16x16xf16, @DDR>) previousDMA(%6 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>

    %8 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>

    %9 = VPURegMapped.Enqueue at(%8 : !VPURegMapped.Index<0:0:0>) (%3 -> %5 : <0:0:1> -> <0:0:3>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %10 = VPURegMapped.Enqueue at(%8 : !VPURegMapped.Index<0:0:0>) (%6 -> %7 : <0:0:4> -> <0:0:5>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DPUVariant>}

    %11 = VPUMI40XX.MappedInference dmas((%2) : (!VPURegMapped.Index<0:0:0>)) barriers(%8 : !VPURegMapped.Index<0:0:0>) workItemTasks(%9 : !VPURegMapped.Index<0:0:0>) dmaCount([[0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(1) workItemCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x16x16xf16, @DDR>
  }
}

//CHECK: VPUMI40XX.NNDMA {port = 0 : i64}
//CHECK: VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64}
//CHECK: VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64}
//CHECK: VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64}
//CHECK: VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64}
//CHECK: VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64}
