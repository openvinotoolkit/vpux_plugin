//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --lower-VPUIP-to-ELF %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

module @Convert {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "Parameter_6" : tensor<1x100xui32>
  } outputsInfo :  {
    DataInfo "Convert_7" : tensor<1x100xui32>
  }

  func.func @main(%arg0: memref<1x100xui32, @DDR>, %arg1: memref<1x100xui32, @DDR>) -> memref<1x100xui32, @DDR> {
    %bar_0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK:       %[[VAL1:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>

    %buf_0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x100xui32, @DDR>
    // CHECK:       %[[VAL2:.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x100xui32, @DDR>

    %cst_0 = const.Declare memref<1x100xui32> = dense<1> : tensor<1x100xui32>
    // CHECK-DAG:       %[[VAL3:.*]] = const.Declare memref<1x100xui32> = dense<1> : tensor<1x100xui32>

    VPURT.Task updates(%bar_0 : !VPURT.Barrier) {
      %dma_0 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x100xui32>) outputs(%buf_0 : memref<1x100xui32, @DDR>) -> memref<1x100xui32, @DDR>
    }

    VPURT.Task waits(%bar_0 : !VPURT.Barrier) {
      %dma_1 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf_0 : memref<1x100xui32, @DDR>) outputs(%arg1 : memref<1x100xui32, @DDR>) -> memref<1x100xui32, @DDR>
    }
    // CHECK:       %[[VAL5:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL2]] : memref<1x100xui32, @DDR>) outputs(%[[VAL6:.*]] : memref<1x100xui32, @DDR>) waits(%[[VAL1]] : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    // CHECK:       %[[VAL4:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL3]] : memref<1x100xui32>) outputs(%[[VAL2]] : memref<1x100xui32, @DDR>) nextDMAIdx(%[[VAL5]] : !VPURegMapped.Index<0:0:1>) updates(%[[VAL1]] : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:       %[[VAL7:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELFNPU37XX.Section
    // CHECK:       %[[VAL8:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section
    // CHECK:       %[[VAL9:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.ConstIO"} -> !ELFNPU37XX.Section {

    return %arg1 : memref<1x100xui32, @DDR>
    // CHECK:       return %arg1 : memref<1x100xui32, @DDR>

  }
}
