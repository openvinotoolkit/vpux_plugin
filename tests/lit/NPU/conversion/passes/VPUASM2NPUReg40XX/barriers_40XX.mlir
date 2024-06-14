//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX40XX allow-custom-values=true" --convert-VPUASM-to-NPUReg40XX %s | FileCheck %s

module @OneDMAWithoutAttributes attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 6 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() {
    VPUASM.ConfigureBarrier @ConfigureBarrier_0_0 idx(!VPURegMapped.Index<0:0:0>) workItemIdx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(3 : 1)
    VPUASM.ConfigureBarrier @ConfigureBarrier_0_1 idx(!VPURegMapped.Index<0:0:1>) (17) => (12) counts(34 : 43)
    VPUASM.ManagedBarrier @ConfigureBarrier_0_0 idx(!VPURegMapped.Index<0:0:2>) workItemIdx(!VPURegMapped.Index<0:0:999>) (0) => (-1) counts(4 : 5)
    VPUASM.ManagedBarrier @ConfigureBarrier_0_1 idx(!VPURegMapped.Index<0:0:3>) (1) => (4) counts(23 : 32)

    return
  }
}

//CHECK-LABEL: @main
//CHECK: NPUReg40XX.ConfigureBarrier
//CHECK: next_same_id_ offset 0 size 32 = UINT 0xFFFFFFFF,
//CHECK: producer_count_ offset 4 size 16 = UINT 3,
//CHECK: consumer_count_ offset 6 size 16 = UINT 1,
//CHECK: real_id_ offset 8 size 8 = UINT 0,
//CHECK-NOT:  tb_work_item_idx

//CHECK: NPUReg40XX.ConfigureBarrier
//CHECK: next_same_id_ offset 0 size 32 = UINT 0xC,
//CHECK: producer_count_ offset 4 size 16 = UINT 0x22,
//CHECK: consumer_count_ offset 6 size 16 = UINT 0x2B,
//CHECK: real_id_ offset 8 size 8 = UINT 0x11,
//CHECK-NOT:  tb_work_item_idx

//CHECK: NPUReg40XX.ManagedBarrier
//CHECK: tb_next_same_id offset 0 size 32 = UINT 0xFFFFFFFF,
//CHECK: tb_producer_count offset 4 size 16 = UINT 4,
//CHECK: tb_consumer_count offset 6 size 16 = UINT 5,
//CHECK: tb_real_id offset 8 size 8 = UINT 0,
//CHECK: tb_work_item_idx offset 12 size 32 = UINT 0x3E7,

//CHECK: NPUReg40XX.ManagedBarrier
//CHECK: tb_next_same_id offset 0 size 32 = UINT 4,
//CHECK: tb_producer_count offset 4 size 16 = UINT 0x17,
//CHECK: tb_consumer_count offset 6 size 16 = UINT 0x20,
//CHECK: tb_real_id offset 8 size 8 = UINT 1,
//CHECK: tb_work_item_idx offset 12 size 32 = UINT 0,
