//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --barriers-to-managed-barriers %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ProfilingTest attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 6 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @main inputsInfo : {
      DataInfo "input_0" : tensor<1x16x16x16xf16>
      DataInfo "input_1" : tensor<16x1x1x1xi64>
  } outputsInfo : {
      DataInfo "output_0" : tensor<1x16x64x64xf16>
  } profilingOutputsInfo : {
      DataInfo "profilingOutput" : tensor<32xui32>
  }
  func.func @main() {
    ELF.Main @ELFMain {
        VPUASM.ConfigureBarrier @ConfigureBarrier_0_0 idx(!VPURegMapped.Index<0:0:0>) workItemIdx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(3 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier_0_1 idx(!VPURegMapped.Index<0:0:1>) (1) => (4) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier_0_2 idx(!VPURegMapped.Index<0:0:2>) workItemIdx(!VPURegMapped.Index<0:0:1>) (2) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier_0_3 idx(!VPURegMapped.Index<0:0:3>) (3) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier_0_4 idx(!VPURegMapped.Index<0:0:4>) workItemIdx(!VPURegMapped.Index<0:0:2>) (0) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @dummyName idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(1 : 1)
    }

    return
  }
}

//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_0 idx(!VPURegMapped.Index<0:0:0>) workItemIdx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(3 : 1)
//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_1 idx(!VPURegMapped.Index<0:0:1>) (1) => (4) counts(1 : 1)
//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_2 idx(!VPURegMapped.Index<0:0:2>) workItemIdx(!VPURegMapped.Index<0:0:1>) (2) => (-1) counts(1 : 1)
//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_3 idx(!VPURegMapped.Index<0:0:3>) (3) => (-1) counts(1 : 1)
//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_4 idx(!VPURegMapped.Index<0:0:4>) workItemIdx(!VPURegMapped.Index<0:0:2>) (0) => (-1) counts(1 : 1)
//CHECK: VPUASM.ManagedBarrier @dummyName idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(1 : 1)
