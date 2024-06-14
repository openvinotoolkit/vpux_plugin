//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX30XX

// CHECK: module @test attributes {VPU.arch = #VPU.arch_kind<NPU30XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>}
module @test {

// CHECK-DAG:    {{  }}IE.PipelineOptions @Options {
// CHECK-DAG:    {{    }}IE.Option @VPU.BarrierMaxVariantSum : 512
// CHECK-DAG:    {{    }}IE.Option @VPU.BarrierMaxVariantCount : 512
// CHECK-DAG:    {{  }}}

// CHECK-DAG:    {{  }}IE.ExecutorResource 1 of @DMA_NN
// CHECK-DAG:    {{  }}IE.ExecutorResource 16 of @SHAVE_UPA
// CHECK-DAG:    {{  }}IE.TileResource 4 of @NCE at 7.000000e+02 MHz {
// CHECK-DAG:    {{    }}IE.ExecutorResource 5 of @DPU
// CHECK-DAG:    {{    }}IE.MemoryResource 825753 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}IE.MemoryResource 917504 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:    {{  }}IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

}
