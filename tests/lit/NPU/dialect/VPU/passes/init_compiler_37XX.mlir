//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX

// CHECK: module @test attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>}
module @test {

// CHECK-DAG:    {{  }}IE.PipelineOptions @Options {
// CHECK-DAG:    {{    }}IE.Option @VPU.BarrierMaxVariantSum : 256
// CHECK-DAG:    {{    }}IE.Option @VPU.BarrierMaxVariantCount : 256
// CHECK-DAG:    {{    }}IE.Option @VPU.AutoPaddingODU : false
// CHECK-DAG:    {{    }}IE.Option @VPU.AutoPaddingIDU : false
// CHECK-DAG:    {{    }}IE.Option @VPU.MaxKernelSize : 11
// CHECK-DAG:    {{  }}}

// CHECK-DAG:    {{  }}IE.ExecutorResource 2 of @DMA_NN
// CHECK-DAG:    {{  }}IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
// CHECK-DAG:    {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @SHAVE_NN
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:    {{  }}IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

}
