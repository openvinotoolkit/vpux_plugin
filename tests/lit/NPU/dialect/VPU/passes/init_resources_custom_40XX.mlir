//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --init-resources="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" %s | FileCheck %s  --strict-whitespace
// REQUIRES: arch-NPU40XX

// CHECK: module @mode attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>}
module @mode attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>} {
}

// -----

// CHECK: module @arch attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>}
module @arch attributes {VPU.arch = #VPU.arch_kind<NPU37XX>} {
}

// -----

// CHECK: module @executors attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>}
module @executors {
    IE.ExecutorResource 5 of @DMA_NN
    IE.TileResource 5 of @NCE at 6.000000e+02 MHz
}

// CHECK-DAG:   {{  }}IE.ExecutorResource 5 of @DMA_NN
// CHECK-DAG:   {{  }}IE.ExecutorResource 1 of @M2I
// CHECK-DAG:   {{  }}IE.TileResource 5 of @NCE at 6.000000e+02 MHz {
// CHECK-DAG:   {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:   {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:   {{    }}IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:   {{    }}IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:   {{  }}IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}

// -----

// CHECK: module @memory attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>}
module @memory {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
        IE.MemoryResource 5 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 10000 bytes of @CMX_NN {VPU.bandwidth = 10 : i64, VPU.derateFactor = 2.0 : f64}
    }
    IE.MemoryResource 500000 bytes of @DDR
}

// CHECK-DAG:   {{  }}IE.ExecutorResource 2 of @DMA_NN
// CHECK-DAG:   {{  }}IE.ExecutorResource 1 of @M2I
// CHECK-DAG:   {{  }}IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
// CHECK-DAG:   {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:   {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:   {{    }}IE.MemoryResource 5 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:   {{    }}IE.MemoryResource 10000 bytes of @CMX_NN {VPU.bandwidth = 10 : i64, VPU.derateFactor = 2.000000e+00 : f64}
// CHECK-DAG:   {{  }}IE.MemoryResource 500000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
