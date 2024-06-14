//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --vpu-arch=%arch% --setup-per-barrier-variant-constraint %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

module @mainModule attributes { VPU.arch = #VPU.arch_kind<NPU37XX> } {
}

// CHECK: module @mainModule attributes
// CHECK: IE.PipelineOptions @Options
// CHECK: IE.Option @VPU.BarrierMaxVariantSum : 256
// CHECK: IE.Option @VPU.BarrierMaxVariantCount : 256
