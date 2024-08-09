//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW revision-id=3" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU40XX

// CHECK: module @test attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>, VPU.revisionID = #VPU.revision_id<REVISION_B>}
module @test {
}
