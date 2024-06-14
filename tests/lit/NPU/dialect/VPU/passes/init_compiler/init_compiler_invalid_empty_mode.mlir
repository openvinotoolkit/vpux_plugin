//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" -verify-diagnostics %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// expected-error@+1 {{CompilationMode is already defined, probably you run '--init-compiler' twice}}
module @arch attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>} {
}
