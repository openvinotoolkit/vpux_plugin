//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --setup-location-verifier="mode=fast" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @mainModule {
}
// CHECK: module @mainModule attributes
// CHECK-SAME: IE.LocationsVerificationMode  = "fast"
