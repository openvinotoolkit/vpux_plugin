//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --setup-location-verifier="mode=thorough" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

module @mainModule {
}
// CHECK: module @mainModule attributes
// CHECK-SAME: IE.LocationsVerificationMode  = "thorough"
