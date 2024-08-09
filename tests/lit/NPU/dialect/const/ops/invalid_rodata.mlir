//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// expected-error@+1 {{'const.Rodata' op expects parent op 'const.Data'}}
const.Rodata @weights dense_resource<blob> : tensor<2x2xf32>

// -----

// expected-error@+1 {{'const.Data' op failed to verify that all operations in the body are of types 'vpux::Const::RodataOp,vpux::Const::RefOp'}}
const.Data @ov_bin {
    const.Declare tensor<2x2xf32> = dense<0.0> : tensor<2x2xf32>
}

// -----

const.Data @ov_bin {
    const.Rodata @wrong_type dense<0.0> : tensor<2x2xf32>
}

// expected-error@+1 {{'const.Declare' op annotated type and the underlying dereferenced type of 'const.Rodata' op do not match}}
const.Declare tensor<4x4xf32> = ref<@ov_bin::@wrong_type> : tensor<4x4xf32>

// -----

func.func @no_weights() -> () {
    return
}

// expected-error@+1 {{'const.Declare' op symbol does not point to a valid const.Rodata op}}
const.Declare tensor<4x4xf32> = ref<@no_weights> : tensor<4x4xf32>

// -----

// expected-error@+1 {{'const.Declare' op symbol does not point to a valid const.Rodata op}}
const.Declare tensor<4x4xf32> = ref<@no_symbol> : tensor<4x4xf32>

// -----

func.func @InvalidParent() -> () {
    // expected-error@+1 {{'const.Data' op expects parent op 'builtin.module'}}
    const.Data @ov_bin {
        const.Rodata @weights dense<0.0> : tensor<2x2xf32>
    }
    return
}
