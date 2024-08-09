//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: ParseAndPrintSimple

{-#
  dialect_resources: {
    builtin: {
      blob: "0x04000000010000000200000003000000"
    }
  }
#-}

const.Data @ParseAndPrintSimple {
    const.Rodata @weights_0 dense<1.000000e+00> : tensor<4x4xf32>
    const.Rodata @weights_1 dense_resource<blob> : tensor<2x2xf32>
}

func.func @ParseAndPrintSimpleFunc() -> tensor<2x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<0.0> : tensor<2x2xf32>
    return %cst : tensor<2x2xf32>
}

// CHECK: const.Rodata @weights_0 dense<1.000000e+00> : tensor<4x4xf32>
// CHECK: const.Rodata @weights_1 dense_resource<blob> : tensor<2x2xf32>
// CHECK: }

// -----

// CHECK-LABEL: ParseAndPrintConstSymbol

const.Data @ParseAndPrintConstSymbol {
    const.Rodata @weights_0 dense<1.000000e+00> : tensor<4x4xf32>
}

func.func @ParseAndPrintConstSymbolFunc() -> tensor<4x4xf32> {
    %cst = const.Declare tensor<4x4xf32> = ref<@ParseAndPrintConstSymbol::@weights_0> : tensor<4x4xf32>
    return %cst : tensor<4x4xf32>
}

// CHECK: const.Rodata @weights_0 dense<1.000000e+00> : tensor<4x4xf32>
// CHECK: const.Declare tensor<4x4xf32> = ref<@ParseAndPrintConstSymbol::@weights_0> : tensor<4x4xf32>
