//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN:vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @ConvertConstToAttrVPUX30XX
func.func @ConvertConstToAttrVPUX30XX(%arg0: tensor<5x6xf16>) -> tensor<2x6xf16> {
    %cst = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    %cst_1 = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %0 = IE.EmbeddingBagOffsetsSum(%arg0, %cst, %cst_0, %cst_1) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>} : tensor<5x6xf16>, tensor<5xsi32>, tensor<2xsi32>, tensor<si32> -> tensor<2x6xf16>
    return %0 : tensor<2x6xf16>

    // CHECK: [[VAR0:%.+]] = IE.EmbeddingBagOffsetsSum(%arg0) {default_index_value = 0 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, per_sample_weights_value = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<5x6xf16> -> tensor<2x6xf16>
    // CHECK: return [[VAR0]] : tensor<2x6xf16>
}
