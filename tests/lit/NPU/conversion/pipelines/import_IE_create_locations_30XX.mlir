//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// This test checks for locations that were created when importing model to IE dialect.
// RUN: vpux-translate --vpu-arch=%arch% --mlir-print-debuginfo --import-IE %data_path_npu%/maxpool_15x15.xml | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK: [[ARG_0:#.+]] = loc("data")
// CHECK: IE.CNNNetwork entryPoint : @main inputsInfo : {
// CHECK: DataInfo "data" : tensor<1x1x15x15xf32>
// CHECK: } outputsInfo : {
// CHECK: DataInfo "pool1" : tensor<1x1x14x14xf32>

// CHECK: func.func @main(%arg0: tensor<1x1x15x15xf32> loc(fused<{name = "data", type = "Parameter"}>[[[ARG_0]]]))

// CHECK: IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x15x15xf32> -> tensor<1x1x15x15xf16> loc([[CONVERT_1:#.+]])
// CHECK: IE.MaxPool(%0) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x15x15xf16> -> tensor<1x1x14x14xf16> loc([[MAXPOOL:#.+]])
// CHECK: IE.Convert(%1) {dstElemType = f32} : tensor<1x1x14x14xf16> -> tensor<1x1x14x14xf32> loc([[CONVERT_2:#.+]])
// CHECK: return %2 : tensor<1x1x14x14xf32> loc([[OUTPUT:#.+]])

// CHECK: [[CONVERT_1]] = loc(fused<{name = "Convert_8", type = "Convert"}>
// CHECK: [[MAXPOOL]] = loc(fused<{name = "MaxPool_4", type = "MaxPool"}>
// CHECK: [[CONVERT_2]] = loc(fused<{name = "pool1", type = "Convert"}>
// CHECK: [[OUTPUT]] = loc(fused<{name = "output", type = "Output"}>
