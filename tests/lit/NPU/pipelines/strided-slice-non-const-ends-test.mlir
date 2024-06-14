//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=VPUX37XX --import-IE --set-upper-bounds="3 3" strided-slice-non-const-ends-test.xml -o %t
// RUN: FileCheck %s --input-file %t

#C = affine_map<(d0) -> (d0)>

// CHECK: module @torch_jit {
// CHECK:  IE.CNNNetwork entryPoint : @main inputsInfo : {
// CHECK:    DataInfo "EditOpenVinoIRParameter_0" : tensor<1xsi64>
// CHECK:  } outputsInfo : {
// CHECK:    DataInfo "/_projection/Slice_9" : tensor<9xf32>
// CHECK:  }
// CHECK:  func.func @main([[ARG0:[^:]+]]: tensor<1xsi64>) -> tensor<?xf32, {bounds = [9], order = #C}> {
// CHECK:    [[CONST0:%.*]] = const.Declare tensor<9xf32>
// CHECK:    [[CONST1:%.*]] = const.Declare tensor<1xsi64>
// CHECK:    [[CONST2:%.*]] = const.Declare tensor<1xsi64>
// CHECK:    [[OUT:%.*]] = IE.StridedSlice([[CONST0]], [[CONST1]], [[ARG0]], [[CONST2]]) {begin_mask = [0], ellipsis_mask = [], end_mask = [0], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 1, 1>, shrink_axis_mask = []}
// CHECK:                : tensor<9xf32>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<?xf32, {bounds = [9], order = #C}>
// CHECK:    return [[OUT]]
// CHECK:  }
// CHECK:}
