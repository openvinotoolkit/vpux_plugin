//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dynamic-shape-transformations %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
!BoundedType = tensor<?x1x548xf16, {bounds = [32, 1, 548], order = #CHW}>

// CHECK-LABEL: @DynamicSoftmax
// CHECK-SAME: [[IN:%.+]]: tensor<?x1x548xf16, {bounds = [32, 1, 548], order = #CHW}>
func.func @DynamicSoftmax(%arg0: !BoundedType) -> !BoundedType {
    %0 = IE.SoftMax(%arg0) {axisInd = 2 : i64} : !BoundedType -> !BoundedType
    return %0 : !BoundedType

    // CHECK-DAG:   [[DIM_2:%.+]] = const.Declare tensor<1xsi64> = dense<548> : tensor<1xsi64>
    // CHECK-DAG:   [[DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[IN]]) {axisInd = 2 : i64}
    // CHECK:       [[SHAPE_OF:%.+]] = IE.ShapeOf([[IN]])
    // CHECK-SAME:      -> tensor<3xsi64>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [0] [1]
    // CHECK-SAME:      to tensor<1xsi64>
    // CHECK:       [[NEW_SHAPE:%.+]] = IE.Concat([[SLICE]], [[DIM_1]], [[DIM_2]])
    // CHECK-SAME:      -> tensor<3xsi64>
    // CHECK:       [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[SOFTMAX]], [[NEW_SHAPE]])

    // CHECK:       return [[DYN_RESHAPE]]
}
