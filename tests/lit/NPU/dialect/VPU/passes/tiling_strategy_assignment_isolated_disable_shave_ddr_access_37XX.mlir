//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tiling-strategy-assignment="tiling-mode=ISOLATED enable-shave-ddr-access-optimization=false" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: func.func @SplitGatherForLargeIORatio
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<51865x512xf16>
func.func @SplitGatherForLargeIORatio(%arg0: tensor<51865x512xf16>) -> tensor<1x1x512xf16> {
    %cst = const.Declare tensor<1x1xsi32> = dense<1> : tensor<1x1xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<51865x512xf16>, tensor<1x1xsi32> -> tensor<1x1x512xf16>
    return %0 : tensor<1x1x512xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1x1xsi32> = dense<1> : tensor<1x1xsi64>, [#const.CastElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 1, 29]} : tensor<51865x512xf16>, tensor<1x1xsi32> -> tensor<1x1x512xf16>

    // CHECK:     return [[Gather0]]
}

// -----

// CHECK-LABEL: func.func @SplitGather4DForLargeIORatio
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x51865x512xf16>
func.func @SplitGather4DForLargeIORatio(%arg0: tensor<1x1x51865x512xf16>) -> tensor<1x1x1x512xf16> {
    %cst = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x51865x512xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x512xf16>
    return %0 : tensor<1x1x1x512xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi64>, [#const.CastElemType<si32>]
    // CHECK:     [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    // CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64, tilingStrategy = [1, 1, 1, 29]
    // CHECK-SAME:      } : tensor<1x1x51865x512xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x512xf16>

    // CHECK:     return [[GATHER]]
}
