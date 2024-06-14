//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-fake-quantize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @UnrollHighValues
func.func @UnrollHighValues(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02>
    %IN_HIGH = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02>
    %OUT_LOW = const.Declare tensor<1x2x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_LOW_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %OUT_HIGH = const.Declare tensor<1x2x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_HIGH_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]

    // CHECK:   [[DATA_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 32]
    // CHECK:   [[DATA_1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 16, 32]
    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16> -> tensor<1x2x16x32xf16>

    // CHECK:   [[FQ_0:%.*]] = IE.FakeQuantize([[DATA_0]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_0]], [[OUT_HIGH_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[FQ_1:%.*]] = IE.FakeQuantize([[DATA_1]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_1]], [[OUT_HIGH_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[FQ_0]], [[FQ_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>
    // CHECK-SAME:  } : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16> -> tensor<1x2x16x32xf16>

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[CONCAT]] : tensor<1x2x16x32xf16>
}

// -----

// CHECK-LABEL: @UnrollAllValues
func.func @UnrollAllValues(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x2x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[IN_LOW_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_LOW_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %IN_HIGH = const.Declare tensor<1x2x1x32xf16> = dense<1.270000e+02> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[IN_HIGH_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_HIGH_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %OUT_LOW = const.Declare tensor<1x2x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_LOW_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %OUT_HIGH = const.Declare tensor<1x2x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_HIGH_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]

    // CHECK:   [[DATA_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 32]
    // CHECK:   [[DATA_1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 16, 32]
    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16> -> tensor<1x2x16x32xf16>

    // CHECK:   [[FQ_0:%.*]] = IE.FakeQuantize([[DATA_0]], [[IN_LOW_0]], [[IN_HIGH_0]], [[OUT_LOW_0]], [[OUT_HIGH_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[FQ_1:%.*]] = IE.FakeQuantize([[DATA_1]], [[IN_LOW_1]], [[IN_HIGH_1]], [[OUT_LOW_1]], [[OUT_HIGH_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[FQ_0]], [[FQ_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>
    // CHECK-SAME:  } : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16> -> tensor<1x2x16x32xf16>

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[CONCAT]] : tensor<1x2x16x32xf16>
}

// -----

// CHECK-LABEL: @UnrollLow1x2x1x1
func.func @UnrollLow1x2x1x1(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x2x1x1xf16> = dense<-1.280000e+02> : tensor<1x2x1x1xf16>
    // CHECK-DAG:   [[IN_LOW_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x2x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]
    // CHECK-DAG:   [[IN_LOW_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x2x1x1xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 1]>]
    %IN_HIGH = const.Declare tensor<1x2x1x1xf16> = dense<1.270000e+02> : tensor<1x2x1x1xf16>
    // CHECK-DAG:   [[IN_HIGH_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x2x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]
    // CHECK-DAG:   [[IN_HIGH_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x2x1x1xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 1]>]
    %OUT_LOW = const.Declare tensor<1x2x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_LOW_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %OUT_HIGH = const.Declare tensor<1x2x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_HIGH_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]

    // CHECK:   [[DATA_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 32]
    // CHECK:   [[DATA_1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 16, 32]

    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x2x1x1xf16>,
        tensor<1x2x1x1xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16> -> tensor<1x2x16x32xf16>
    // CHECK:   [[FQ_0:%.*]] = IE.FakeQuantize([[DATA_0]], [[IN_LOW_0]], [[IN_HIGH_0]], [[OUT_LOW_0]], [[OUT_HIGH_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[FQ_1:%.*]] = IE.FakeQuantize([[DATA_1]], [[IN_LOW_1]], [[IN_HIGH_1]], [[OUT_LOW_1]], [[OUT_HIGH_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[FQ_0]], [[FQ_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>
    // CHECK-SAME:  } : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16> -> tensor<1x2x16x32xf16>

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[CONCAT]] : tensor<1x2x16x32xf16>
}

// -----

// CHECK-LABEL: @UnrollThreeAxes
func.func @UnrollThreeAxes(%arg0: tensor<1x2x2x32xf16>) -> tensor<1x2x2x32xf16> {
    %IN_LOW = const.Declare tensor<1x2x2x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>
    // CHECK-DAG:   [[IN_LOW_0_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_LOW_0_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_LOW_1_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_LOW_1_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]

    %IN_HIGH = const.Declare tensor<1x2x2x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>
    // CHECK-DAG:   [[IN_HIGH_0_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_HIGH_0_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_HIGH_1_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_HIGH_1_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]

    %OUT_LOW = const.Declare tensor<1x2x2x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>
    // CHECK-DAG:   [[OUT_LOW_0_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_0_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]

    %OUT_HIGH = const.Declare tensor<1x2x2x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>
    // CHECK-DAG:   [[OUT_HIGH_0_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_0_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 2, 32]>, #const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]

    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x2x32xf16>,
        tensor<1x2x2x32xf16>,
        tensor<1x2x2x32xf16>,
        tensor<1x2x2x32xf16>,
        tensor<1x2x2x32xf16> -> tensor<1x2x2x32xf16>
    // CHECK:   [[DATA_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 2, 32]
    // CHECK:   [[DATA_1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 2, 32]

    // CHECK:   [[DATA_0_0:%.*]] = IE.Slice [[DATA_0]] [0, 0, 0, 0] [1, 1, 1, 32]
    // CHECK:   [[DATA_0_1:%.*]] = IE.Slice [[DATA_0]] [0, 0, 1, 0] [1, 1, 1, 32]

    // CHECK:   [[FQ_0_0:%.*]] = IE.FakeQuantize([[DATA_0_0]], [[IN_LOW_0_0]], [[IN_HIGH_0_0]], [[OUT_LOW_0_0]], [[OUT_HIGH_0_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }
    // CHECK:   [[FQ_0_1:%.*]] = IE.FakeQuantize([[DATA_0_1]], [[IN_LOW_0_1]], [[IN_HIGH_0_1]], [[OUT_LOW_0_1]], [[OUT_HIGH_0_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT_0:%.*]] = IE.Concat([[FQ_0_0]], [[FQ_0_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:  } : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x2x32xf16>

    // CHECK:   [[DATA_1_0:%.*]] = IE.Slice [[DATA_1]] [0, 0, 0, 0] [1, 1, 1, 32]
    // CHECK:   [[DATA_1_1:%.*]] = IE.Slice [[DATA_1]] [0, 0, 1, 0] [1, 1, 1, 32]

    // CHECK:   [[FQ_1_0:%.*]] = IE.FakeQuantize([[DATA_1_0]], [[IN_LOW_1_0]], [[IN_HIGH_1_0]], [[OUT_LOW_1_0]], [[OUT_HIGH_1_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }
    // CHECK:   [[FQ_1_1:%.*]] = IE.FakeQuantize([[DATA_1_1]], [[IN_LOW_1_1]], [[IN_HIGH_1_1]], [[OUT_LOW_1_1]], [[OUT_HIGH_1_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat([[FQ_1_0]], [[FQ_1_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:  } : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x2x32xf16>

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[CONCAT_0]], [[CONCAT_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>
    // CHECK-SAME:  } : tensor<1x1x2x32xf16>, tensor<1x1x2x32xf16> -> tensor<1x2x2x32xf16>

    return %0 : tensor<1x2x2x32xf16>
    // CHECK:   return [[CONCAT]] : tensor<1x2x2x32xf16>
}

// -----

// CHECK-LABEL: @SkipUnrollWithOneAxis
func.func @SkipUnrollWithOneAxis(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x1x1x32xf16>
    // CHECK:  [[IN_LOW:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02>
    %IN_HIGH = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x1x1x32xf16>
    // CHECK:  [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02>
    %OUT_LOW = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x1x1x32xf16>
    // CHECK:  [[OUT_LOW:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01>
    %OUT_HIGH = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x1x1x32xf16>
    // CHECK:  [[OUT_HIGH:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01>

    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x1x1x32xf16>,
        tensor<1x1x1x32xf16>,
        tensor<1x1x1x32xf16>,
        tensor<1x1x1x32xf16> -> tensor<1x2x16x32xf16>
    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[FQ]] : tensor<1x2x16x32xf16>
}

// -----

// CHECK-LABEL: @SkipUnrollWithNoAxes
func.func @SkipUnrollWithNoAxes(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
    // CHECK:  [[IN_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02>
    %IN_HIGH = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    // CHECK:  [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02>
    %OUT_LOW = const.Declare tensor<1x1x1x1xf16> = dense<-6.400000e+01> : tensor<1x1x1x1xf16>
    // CHECK:  [[OUT_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-6.400000e+01>
    %OUT_HIGH = const.Declare tensor<1x1x1x1xf16> = dense<6.400000e+01> : tensor<1x1x1x1xf16>
    // CHECK:  [[OUT_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<6.400000e+01>

    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x1x1x1xf16> -> tensor<1x2x16x32xf16>
    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[FQ]] : tensor<1x2x16x32xf16>
}
