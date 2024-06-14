//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1000000 bytes of @CMX_NN
}

// CHECK-LABEL: @Accumulate
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @Accumulate(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) :
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.Accumulate([[LHS]], [[RHS]], [[LHS_SCALES]], [[RHS_SCALES]]) {
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:  }

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   return [[ACCUMULATE]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @FakeQuantizeAssignedSplitOverHeightInParamPerAxis
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: tensor<1x3x384x640xf16>)
func.func @FakeQuantizeAssignedSplitOverHeightInParamPerAxis(%arg0: tensor<1x3x384x640xf16>) -> tensor<1x3x384x640xf16> {
    %inLow = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+01> : tensor<1x3x1x1xf16>
    %inHigh = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+01> : tensor<1x3x1x1xf16>
    %outLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %outHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x384x640xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x384x640xf16>
    return %fq : tensor<1x3x384x640xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+01> : tensor<1x3x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+01> : tensor<1x3x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    //CHECK: [[FQ:%.+]] = VPU.FakeQuantize([[INPUT_DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    //CHECK-SAME:         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:    tensor<1x3x384x640xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x384x640xf16>
    //CHECK:   return [[FQ]] : tensor<1x3x384x640xf16>
}

// -----

// CHECK-LABEL: @FakeQuantizeAssignedSplitOverKernelOutParamPerAxis
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: tensor<1x128x1x512xf16>)
func.func @FakeQuantizeAssignedSplitOverKernelOutParamPerAxis(%arg0: tensor<1x128x1x512xf16>) -> tensor<1x128x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %inHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    %outLow = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    %outHigh = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x512xf16>
    return %fq : tensor<1x128x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    //CHECK: [[FQ:%.+]] = VPU.FakeQuantize([[INPUT_DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    //CHECK-SAME:         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:   tensor<1x128x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x512xf16>
    //CHECK:   return [[FQ]] : tensor<1x128x1x512xf16>
}

// -----

// CHECK-LABEL: @FakeQuantizeAssignedClustering
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: tensor<1x1x1x512xf16>)
func.func @FakeQuantizeAssignedClustering(%arg0: tensor<1x1x1x512xf16>) -> tensor<1x1x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %inHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    %outLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %outHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, levels = 256 : i64} : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>
    return %fq : tensor<1x1x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    //CHECK: [[FQ:%.+]] = VPU.FakeQuantize([[INPUT_DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    //CHECK-SAME:         {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, levels = 256 : i64,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:   tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>
    //CHECK:   return [[FQ]] : tensor<1x1x1x512xf16>
}

// -----

// CHECK-LABEL: @PadAssignedSplitOverHeight
func.func @PadAssignedSplitOverHeight(%arg0: tensor<1x16x20x50xf16>) -> tensor<1x18x20x60xf16> {

    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 2, 0, 10]} : tensor<1x16x20x50xf16> -> tensor<1x18x20x60xf16>
    return %0 : tensor<1x18x20x60xf16>

    //CHECK:   [[PAD:%.+]] = VPU.Pad({{[^:]+}}) {
    //CHECK-SAME        mode = #IE.pad_mode<EDGE>,
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:       pad_value_attr = 0.000000e+00 : f64,
    //CHECK-SAME:       pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 2, 0, 10]}
    //CHECK-SAME:       tensor<1x16x20x50xf16> -> tensor<1x18x20x60xf16>
    //CHECK:   return [[PAD]] : tensor<1x18x20x60xf16>
}

// -----

// CHECK-LABEL: @PadAssignedSplitOverKernel
func.func @PadAssignedSplitOverKernel(%arg0: tensor<1x16x20x50xf16>) -> tensor<1x16x33x60xf16> {

    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 10], pads_end_attr = [0, 0, 13, 0]} : tensor<1x16x20x50xf16> -> tensor<1x16x33x60xf16>
    return %0 : tensor<1x16x33x60xf16>

    //CHECK:   [[PAD:%.+]] = VPU.Pad({{[^:]+}}) {
    //CHECK-SAME        mode = #IE.pad_mode<EDGE>,
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    //CHECK-SAME:       pad_value_attr = 0.000000e+00 : f64,
    //CHECK-SAME:       pads_begin_attr = [0, 0, 0, 10], pads_end_attr = [0, 0, 13, 0]}
    //CHECK-SAME:       tensor<1x16x20x50xf16> -> tensor<1x16x33x60xf16>
    //CHECK:   return [[PAD]] : tensor<1x16x33x60xf16>
}

// -----

// CHECK-LABEL: @SelectAssignedSplitOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x10x40x40xf16>, [[INPUT0:%.+]]: tensor<1x10x40x40xf16>)
func.func @SelectAssignedSplitOverHeight(%arg0: tensor<1x10x40x40xf16>, %arg1: tensor<1x10x40x40xf16>) -> tensor<1x10x40x40xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16>
    return %0 : tensor<1x10x40x40xf16>

    //CHECK-DAG:    [[INPUT1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    //CHECK:        [[SELECT:%.+]] = VPU.Select([[INPUT]], [[INPUT1]], [[INPUT0]]) {
    //CHECK-SAME:           auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    //CHECK-SAME:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK-SAME:           } : tensor<1x10x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16>
    //CHECK:        return [[SELECT]] : tensor<1x10x40x40xf16>
}

// -----

// CHECK-LABEL: @SelectAssignedSplitOverKernel
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x32x1x40xf16>, [[INPUT0:%.+]]: tensor<1x32x1x40xf16>)
func.func @SelectAssignedSplitOverKernel(%arg0: tensor<1x32x1x40xf16>, %arg1: tensor<1x32x1x40xf16>) -> tensor<1x32x1x40xf16> {
    %cst = const.Declare tensor<1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x40xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x40xf16> -> tensor<1x32x1x40xf16>
    return %0 : tensor<1x32x1x40xf16>

    //CHECK-DAG:    [[INPUT1:%.+]] = const.Declare tensor<1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>
    //CHECK:        [[SELECT:%.+]] = VPU.Select([[INPUT]], [[INPUT1]], [[INPUT0]]) {
    //CHECK-SAME:           auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    //CHECK-SAME:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK-SAME:           } : tensor<1x32x1x40xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x40xf16> -> tensor<1x32x1x40xf16>
    //CHECK:        return [[SELECT]] : tensor<1x32x1x40xf16>
}

// -----

// CHECK-LABEL: @SelectAssignedClustering
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x1x40xf16>, [[INPUT0:%.+]]: tensor<1x1x1x40xf16>)
func.func @SelectAssignedClustering(%arg0: tensor<1x1x1x40xf16>, %arg1: tensor<1x1x1x40xf16>) -> tensor<1x1x1x40xf16> {
    %cst = const.Declare tensor<1x1x1x40xf16> = dense<0.000000e+00> : tensor<1x1x1x40xf16>
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16> -> tensor<1x1x1x40xf16>
    return %0 : tensor<1x1x1x40xf16>

    //CHECK-DAG:    [[INPUT1:%.+]] = const.Declare tensor<1x1x1x40xf16> = dense<0.000000e+00> : tensor<1x1x1x40xf16>
    //CHECK:        [[SELECT:%.+]] = VPU.Select([[INPUT]], [[INPUT1]], [[INPUT0]]) {
    //CHECK-SAME:           auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    //CHECK-SAME:           multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    //CHECK-SAME:           } : tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16> -> tensor<1x1x1x40xf16>
    //CHECK:        return [[SELECT]] : tensor<1x1x1x40xf16>
}
