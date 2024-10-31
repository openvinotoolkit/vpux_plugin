//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

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
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: tensor<1x1x384x640xf16>)
func.func @FakeQuantizeAssignedSplitOverHeightInParamPerAxis(%arg0: tensor<1x1x384x640xf16>) -> tensor<1x1x384x640xf16> {
    %inLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %inHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    %outLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %outHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x384x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x384x640xf16>
    return %fq : tensor<1x1x384x640xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    //CHECK: [[FQ:%.+]] = VPU.FakeQuantize([[INPUT_DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    //CHECK-SAME:         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:    tensor<1x1x384x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x384x640xf16>
    //CHECK:   return [[FQ]] : tensor<1x1x384x640xf16>
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Mvn1NormAssignedSOH
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x256x256x256xf16, {order = #NHWC}>, [[MEAN_VAR:%.+]]: tensor<1x256x1x2xf16, {order = #NHWC}>)
func.func @Mvn1NormAssignedSOH(%arg0: tensor<1x256x256x256xf16, {order = #NHWC}>, %arg1: tensor<1x256x1x2xf16, {order = #NHWC}>) -> tensor<1x256x256x256xf16, {order = #NHWC}> {
   %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<1x256x1x2xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
   return %0 : tensor<1x256x256x256xf16, {order = #NHWC}>

   // CHECK:       [[OUT:%.*]] = VPU.MVN1Normalize([[INPUT]], [[MEAN_VAR]])
   // CHECK-SAME :     {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true} :
   // CHECK-SAME :     tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<1x256x1x2xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
   // CHECK:       return [[OUT]] : tensor<1x256x256x256xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SelectAssignedSplitOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x40x40xf16>, [[INPUT0:%.+]]: tensor<1x1x40x40xf16>)
func.func @SelectAssignedSplitOverHeight(%arg0: tensor<1x1x40x40xf16>, %arg1: tensor<1x1x40x40xf16>) -> tensor<1x1x40x40xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x1x40x40xf16> -> tensor<1x1x40x40xf16>
    return %0 : tensor<1x1x40x40xf16>

    //CHECK-DAG:    [[INPUT1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    //CHECK:        [[SELECT:%.+]] = VPU.Select([[INPUT]], [[INPUT1]], [[INPUT0]]) {
    //CHECK-SAME:           auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    //CHECK-SAME:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK-SAME:           } : tensor<1x1x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x1x40x40xf16> -> tensor<1x1x40x40xf16>
    //CHECK:        return [[SELECT]] : tensor<1x1x40x40xf16>
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

// CHECK-LABEL: @SelectAssignedSplitOverWidth
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x1x40xf16>, [[INPUT0:%.+]]: tensor<1x1x1x40xf16>)
func.func @SelectAssignedSplitOverWidth(%arg0: tensor<1x1x1x40xf16>, %arg1: tensor<1x1x1x40xf16>) -> tensor<1x1x1x40xf16> {
    %cst = const.Declare tensor<1x1x1x40xf16> = dense<0.000000e+00> : tensor<1x1x1x40xf16>
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16> -> tensor<1x1x1x40xf16>
    return %0 : tensor<1x1x1x40xf16>

    //CHECK-DAG:    [[INPUT1:%.+]] = const.Declare tensor<1x1x1x40xf16> = dense<0.000000e+00> : tensor<1x1x1x40xf16>
    //CHECK:        [[SELECT:%.+]] = VPU.Select([[INPUT]], [[INPUT1]], [[INPUT0]]) {
    //CHECK-SAME:           auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    //CHECK-SAME:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>
    //CHECK-SAME:           } : tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16> -> tensor<1x1x1x40xf16>
    //CHECK:        return [[SELECT]] : tensor<1x1x1x40xf16>
}

// -----

// CHECK-LABEL:   @LSTMGatesAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x100x2048xf16>, [[INPUT_1:%.+]]: tensor<1x1x100x512xf16>
func.func @LSTMGatesAssignedSplitOverHeight(%arg0: tensor<1x1x100x2048xf16>, %arg1: tensor<1x1x100x512xf16>) -> (tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>) {
    %0, %1 = VPU.LSTMGates(%arg0, %arg1) : tensor<1x1x100x2048xf16>, tensor<1x1x100x512xf16> -> tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>

    return %0, %1 : tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>

    //CHECK:   [[LSTMGATES_0:%.+]], [[LSTMGATES_1:%.+]] = VPU.LSTMGates([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x100x2048xf16>, tensor<1x1x100x512xf16> -> tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>
    //CHECK:   return [[LSTMGATES_0]], [[LSTMGATES_1]] : tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>
}

// -----

// CHECK-LABEL:   @LSTMGatesAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x1x2048xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x512xf16>
func.func @LSTMGatesAssignedClustering(%arg0: tensor<1x1x1x2048xf16>, %arg1: tensor<1x1x1x512xf16>) -> (tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>) {
    %0, %1 = VPU.LSTMGates(%arg0, %arg1) : tensor<1x1x1x2048xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>

    return %0, %1 : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>

    //CHECK:   [[LSTMGATES_0:%.+]], [[LSTMGATES_1:%.+]] = VPU.LSTMGates([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<1x1x1x2048xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>
    //CHECK:   return [[LSTMGATES_0]], [[LSTMGATES_1]] : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>
}

// -----

// CHECK-LABEL:   @AndAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @AndAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[And:%.+]] = VPU.And([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[And]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @AndAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @AndAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xf16> {

    %0 = VPU.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[And:%.+]] = VPU.And([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>
    //CHECK:   return [[And]] : tensor<1x1x256x256xf16>
}

// -----

// CHECK-LABEL:   @AndAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @AndAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xf16> {

    %0 = VPU.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>

    return %0 : tensor<2x106x1x256xf16>

    //CHECK:   [[And:%.+]] = VPU.And([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>
    //CHECK:   return [[And]] : tensor<2x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @SinAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x106x1x256xf16>
func.func @SinAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>) -> tensor<1x106x1x256xf16> {
    %0 = VPU.Sin(%arg0) : tensor<1x106x1x256xf16> -> tensor<1x106x1x256xf16>
    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[SIN:%.+]] = VPU.Sin([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

// CHECK-LABEL:   @SinAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x256x256xf16>
func.func @SinAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256xf16> {
    %0 = VPU.Sin(%arg0) : tensor<1x1x256x256xf16> -> tensor<1x1x256x256xf16>
    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[SIN:%.+]] = VPU.Sin([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

// CHECK-LABEL:   @SinAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x256xf16>
func.func @SinAssignedClustering(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16> {
    %0 = VPU.Sin(%arg0) : tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16>
    return %0 : tensor<1x1x1x256xf16>

    //CHECK:   [[SIN:%.+]] = VPU.Sin([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}

// -----

// CHECK-LABEL:   @CosAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x106x1x256xf16>
func.func @CosAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>) -> tensor<1x106x1x256xf16> {
    %0 = VPU.Cos(%arg0) : tensor<1x106x1x256xf16> -> tensor<1x106x1x256xf16>
    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[COS:%.+]] = VPU.Cos([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

// CHECK-LABEL:   @CosAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x256x256xf16>
func.func @CosAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256xf16> {
    %0 = VPU.Cos(%arg0) : tensor<1x1x256x256xf16> -> tensor<1x1x256x256xf16>
    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[COS:%.+]] = VPU.Cos([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

// CHECK-LABEL:   @CosAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x256xf16>
func.func @CosAssignedClustering(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16> {
    %0 = VPU.Cos(%arg0) : tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16>
    return %0 : tensor<1x1x1x256xf16>

    //CHECK:   [[COS:%.+]] = VPU.Cos([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}

// -----

// CHECK-LABEL:   @ExpAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x106x1x256xf16>
func.func @ExpAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>) -> tensor<1x106x1x256xf16> {
    %0 = VPU.Exp(%arg0) : tensor<1x106x1x256xf16> -> tensor<1x106x1x256xf16>
    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[EXP:%.+]] = VPU.Exp([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

// CHECK-LABEL:   @ExpAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x256x256xf16>
func.func @ExpAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256xf16> {
    %0 = VPU.Exp(%arg0) : tensor<1x1x256x256xf16> -> tensor<1x1x256x256xf16>
    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[EXP:%.+]] = VPU.Exp([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

// CHECK-LABEL:   @ExpAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x256xf16>
func.func @ExpAssignedClustering(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16> {
    %0 = VPU.Exp(%arg0) : tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16>
    return %0 : tensor<1x1x1x256xf16>

    //CHECK:   [[EXP:%.+]] = VPU.Exp([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}

// -----

// CHECK-LABEL:   @SwishAssignedSplitOverWidth
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x256xf16>
func.func @SwishAssignedSplitOverWidth(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16> {
    %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64}: tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16>
    return %0 : tensor<1x1x1x256xf16>

    //CHECK:   [[SWISH:%.+]] = VPU.Swish([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>}
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.056323952768363203:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SoftMaxAssignedSplitOverHeightAndBigConvolutionToo
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
func.func @SoftMaxAssignedSplitOverHeightAndBigConvolutionToo(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<4096x48x1x1x!qElemType, {order = #NHWC}> = dense<128> : tensor<4096x48x1x1xui8, {order = #NHWC}>
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %0 = VPU.NCE.Convolution(%arg0, %cst, %cst_0) {opaque_ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    %1 = VPU.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    return %1 : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    //CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]],
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:       [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV]])
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

// CHECK-LABEL:   @GatherAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x192x4x1xf16>
func.func @GatherAssignedSplitOverKernel(%arg0: tensor<1x192x4x1xf16>) -> tensor<1x192x8x1xf16> {
    %cst = const.Declare tensor<1x8x1x1xsi32> = dense<1> : tensor<1x8x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x192x4x1xf16>, tensor<1x8x1x1xsi32> -> tensor<1x192x8x1xf16>
    return %0 : tensor<1x192x8x1xf16>

    // CHECK-DAG:   [[INDICES:%.+]] = const.Declare tensor<1x8x1x1xsi32> = dense<1> : tensor<1x8x1x1xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

// CHECK-LABEL:   @GatherAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x64x72xf16>
func.func @GatherAssignedSplitOverHeight(%arg0: tensor<1x1x64x72xf16>) -> tensor<1x1x16x72xf16> {
    %cst = const.Declare tensor<1x16x1x1xsi32> = dense<1> : tensor<1x16x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x64x72xf16>, tensor<1x16x1x1xsi32> -> tensor<1x1x16x72xf16>
    return %0 : tensor<1x1x16x72xf16>

    // CHECK-DAG:   [[INDICES:%.+]] = const.Declare tensor<1x16x1x1xsi32> = dense<1> : tensor<1x16x1x1xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

// CHECK-LABEL:   @GatherAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x64x54xf16>
func.func @GatherAssignedClustering(%arg0: tensor<1x1x64x54xf16>) -> tensor<1x1x1x54xf16> {
    %cst = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x64x54xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x54xf16>
    return %0 : tensor<1x1x1x54xf16>

    // CHECK-DAG:   [[INDICES:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}
