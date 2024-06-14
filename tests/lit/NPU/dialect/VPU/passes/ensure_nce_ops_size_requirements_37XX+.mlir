//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEAveragePoolOverOW
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x3x8832xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverOW(%arg0: tensor<1x16x3x8832xf16, {order = #NHWC}>) -> tensor<1x16x1x8832xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [3, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<clamp_high = 2147483647 : i64,
               clamp_low = -2147483648 : i64,
               fp_prelu_alpha = 1.000000e+00 : f64,
               lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = <NOOP>,
               quant_scale = [0.33333333333333331]>,
               strides = [1, 1]}
        -> tensor<1x16x1x8832xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x8832xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 3, 4416]
    // CHECK-SAME:      : tensor<1x16x3x8832xf16, {order = #NHWC}> to tensor<1x16x3x4416xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[ACTIVATION_TILE_0]]) {kernel_size = [3, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:              quant_scale = [0.33333333333333331],
    // CHECK-SAME:              fp_prelu_alpha = 1.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x1x4416xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 4416] [1, 16, 3, 4416]
    // CHECK-SAME:      : tensor<1x16x3x8832xf16, {order = #NHWC}> to tensor<1x16x3x4416xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[ACTIVATION_TILE_1]]) {kernel_size = [3, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:              quant_scale = [0.33333333333333331],
    // CHECK-SAME:              fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x1x4416xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 4416]
    // CHECK-SAME:          -> tensor<1x16x1x8832xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x8832xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverIC3Convs
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3Convs(%arg0: tensor<1x16640x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<512x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [512, 16640, 1, 1],
    strides = [1, 1]
  } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<512x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [512, 5536, 1, 1]>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [512, 5552, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [512, 5552, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE0:%.+]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [512, 5536, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <ADD>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]>} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0:%.+]], [[CONV_OUT2:%.+]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverIC3ConvsWithOutNCHW
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3ConvsWithOutNCHW(%arg0: tensor<1x16640x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NCHW}> {
  %weights = const.Declare tensor<512x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [512, 16640, 1, 1],
    strides = [1, 1]
  } -> tensor<1x512x4x1xf16, {order = #NCHW}>

  return %0 : tensor<1x512x4x1xf16, {order = #NCHW}>


  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<512x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [512, 5536, 1, 1]>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<512x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE2:%.+]])
  // CHECK-SAME:        {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [512, 5552, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]])
  // CHECK-SAME:        {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [512, 5552, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]])
  // CHECK-SAME:        {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [512, 5536, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <ADD>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,  lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]>}
  // CHECK-SAME:        -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0:%.+]], [[CONV_OUT2:%.+]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
  // CHECK-SAME:        -> tensor<1x512x4x1xf16, {order = #NCHW}>
  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x512x4x1xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverICandOC
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverICandOC(%arg0: tensor<1x16640x4x1xf16, {order = #NHWC}>) -> tensor<1x9216x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<9216x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<9216x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [9216, 16640, 1, 1],
    strides = [1, 1]
  } -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:   [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER0:%.+]] = const.Declare tensor<4608x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[4608, 11104, 0, 0], [4608, 5536, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER1:%.+]] = const.Declare tensor<4608x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [4608, 5536, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER2:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[4608, 5552, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE3:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER3:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE4:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER4:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[4608, 0, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE5:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER5:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [4608, 5552, 1, 1]>]

  // CHECK:       [[INPUT_SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER5:%.+]], [[WEIGHTS_TABLE5:%.+]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER4:%.+]], [[WEIGHTS_TABLE4:%.+]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT0:%.+]] = VPU.Concat([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE1:%.+]] = VPU.Slice %arg0 [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER3:%.+]], [[WEIGHTS_TABLE3:%.+]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT3:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE2:%.+]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT1:%.+]] = VPU.Concat([[CONV_OUT2:%.+]], [[CONV_OUT3:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE2:%.+]] = VPU.Slice %arg0 [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT4:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [4608, 5536, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT5:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [4608, 5536, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT2:%.+]] = VPU.Concat([[CONV_OUT4:%.+]], [[CONV_OUT5:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE3:%.+]] = VPU.Slice [[CONCAT_OUT0:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE4:%.+]] = VPU.Slice [[CONCAT_OUT1:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE3:%.+]], [[INPUT_SLICE4:%.+]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <ADD>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]>} -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE5:%.+]] = VPU.Slice [[CONCAT_OUT0:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE6:%.+]] = VPU.Slice [[CONCAT_OUT1:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE5:%.+]], [[INPUT_SLICE6:%.+]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <ADD>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]>} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT3:%.+]] = VPU.Concat([[ADD_OUT0:%.+]], [[ADD_OUT1:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE7:%.+]] = VPU.Slice [[CONCAT_OUT3:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE8:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT2:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE7:%.+]], [[INPUT_SLICE8:%.+]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE9:%.+]] = VPU.Slice [[CONCAT_OUT3:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE10:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[ADD_OUT3:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE9:%.+]], [[INPUT_SLICE10:%.+]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT4:%.+]] = VPU.Concat([[ADD_OUT2:%.+]], [[ADD_OUT3:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       return [[CONCAT_OUT4:%.+]] : tensor<1x9216x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 0.007874015733307484>

// CHECK-LABEL:   @SplitNCEConvOverIC3ConvsMixedPrecision
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x1x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3ConvsMixedPrecision(%arg0: tensor<1x16640x1x1xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<16x16640x1x1x!qElemType, {order = #NHWC}> = dense<64.0> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>]
  %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [16, 16640, 1, 1],
    strides = [1, 1]
  } -> tensor<1x16x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<16x5536x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.SubView<[0, 11104, 0, 0], [16, 5536, 1, 1]>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<16x5552x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.SubView<[0, 5552, 0, 0], [16, 5552, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<16x5552x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.SubView<[0, 0, 0, 0], [16, 5552, 1, 1]>]

  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 10, 0]]], [[[5536, 704, 10, 0]]], [[[11072, 1408, 10, 0]]], [[[16608, 2112, 10, 0]]], [[[22144, 2816, 10, 0]]], [[[27680, 3520, 10, 0]]], [[[33216, 4224, 10, 0]]], [[[38752, 4928, 10, 0]]], [[[44288, 5632, 10, 0]]], [[[49824, 6336, 10, 0]]], [[[55360, 7040, 10, 0]]], [[[60896, 7744, 10, 0]]], [[[66432, 8448, 10, 0]]], [[[71968, 9152, 10, 0]]], [[[77504, 9856, 10, 0]]], [[[83040, 10560, 10, 0]]]]> : tensor<16x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 10, 0]]], [[[5552, 704, 10, 0]]], [[[11104, 1408, 10, 0]]], [[[16656, 2112, 10, 0]]], [[[22208, 2816, 10, 0]]], [[[27760, 3520, 10, 0]]], [[[33312, 4224, 10, 0]]], [[[38864, 4928, 10, 0]]], [[[44416, 5632, 10, 0]]], [[[49968, 6336, 10, 0]]], [[[55520, 7040, 10, 0]]], [[[61072, 7744, 10, 0]]], [[[66624, 8448, 10, 0]]], [[[72176, 9152, 10, 0]]], [[[77728, 9856, 10, 0]]], [[[83280, 10560, 10, 0]]]]> : tensor<16x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 10, 10]]], [[[5552, 704, 10, 10]]], [[[11104, 1408, 10, 10]]], [[[16656, 2112, 10, 10]]], [[[22208, 2816, 10, 10]]], [[[27760, 3520, 10, 10]]], [[[33312, 4224, 10, 10]]], [[[38864, 4928, 10, 10]]], [[[44416, 5632, 10, 10]]], [[[49968, 6336, 10, 10]]], [[[55520, 7040, 10, 10]]], [[[61072, 7744, 10, 10]]], [[[66624, 8448, 10, 10]]], [[[72176, 9152, 10, 10]]], [[[77728, 9856, 10, 10]]], [[[83280, 10560, 10, 10]]]]> : tensor<16x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 1, 1] : tensor<1x16640x1x1xf16, {order = #NHWC}> to tensor<1x5552x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]]  = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[WEIGHTS_TABLE2]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 5552, 1, 1], strides = [1, 1]} -> tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 1, 1] : tensor<1x16640x1x1xf16, {order = #NHWC}> to tensor<1x5552x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE1]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 5552, 1, 1], strides = [1, 1]} -> tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 1, 1] : tensor<1x16640x1x1xf16, {order = #NHWC}> to tensor<1x5536x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]]  = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 5536, 1, 1], strides = [1, 1]} -> tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <ADD>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]>} -> tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:       return [[ADD_OUT1]] : tensor<1x16x1x1xf16, {order = #NHWC}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 0.007874015733307484>

// CHECK-LABEL:   @SplitNCEConvOverIC3ConvsMixedPrecisionI4
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16416x1x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3ConvsMixedPrecisionI4(%arg0: tensor<1x16416x1x1xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<16x16416x1x1x!qElemType, {order = #NHWC}> = dense<64.0> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si4>, #const.QuantCast<!qElemType>]
  %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [16, 16416, 1, 1],
    strides = [1, 1]
  } -> tensor<1x16x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>
// CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si4>, #const.QuantCast<!qElemType>, #const.SubView<[0, 10944, 0, 0], [16, 5472, 1, 1]>]
// CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si4>, #const.QuantCast<!qElemType>, #const.SubView<[0, 5472, 0, 0], [16, 5472, 1, 1]>]
// CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si4>, #const.QuantCast<!qElemType>, #const.SubView<[0, 0, 0, 0], [16, 5472, 1, 1]>]
// CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32>
// CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 10, 0]]], [[[2736, 352, 10, 0]]], [[[5472, 704, 10, 0]]], [[[8208, 1056, 10, 0]]], [[[10944, 1408, 10, 0]]], [[[13680, 1760, 10, 0]]], [[[16416, 2112, 10, 0]]], [[[19152, 2464, 10, 0]]], [[[21888, 2816, 10, 0]]], [[[24624, 3168, 10, 0]]], [[[27360, 3520, 10, 0]]], [[[30096, 3872, 10, 0]]], [[[32832, 4224, 10, 0]]], [[[35568, 4576, 10, 0]]], [[[38304, 4928, 10, 0]]], [[[41040, 5280, 10, 0]]]]> : tensor<16x1x1x4xsi32>
// CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32>
// CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 10, 10]]], [[[2736, 352, 10, 10]]], [[[5472, 704, 10, 10]]], [[[8208, 1056, 10, 10]]], [[[10944, 1408, 10, 10]]], [[[13680, 1760, 10, 10]]], [[[16416, 2112, 10, 10]]], [[[19152, 2464, 10, 10]]], [[[21888, 2816, 10, 10]]], [[[24624, 3168, 10, 10]]], [[[27360, 3520, 10, 10]]], [[[30096, 3872, 10, 10]]], [[[32832, 4224, 10, 10]]], [[[35568, 4576, 10, 10]]], [[[38304, 4928, 10, 10]]], [[[41040, 5280, 10, 10]]]]> : tensor<16x1x1x4xsi32>
// CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
// CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[WEIGHTS_TABLE1]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 5472, 1, 1], strides = [1, 1]} -> tensor<1x16x1x1xf16, {order = #NHWC}>
// CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5472, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
// CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE0]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 5472, 1, 1], strides = [1, 1]} -> tensor<1x16x1x1xf16, {order = #NHWC}>
// CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 10944, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
// CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 5472, 1, 1], strides = [1, 1]} -> tensor<1x16x1x1xf16, {order = #NHWC}>
// CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <ADD>
// CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>
// CHECK:      return [[ADD_OUT1]] : tensor<1x16x1x1xf16, {order = #NHWC}>
}
