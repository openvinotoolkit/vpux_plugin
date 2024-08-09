//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-upsampling-to-strided-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: @ConvertUpsamplingToStridedConcatW
func.func @ConvertUpsamplingToStridedConcatW(%arg0: tensor<1x1x2x2xf16>) -> tensor<1x1x2x3xf16> {
    %0 = IE.Upsampling(%arg0) {
        pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 0]>,
        upsampling_factor = [2, 1, 1]
    } : tensor<1x1x2x2xf16> -> tensor<1x1x2x3xf16>
    return %0 : tensor<1x1x2x3xf16>

    // CHECK-NOT:   IE.Upsampling
    // CHECK-DAG:       [[CST:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x1x2x2xf16>
    // CHECK-SAME:      dense<0.000000e+00>
    // CHECK:       [[CONCAT:%.*]] = IE.Concat(%arg0, [[CST]])
    // CHECK-SAME:      axis = 3
    // CHECK-SAME:      stride = 2
    // CHECK-SAME:      tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x2x4xf16>
    // CHECK:       [[SLICE:%.*]] =  IE.Slice [[CONCAT]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 1, 2, 3]
    // CHECK-SAME:      tensor<1x1x2x4xf16> to tensor<1x1x2x3xf16>
    // CHECK:       return [[SLICE]]
}

// CHECK-LABEL: @ConvertUpsamplingToStridedConcatH
func.func @ConvertUpsamplingToStridedConcatH(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x4x2xf32> {
    %0 = IE.Upsampling(%arg0) {
        pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 0]>,
        upsampling_factor = [1, 3, 1]
    } : tensor<1x1x2x2xf32> -> tensor<1x1x4x2xf32>
    return %0 : tensor<1x1x4x2xf32>

    // CHECK-NOT:   IE.Upsampling
    // CHECK-DAG:       [[CST:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x1x2x2xf32>
    // CHECK:       [[CONCAT:%.*]] = IE.Concat(%arg0, [[CST]], [[CST]])
    // CHECK-SAME:      axis = 2
    // CHECK-SAME:      stride = 3
    // CHECK-SAME:      tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x6x2xf32>
    // CHECK:       [[SLICE:%.*]] =  IE.Slice [[CONCAT]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 1, 4, 2]
    // CHECK-SAME:      tensor<1x1x6x2xf32> to tensor<1x1x4x2xf32>
    // CHECK:       return [[SLICE]]
}

// CHECK-LABEL: @ConvertUpsamplingToStridedConcatC
func.func @ConvertUpsamplingToStridedConcatC(%arg0: tensor<1x2x2x2xf16>) -> tensor<1x3x2x2xf16> {
    %0 = IE.Upsampling(%arg0) {
        pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 0]>,
        upsampling_factor = [1, 1, 2]
    } : tensor<1x2x2x2xf16> -> tensor<1x3x2x2xf16>
    return %0 : tensor<1x3x2x2xf16>

    // CHECK-NOT:   IE.Upsampling
    // CHECK-DAG:       [[CST:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x2x2x2xf16>
    // CHECK:       [[CONCAT:%.*]] = IE.Concat(%arg0, [[CST]])
    // CHECK-SAME:      axis = 1
    // CHECK-SAME:      stride = 2
    // CHECK-SAME:      tensor<1x2x2x2xf16>, tensor<1x2x2x2xf16> -> tensor<1x4x2x2xf16>
    // CHECK:       [[SLICE:%.*]] =  IE.Slice [[CONCAT]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 3, 2, 2]
    // CHECK-SAME:      tensor<1x4x2x2xf16> to tensor<1x3x2x2xf16>
    // CHECK:       return [[SLICE]]
}

// CHECK-LABEL: @ConvertUpsamplingToStridedConcatHWC
func.func @ConvertUpsamplingToStridedConcatHWC(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x4x4x4xf32> {
    %0 = IE.Upsampling(%arg0) {
        pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 0]>,
        upsampling_factor = [3, 3, 3]
    } : tensor<1x2x2x2xf32> -> tensor<1x4x4x4xf32>
    return %0 : tensor<1x4x4x4xf32>

    // CHECK-NOT:   IE.Upsampling
    // CHECK-DAG:       [[CST_C:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x2x2x2xf32>
    // CHECK:       [[CONCAT_C:%.*]] = IE.Concat(%arg0, [[CST_C]], [[CST_C]])
    // CHECK-SAME:      axis = 1
    // CHECK-SAME:      stride = 3
    // CHECK-SAME:      tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32> -> tensor<1x6x2x2xf32>
    // CHECK-DAG:       [[CST_H:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x6x2x2xf32>
    // CHECK:       [[CONCAT_H:%.*]] = IE.Concat([[CONCAT_C]], [[CST_H]], [[CST_H]])
    // CHECK-SAME:      axis = 2
    // CHECK-SAME:      stride = 3
    // CHECK-SAME:      tensor<1x6x2x2xf32>, tensor<1x6x2x2xf32>, tensor<1x6x2x2xf32> -> tensor<1x6x6x2xf32>
    // CHECK-DAG:       [[CST_W:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x6x6x2xf32>
    // CHECK:       [[CONCAT_W:%.*]] = IE.Concat([[CONCAT_H]], [[CST_W]], [[CST_W]])
    // CHECK-SAME:      axis = 3
    // CHECK-SAME:      stride = 3
    // CHECK-SAME:      tensor<1x6x6x2xf32>, tensor<1x6x6x2xf32> -> tensor<1x6x6x6xf32>
    // CHECK:       [[SLICE:%.*]] =  IE.Slice [[CONCAT_W]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 4, 4]
    // CHECK-SAME:      tensor<1x6x6x6xf32> to tensor<1x4x4x4xf32>
    // CHECK:       return [[SLICE]]
}

// CHECK-LABEL: @ConvertUpsamplingToStridedConcatPadWH
func.func @ConvertUpsamplingToStridedConcatPadWH(%arg0: tensor<1x1x2x2xf16>) -> tensor<1x1x4x4xf16> {
    %0 = IE.Upsampling(%arg0) {
        pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [1, 0], pads_width = [1, 0]>,
        upsampling_factor = [2, 2, 1]
    } : tensor<1x1x2x2xf16> -> tensor<1x1x4x4xf16>
    return %0 : tensor<1x1x4x4xf16>

    // CHECK-NOT:   IE.Upsampling
    // CHECK-DAG:       [[CST_H:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x1x2x2xf16>
    // CHECK:       [[CONCAT_H:%.*]] = IE.Concat(%arg0, [[CST_H]])
    // CHECK-SAME:      axis = 2
    // CHECK-SAME:      stride = 2
    // CHECK-SAME:      tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x4x2xf16>
    // CHECK-DAG:       [[CST_W:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x1x4x2xf16>
    // CHECK:       [[CONCAT_W:%.*]] = IE.Concat([[CONCAT_H]], [[CST_W]])
    // CHECK-SAME:      axis = 3
    // CHECK-SAME:      stride = 2
    // CHECK-SAME:      tensor<1x1x4x2xf16>, tensor<1x1x4x2xf16> -> tensor<1x1x4x4xf16>
    // CHECK:       [[SLICE:%.*]] =  IE.Slice [[CONCAT_W]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 1, 3, 3]
    // CHECK-SAME:      tensor<1x1x4x4xf16> to tensor<1x1x3x3xf16>
    // CHECK:       [[PADING:%.*]] = IE.Pad([[SLICE]])
    // CHECK-SAME:      mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:      pad_value_attr = 0.000000e+00 : f64
    // CHECK-SAME:      pads_begin_attr = [0, 0, 1, 1]
    // CHECK-SAME:      pads_end_attr = [0, 0, 0, 0]}
    // CHECK-SAME:      tensor<1x1x3x3xf16> -> tensor<1x1x4x4xf16>
    // CHECK:       return [[PADING]]
}

// CHECK-LABEL: @ConvertDMAbleUpsamplingToStridedConcatPadWH
func.func @ConvertDMAbleUpsamplingToStridedConcatPadWH(%arg0: tensor<1x1x2x2xf16>) -> tensor<1x1x5x5xf16> {
    %0 = IE.Upsampling(%arg0) {
        pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [1, 1], pads_width = [1, 1]>,
        upsampling_factor = [2, 2, 1]
    } : tensor<1x1x2x2xf16> -> tensor<1x1x5x5xf16>
    return %0 : tensor<1x1x5x5xf16>

    // CHECK:       [[DMABLEUPSAMPLING:%.*]] = IE.Upsampling(%arg0)
    // CHECK-SAME:      #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>,
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]} :
    // CHECK-SAME:      tensor<1x1x2x2xf16> -> tensor<1x1x4x4xf16>
    // CHECK:       [[PADING:%.*]] = IE.Pad([[DMABLEUPSAMPLING]])
    // CHECK-SAME:      mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:      pad_value_attr = 0.000000e+00 : f64
    // CHECK-SAME:      pads_begin_attr = [0, 0, 1, 1]
    // CHECK-SAME:      pads_end_attr = [0, 0, 0, 0]}
    // CHECK-SAME:      tensor<1x1x4x4xf16> -> tensor<1x1x5x5xf16>
    // CHECK:       return [[PADING]]
}

// CHECK-LABEL: @ConvertUpsamplingToStridedConcatPadC
func.func @ConvertUpsamplingToStridedConcatPadC(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x5x3x3xf32> {
    %0 = IE.Upsampling(%arg0) {
        pad = #IE.UpsamplingPad<pads_channel = [1, 1], pads_height = [0, 0], pads_width = [0, 0]>,
        upsampling_factor = [2, 2, 2]
    } : tensor<1x2x2x2xf32> -> tensor<1x5x3x3xf32>
    return %0 : tensor<1x5x3x3xf32>

    // CHECK-NOT:   IE.Upsampling
    // CHECK-DAG:       [[CST_C:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x2x2x2xf32>
    // CHECK:       [[CONCAT_C:%.*]] = IE.Concat(%arg0, [[CST_C]])
    // CHECK-SAME:      axis = 1
    // CHECK-SAME:      stride = 2
    // CHECK-SAME:      tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32> -> tensor<1x4x2x2xf32>
    // CHECK-DAG:       [[CST_H:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x4x2x2xf32>
    // CHECK:       [[CONCAT_H:%.*]] = IE.Concat([[CONCAT_C]], [[CST_H]])
    // CHECK-SAME:      axis = 2
    // CHECK-SAME:      stride = 2
    // CHECK-SAME:      tensor<1x4x2x2xf32>, tensor<1x4x2x2xf32> -> tensor<1x4x4x2xf32>
    // CHECK-DAG:       [[CST_W:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x4x4x2xf32>
    // CHECK:       [[CONCAT_W:%.*]] = IE.Concat([[CONCAT_H]], [[CST_W]])
    // CHECK-SAME:      axis = 3
    // CHECK-SAME:      stride = 2
    // CHECK-SAME:      tensor<1x4x4x2xf32>, tensor<1x4x4x2xf32> -> tensor<1x4x4x4xf32>
    // CHECK:       [[SLICE:%.*]] =  IE.Slice [[CONCAT_W]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 3, 3]
    // CHECK-SAME:      tensor<1x4x4x4xf32> to tensor<1x4x3x3xf32>
    // CHECK:       [[PADING:%.*]] = IE.Pad([[SLICE]])
    // CHECK-SAME:      mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:      pad_value_attr = 0.000000e+00 : f64
    // CHECK-SAME:      pads_begin_attr = [0, 1, 0, 0]
    // CHECK-SAME:      pads_end_attr = [0, 0, 0, 0]}
    // CHECK-SAME:      tensor<1x4x3x3xf32> -> tensor<1x5x3x3xf32>
    // CHECK:       return [[PADING]]
}

// CHECK-LABEL: @ConvertUpsamplingHCToStridedConcatPadW
func.func @ConvertUpsamplingHCToStridedConcatPadW(%arg0: tensor<1x3x3x3xf32>) -> tensor<1x11x9x8xf32> {
    %0 = IE.Upsampling(%arg0) {
        pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [3, 2]>,
        upsampling_factor = [1, 4, 5]
    } : tensor<1x3x3x3xf32> -> tensor<1x11x9x8xf32>
    return %0 : tensor<1x11x9x8xf32>

    // CHECK-NOT:   IE.Upsampling
    // CHECK-DAG:       [[CST_C:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x3x3x3xf32>
    // CHECK:       [[CONCAT_C:%.*]] = IE.Concat(%arg0, [[CST_C]], [[CST_C]], [[CST_C]], [[CST_C]])
    // CHECK-SAME:      axis = 1
    // CHECK-SAME:      stride = 5
    // CHECK-SAME:      tensor<1x3x3x3xf32>, tensor<1x3x3x3xf32> -> tensor<1x15x3x3xf32>
    // CHECK-DAG:       [[CST_H:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x15x3x3xf32>
    // CHECK:       [[CONCAT_H:%.*]] = IE.Concat([[CONCAT_C]], [[CST_H]], [[CST_H]], [[CST_H]])
    // CHECK-SAME:      axis = 2
    // CHECK-SAME:      stride = 4
    // CHECK-SAME:      tensor<1x15x3x3xf32>, tensor<1x15x3x3xf32> -> tensor<1x15x12x3xf32>
    // CHECK:       [[SLICE:%.*]] =  IE.Slice [[CONCAT_H]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 11, 9, 3]
    // CHECK-SAME:      tensor<1x15x12x3xf32> to tensor<1x11x9x3xf32>
    // CHECK:       [[PADING:%.*]] = IE.Pad([[SLICE]])
    // CHECK-SAME:      mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:      pad_value_attr = 0.000000e+00 : f64
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 3]
    // CHECK-SAME:      pads_end_attr = [0, 0, 0, 2]
    // CHECK-SAME:      tensor<1x11x9x3xf32> -> tensor<1x11x9x8xf32>
    // CHECK:       return [[PADING]]
}

// CHECK-LABEL: @ConvertUpsamplingToStridedConcatPadNoFactor
func.func @ConvertUpsamplingToStridedConcatPadNoFactor(%arg0: tensor<1x1x1x1xf16>) -> tensor<1x2x2x1xf16> {
    %0 = IE.Upsampling(%arg0) {
        pad = #IE.UpsamplingPad<pads_channel = [0, 1], pads_height = [1, 0], pads_width = [0, 0]>,
        upsampling_factor = [2, 1, 1]
    } : tensor<1x1x1x1xf16> -> tensor<1x2x2x1xf16>
    return %0 : tensor<1x2x2x1xf16>

    // CHECK-NOT:   IE.Upsampling
    // CHECK:       [[PADING:%.*]] = IE.Pad(%arg0)
    // CHECK-SAME:      mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:      pad_value_attr = 0.000000e+00 : f64
    // CHECK-SAME:      pads_begin_attr = [0, 0, 1, 0]
    // CHECK-SAME:      pads_end_attr = [0, 1, 0, 0]
    // CHECK-SAME:      tensor<1x1x1x1xf16> -> tensor<1x2x2x1xf16>
    // CHECK:       return [[PADING]]
}
