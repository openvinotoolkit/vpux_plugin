//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --detection-output-decomposition --mlir-elide-elementsattrs-if-larger 32 %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#CWH = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: func.func @FaceDetectionAdas
func.func @FaceDetectionAdas(%arg0: tensor<1x40448xf16>, %arg1: tensor<1x20224xf16>, %arg2: tensor<1x2x40448xf16>) -> tensor<1x1x200x7xf16> {
  %9 = VPU.DetectionOutput(%arg0, %arg1, %arg2) {
    attr = #IE.DetectionOutput<background_label_id = 0 : i64,
            clip_after_nms = false,
            clip_before_nms = false,
            code_type = #IE.detection_output_code_type<CENTER_SIZE>,
            confidence_threshold = 1.000000e-03 : f64,
            decrease_label_id = false,
            input_height = 1 : i64,
            input_width = 1 : i64,
            keep_top_k = [200],
            nms_threshold = 4.500000e-01 : f64,
            normalized = true,
            num_classes = 2 : i64,
            objectness_score = 0.000000e+00 : f64,
            share_location = true,
            top_k = 400 : i64,
            variance_encoded_in_target = false
    >, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>} : tensor<1x40448xf16>, tensor<1x20224xf16>, tensor<1x2x40448xf16> -> tensor<1x1x200x7xf16>
  return %9 : tensor<1x1x200x7xf16>

// CHECK:       [[PRIORS:%.+]] = VPU.Reshape(%arg2)
// CHECK-SAME:      shape_value = [1, 2, 10112, 4]

// CHECK:       [[LOGITS_4D:%.+]] = VPU.Reshape(%arg0)
// CHECK-SAME:      shape_value = [1, 10112, 1, 4]
// CHECK:       [[LOGITS_TRANSPOSED:%.+]] = VPU.MemPermute([[LOGITS_4D]])
// CHECK-SAME:      tensor<1x10112x1x4xf16> -> tensor<1x1x10112x4xf16>

// CHECK:       [[DECODED_BOXES:%.+]] = VPU.DetectionOutputDecodeBoxes([[LOGITS_TRANSPOSED]], [[PRIORS]])
// CHECK-SAME:      code_type = #IE.detection_output_code_type<CENTER_SIZE>

// CHECK:       [[CONFIDENCE:%.+]] = VPU.Reshape(%arg1)
// CHECK-SAME:      shape_value = [1, 1, 10112, 2]
// CHECK:       [[CONF_CWH:%.+]] = VPU.MemPermute([[CONFIDENCE]])
// CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCWH

// CHECK:       [[INDICES_CONST:%.+]] = const.Declare tensor<1x1x2x10112xsi32>
// CHECK:       [[SORTING_BUFFER:%.+]] = const.Declare tensor<1x1x16x256xsi32>
// CHECK:       [[CONF:%.+]], [[INDICES:%.+]], [[SIZES:%.+]] = VPU.DetectionOutputSort([[CONF_CWH]], [[INDICES_CONST]], [[SORTING_BUFFER]])
// CHECK-SAME:      confidence_threshold = 1.000000e-03 : f64,
// CHECK-SAME:      top_k = 400 : i64

// CHECK:       [[TOPK_CONF:%.+]] = VPU.Slice [[CONF]] [0, 0, 0, 0] [1, 1, 2, 400]
// CHECK:       [[TOPK_INDICES:%.+]] = VPU.Slice [[INDICES]] [0, 0, 0, 0] [1, 1, 2, 400]
// CHECK:       [[SIZE_RESHAPED:%.+]] = VPU.Reshape([[SIZES]])
// CHECK-SAME:      tensor<1x1x2x1xsi32> -> tensor<1x1x1x2xsi32>

// CHECK:       [[OUT_CONF:%.+]], [[OUT_BOXES:%.+]], [[OUT_SIZES:%.+]] = VPU.DetectionOutputNmsCaffe([[TOPK_CONF]], [[DECODED_BOXES]], [[TOPK_INDICES]], [[SIZE_RESHAPED]])
// CHECK-SAME:      background_id = 0 : i64,
// CHECK-SAME:      nms_threshold = 4.500000e-01 : f64

// CHECK:       [[RESULT:%.+]] = VPU.DetectionOutputCollectResults([[OUT_CONF]], [[OUT_BOXES]], [[OUT_SIZES]])
// CHECK-SAME:      keep_top_k = 200 : i64

// CHECK:       return [[RESULT]] : tensor<1x1x200x7xf16>
}

// -----

// CHECK-LABEL: func.func @NotNormalizedEncodedVariance
func.func @NotNormalizedEncodedVariance(%arg0: tensor<1x40448xf16>, %arg1: tensor<1x20224xf16>, %arg2: tensor<1x1x50560xf16>) -> tensor<1x1x200x7xf16> {
  %9 = VPU.DetectionOutput(%arg0, %arg1, %arg2) {
    attr = #IE.DetectionOutput<background_label_id = 0 : i64,
            clip_after_nms = false,
            clip_before_nms = false,
            code_type = #IE.detection_output_code_type<CENTER_SIZE>,
            confidence_threshold = 1.000000e-03 : f64,
            decrease_label_id = false,
            input_height = 640 : i64,
            input_width = 480 : i64,
            keep_top_k = [200],
            nms_threshold = 4.500000e-01 : f64,
            normalized = false,
            num_classes = 2 : i64,
            objectness_score = 0.000000e+00 : f64,
            share_location = true,
            top_k = 400 : i64,
            variance_encoded_in_target = true
    >, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>} : tensor<1x40448xf16>, tensor<1x20224xf16>, tensor<1x1x50560xf16> -> tensor<1x1x200x7xf16>
  return %9 : tensor<1x1x200x7xf16>

// CHECK:       [[NON_NORMALIZED:%.+]] = VPU.Reshape(%arg2)
// CHECK-SAME:      shape_value = [1, 1, 10112, 5]
// CHECK:       [[NORM_PRIORS:%.+]] = VPU.DetectionOutputNormalize([[NON_NORMALIZED]])
// CHECK-SAME:      tensor<1x1x10112x5xf16> ->
// CHECK-SAME:      tensor<1x1x10112x4xf16>
// CHECK:       VPU.DetectionOutputDecodeBoxes
// CHECK-SAME:     [[NORM_PRIORS]]
}
