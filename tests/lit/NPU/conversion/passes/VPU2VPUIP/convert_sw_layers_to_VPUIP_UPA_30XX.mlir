//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-sw-layers-to-VPUIP-UPA --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @SingleLayer
func.func @SingleLayer(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %0: tensor<1x1x1x1000xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x1000xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x1000xf16>)
    // CHECK: return
    // CHECK-SAME:  tensor<1x1x1x1000xf16>
}

// -----

// CHECK-LABEL: @CTCGreedyDecoder
func.func @CTCGreedyDecoder(%arg0: tensor<20x1x128xf16>, %arg1: tensor<20x1xf16>) -> tensor<1x20x1x1xf16> {
    %0 = VPU.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<20x1x128xf16>, tensor<20x1xf16> -> tensor<1x20x1x1xf16>
    return %0 : tensor<1x20x1x1xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x20x1x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.CTCGreedyDecoderUPA {mergeRepeated}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<20x1x128xf16>, {{[^:]+}} : memref<20x1xf16>) outputs([[VAR0]] : memref<1x20x1x1xf16>) -> memref<1x20x1x1xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x20x1x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL1
func.func @ReduceL1(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 = VPU.ReduceL1(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ReduceUPA {axes_value = [3], keep_dims, task_type = #VPUIP.reduce_layer_type<L1>}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>) outputs([[VAR0]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL2
func.func @ReduceL2(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 =  VPU.ReduceL2(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ReduceUPA {axes_value = [3], keep_dims, task_type = #VPUIP.reduce_layer_type<L2>}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>) outputs([[VAR0]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceProd
func.func @ReduceProd(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 = VPU.ReduceProd(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ReduceUPA {axes_value = [3], keep_dims, task_type = #VPUIP.reduce_layer_type<PROD>}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>) outputs([[VAR0]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @Bucketize
func.func @Bucketize(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xsi32> {
    %0 = const.Declare tensor<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    %1 = VPU.Bucketize(%arg0, %0) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    return %1 : tensor<1x32x112x112xsi32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x112xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.BucketizeUPA {output_type = si32, with_right_bound}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>, {{[^:]+}} : memref<2xsi32>) ouputs([[VAR0]] : memref<1x32x112x112xsi32>) -> memref<1x32x112x112xsi32>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x32x112x112xsi32>
}

// -----

// CHECK-LABEL: @ExtractImagePatches
func.func @ExtractImagePatches(%arg0: tensor<64x3x10x10xf32>) -> tensor<64x27x2x2xf32> {
    %0 = VPU.ExtractImagePatches(%arg0) {sizes = [3, 3], strides = [5, 5], rates = [1, 1], autoPad = #IE.pad_type<VALID>} : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    return %0 : tensor<64x27x2x2xf32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<64x27x2x2xf32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.ExtractImagePatchesUPA {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [3, 3], strides = [5, 5]}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<64x3x10x10xf32>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<64x27x2x2xf32>) -> memref<64x27x2x2xf32>
    // CHECK:       return
    // CHECK-SAME:  tensor<64x27x2x2xf32>
}

// -----

// CHECK-LABEL: @Selu
func.func @Selu(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16> {
    %0 = VPU.Selu(%arg0) {alpha_value = 1.000000e+00 : f64, lambda_value = 2.000000e+00 : f64} : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    return %0 : tensor<1x32x112x112xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x112xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.SeluUPA {alphaValue = 1.000000e+00 : f64, lambdaValue = 2.000000e+00 : f64}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>) outputs([[VAR0]] : memref<1x32x112x112xf16>) -> memref<1x32x112x112xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x32x112x112xf16>
}

// -----

// CHECK-LABEL: @Roll
func.func @Roll(%arg0: tensor<3x10x100x200xf16>) -> tensor<3x10x100x200xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %cst_0 = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    %0 = VPU.Roll(%arg0, %cst, %cst_0) : tensor<3x10x100x200xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<3x10x100x200xf16>
    return %0 : tensor<3x10x100x200xf16>

    // CHECK-DAG:       [[VAR0:%.*]] = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
    // CHECK-DAG:       [[VAR1:%.*]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]

    // CHECK:       [[VAR4:%.*]] = memref.alloc() : memref<3x10x100x200xf16>
    // CHECK:       [[VAR5:%.*]] = VPUIP.RollUPA inputs({{[^:]+}} : memref<3x10x100x200xf16>, {{[^:]+}} : memref<1xsi32>, {{[^:]+}} : memref<2xsi32>) outputs([[VAR4]] : memref<3x10x100x200xf16>) -> memref<3x10x100x200xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<3x10x100x200xf16>
}

// -----

// CHECK-LABEL: @AdaptiveAvgPool
func.func @AdaptiveAvgPool(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x56x56xf16> {
    %0 = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %1 = VPU.AdaptiveAvgPool(%arg0, %0) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    return %1 : tensor<1x32x56x56xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x56x56xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.AdaptiveAvgPoolUPA
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>, {{[^:]+}} : memref<2xsi32>) outputs([[VAR0]] : memref<1x32x56x56xf16>) -> memref<1x32x56x56xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x32x56x56xf16>
}

// -----

// CHECK-LABEL: @AdaptiveMaxPool
func.func @AdaptiveMaxPool(%arg0: tensor<1x32x112x112xf16>) -> (tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>) {
    %0 = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %1, %2 = VPU.AdaptiveMaxPool(%arg0, %0) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    return %1, %2 : tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x56x56xf16>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x32x56x56xsi32>
    // CHECK:       [[VAR2:%.*]], [[VAR3:%.*]] = VPUIP.AdaptiveMaxPoolUPA {index_element_type = si32}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>, {{[^:]+}} : memref<2xsi32>) outputs([[VAR0]] : memref<1x32x56x56xf16>, [[VAR1]] : memref<1x32x56x56xsi32>) -> memref<1x32x56x56xf16>, memref<1x32x56x56xsi32>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
}

// -----

// CHECK-LABEL: @GatherND
func.func @GatherND(%arg0: tensor<2x3x4xsi32>) -> tensor<2x1x3x4xsi32> {
    %cst = const.Declare tensor<2x1x1xsi32> = dense<0> : tensor<2x1x1xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 0 : i64} : tensor<2x3x4xsi32>, tensor<2x1x1xsi32> -> tensor<2x1x3x4xsi32>
    return %0 : tensor<2x1x3x4xsi32>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<2x1x1xsi32> = dense<0> : tensor<2x1x1xsi32>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<2x1x3x4xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.GatherNDUPA {batch_dims = 0 : i64}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<2x3x4xsi32>, {{[^:]+}} : memref<2x1x1xsi32>) outputs([[VAR0]] : memref<2x1x3x4xsi32>) -> memref<2x1x3x4xsi32>
    // CHECK:       return
    // CHECK-SAME:  tensor<2x1x3x4xsi32>
}

// -----

// CHECK-LABEL: @NonMaxSuppression
func.func @NonMaxSuppression(%arg0: tensor<3x100x4xf16>, %arg1: tensor<3x5x100xf16>, %arg2: tensor<300x3xsi32>, %arg3: tensor<300x3xf16>, %arg4: tensor<1xsi32>) -> (tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>) {
    %1, %2, %3 = VPU.NonMaxSuppression(%arg0, %arg1) {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} : tensor<3x100x4xf16>, tensor<3x5x100xf16> -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    return %1, %2, %3 : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<300x3xsi32>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<300x3xf16>
    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<1xsi32>
    // CHECK:       [[VAR3:%.*]], [[VAR4:%.*]], [[VAR5:%.*]]  = VPUIP.NonMaxSuppressionUPA {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64}
    // CHECK-SAME:      inputs({{[^:]+}} : memref<3x100x4xf16>, {{[^:]+}} : memref<3x5x100xf16> outputs([[VAR0]] : memref<300x3xsi32>, [[VAR1]] : memref<300x3xf16>, [[VAR2]] : memref<1xsi32>) -> memref<300x3xsi32>, memref<300x3xf16>, memref<1xsi32>
    // CHECK:       return
    // CHECK-SAME:  tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
}

// -----

// CHECK-LABEL: @EmbeddingBagOffsetsSum
    func.func @EmbeddingBagOffsetsSum(%arg0: tensor<5x6x4xsi32>, %arg1: tensor<4x6x4xsi32>) -> tensor<4x6x4xsi32> {
    %0 = VPU.EmbeddingBagOffsetsSum(%arg0) {default_index_value = 4 : i32, indices_value = [0, 1, 2, 2, 3],
    offsets_value = [0, 0, 2, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
    per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} : tensor<5x6x4xsi32> -> tensor<4x6x4xsi32>
    return %0 : tensor<4x6x4xsi32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<4x6x4xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.EmbeddingBagOffsetsSumUPA {default_index_value = 4 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 0, 2, 2], weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} inputs({{[^:]+}} : memref<5x6x4xsi32>) outputs([[VAR0]] : memref<4x6x4xsi32>) -> memref<4x6x4xsi32>
    // CHECK:       return
    // CHECK-SAME:  tensor<4x6x4xsi32>
}

// -----

// CHECK-LABEL: @EmbeddingSegmentsSum
func.func @EmbeddingSegmentsSum(%arg0: tensor<5x6x4xsi32>, %arg1: tensor<7x6x4xsi32>) -> tensor<7x6x4xsi32> {
    %0 = VPU.EmbeddingSegmentsSum(%arg0) {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32,
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01], segment_ids_value = [0, 1, 2, 3, 4]} : tensor<5x6x4xsi32> -> tensor<7x6x4xsi32>
    return %0 : tensor<7x6x4xsi32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<7x6x4xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.EmbeddingSegmentsSumUPA {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32, per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01], segment_ids_value = [0, 1, 2, 3, 4]} inputs({{[^:]+}} : memref<5x6x4xsi32>) ouputs([[VAR0]] : memref<7x6x4xsi32>) -> memref<7x6x4xsi32>
    // CHECK:       return
    // CHECK-SAME:  tensor<7x6x4xsi32>
}

// -----

// CHECK-LABEL: @NormalizeIE
 func.func @NormalizeIE(%arg0: tensor<1x128x50x85xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x128x50x85xf16> {
    %0 = VPU.NormalizeIE(%arg0, %arg1) {across_spatial = true, channel_shared = true, eps = 1.000000e-05 : f64} : tensor<1x128x50x85xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x50x85xf16>
    return %0 : tensor<1x128x50x85xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x128x50x85xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.NormalizeIEUPA {across_spatial = true, channel_shared = true, eps = 1.000000e-05 : f64} inputs({{[^:]+}} : memref<1x128x50x85xf16>, {{[^:]+}} : memref<1x1x1x1xf16>) outputs([[VAR0]] : memref<1x128x50x85xf16>) -> memref<1x128x50x85xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x128x50x85xf16>
}

// -----

// CHECK-LABEL: @CumSum
func.func @CumSum(%arg0: tensor<1x9xf16>) -> tensor<1x9xf16> {
    %0 = VPU.CumSum(%arg0) {axis_value = 1 : i64 , exclusive, reverse} : tensor<1x9xf16> -> tensor<1x9xf16>
    return %0 : tensor<1x9xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x9xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.CumSumUPA {axis_value = 1 : i64, exclusive, reverse} inputs({{[^:]+}} : memref<1x9xf16>) outputs([[VAR0]] : memref<1x9xf16>) -> memref<1x9xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x9xf16>
}

// -----

// CHECK-LABEL: @DeformablePSROIPooling
func.func @DeformablePSROIPooling(%arg0: tensor<1x441x8x8xf16>, %arg1: tensor<30x5xf16>) -> tensor<30x49x3x3xf16> {
    %0 = VPU.DeformablePSROIPooling(%arg0, %arg1) {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 1 : i64, spatial_bins_y = 1 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.000000e+00 : f64} : tensor<1x441x8x8xf16>, tensor<30x5xf16> -> tensor<30x49x3x3xf16>
    return %0 : tensor<30x49x3x3xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<30x49x3x3xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.DeformablePSROIPoolingUPA {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 1 : i64, spatial_bins_y = 1 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.000000e+00 : f64} inputs({{[^:]+}} : memref<1x441x8x8xf16>, {{[^:]+}} : memref<30x5xf16>) outputs([[VAR0]] : memref<30x49x3x3xf16>) -> memref<30x49x3x3xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<30x49x3x3xf16>
}

// -----

// CHECK-LABEL: @Tan
func.func @Tan(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16> {
    %0 = VPU.Tan(%arg0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    return %0 : tensor<1x32x112x112xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x112xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.TanUPA
    // CHECK-SAME:      inputs({{[^:]+}} : memref<1x32x112x112xf16>) outputs([[VAR0]] : memref<1x32x112x112xf16>) -> memref<1x32x112x112xf16>
    // CHECK:       return
    // CHECK-SAME:  tensor<1x32x112x112xf16>
}
