//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-affine-reshape --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @ShiftReshapeAndTranspose(%arg0: tensor<1x12x7x1xf32>, %arg1: tensor<1x1x12x7xf32>) -> (tensor<7x12x1x1xf32>, tensor<7x12x1x1xf32>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x12x7x1xf32> -> tensor<12x7x1x1xf32>
    %1 = IE.Transpose(%0) {order_value = #CNHW} : tensor<12x7x1x1xf32> -> tensor<7x12x1x1xf32>

    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf32> -> tensor<12x7x1x1xf32>
    %3 = IE.Transpose(%2) {order_value = #CNHW} : tensor<12x7x1x1xf32> -> tensor<7x12x1x1xf32>

    return %1, %3 : tensor<7x12x1x1xf32>, tensor<7x12x1x1xf32>

    // CHECK:               [[TRANSPOSE0:%.*]] = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x12x7x1xf32> -> tensor<1x7x12x1xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[TRANSPOSE0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [7, 12, 1, 1]} : tensor<1x7x12x1xf32> -> tensor<7x12x1x1xf32>

    // CHECK:               [[TRANSPOSE1:%.*]] = IE.Transpose(%arg1) {order_value = #NCWH} : tensor<1x1x12x7xf32> -> tensor<1x1x7x12xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [7, 12, 1, 1]} : tensor<1x1x7x12xf32> -> tensor<7x12x1x1xf32>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<7x12x1x1xf32>, tensor<7x12x1x1xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
#WHC = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

func.func @CollapseReshapeAndTranspose(%arg0: tensor<1x1x12x7xf32>) -> (tensor<7x12xf32>, tensor<7x1x12xf32>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [12, 7] } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>
    %1 = IE.Transpose(%0) {order_value = #CN} : tensor<12x7xf32> -> tensor<7x12xf32>
    
    %2 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 1, 7] } : tensor<1x1x12x7xf32> -> tensor<12x1x7xf32>
    %3 = IE.Transpose(%2) {order_value = #WHC} : tensor<12x1x7xf32> -> tensor<7x1x12xf32>

    return %1, %3 : tensor<7x12xf32>, tensor<7x1x12xf32>

    // CHECK:               [[TRANSPOSE0:%.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x1x12x7xf32> -> tensor<1x1x7x12xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[TRANSPOSE0]]) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [7, 12]} : tensor<1x1x7x12xf32> -> tensor<7x12xf32>

    // CHECK:               [[TRANSPOSE1:%.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x1x12x7xf32> -> tensor<1x1x7x12xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE1]]) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [7, 1, 12]} : tensor<1x1x7x12xf32> -> tensor<7x1x12xf32>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<7x12xf32>, tensor<7x1x12xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @ExtendReshapeAndTranspose(%arg0: tensor<7x12xf32>) -> (tensor<1x1x12x7xf32>, tensor<1x12x7x1xf32>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 7, 12] } : tensor<7x12xf32> -> tensor<1x1x7x12xf32>
    %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x1x7x12xf32> -> tensor<1x1x12x7xf32>

    %2 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 12, 1]} : tensor<7x12xf32> -> tensor<1x7x12x1xf32>
    %3 = IE.Transpose(%2) {order_value = #NHCW} : tensor<1x7x12x1xf32> -> tensor<1x12x7x1xf32>

    return %1, %3 : tensor<1x1x12x7xf32>, tensor<1x12x7x1xf32>

    // CHECK:               [[TRANSPOSE0:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<7x12xf32> -> tensor<12x7xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[TRANSPOSE0]]) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 12, 7]} : tensor<12x7xf32> -> tensor<1x1x12x7xf32>

    // CHECK:               [[TRANSPOSE1:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<7x12xf32> -> tensor<12x7xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE1]]) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 12, 7, 1]} : tensor<12x7xf32> -> tensor<1x12x7x1xf32>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<1x1x12x7xf32>, tensor<1x12x7x1xf32>
}

// -----

func.func @ShiftReshapeAndExpand(%arg0: tensor<1x1x12x7xf32>, %arg1: tensor<1x19x80x1xf16>) -> (tensor<12x16x1x1xf32>, tensor<1x16x19x80xf16>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf32> -> tensor<12x7x1x1xf32>
    %1 = IE.Expand(%0) {pads_begin = [0, 3, 0, 0], pads_end = [0, 6, 0, 0]} : tensor<12x7x1x1xf32> -> tensor<12x16x1x1xf32>

    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 19, 80]} : tensor<1x19x80x1xf16> -> tensor<1x1x19x80xf16>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x19x80xf16> -> tensor<1x16x19x80xf16>

    return %1, %3 : tensor<12x16x1x1xf32>, tensor<1x16x19x80xf16>

    // CHECK:               [[EXPAND0:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 3], pads_end = [0, 0, 0, 6]} : tensor<1x1x12x7xf32> -> tensor<1x1x12x16xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[EXPAND0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 16, 1, 1]} : tensor<1x1x12x16xf32> -> tensor<12x16x1x1xf32>

    // CHECK:               [[EXPAND1:%.*]] = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [15, 0, 0, 0]} : tensor<1x19x80x1xf16> -> tensor<16x19x80x1xf16>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[EXPAND1]]) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16, 19, 80]} : tensor<16x19x80x1xf16> -> tensor<1x16x19x80xf16>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<12x16x1x1xf32>, tensor<1x16x19x80xf16>
}

// -----

func.func @CollapseReshapeAndExpand(%arg0: tensor<1x1x12x7xf32>) -> tensor<16x7xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [12, 7] } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>
    %1 = IE.Expand(%0) {pads_begin = [0, 0], pads_end = [4, 0]}  : tensor<12x7xf32> -> tensor<16x7xf32>
    return %1 : tensor<16x7xf32>

    // CHECK:               [[EXPAND:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x1x12x7xf32> -> tensor<1x1x16x7xf32>
    // CHECK:               [[RESHAPE:%.*]]  = IE.AffineReshape([[EXPAND]]) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [16, 7]} : tensor<1x1x16x7xf32> -> tensor<16x7xf32>
    // CHECK:               return [[RESHAPE]] : tensor<16x7xf32>
}

// -----

func.func @ExtendReshapeAndExpand(%arg0: tensor<7x12xf32>) -> tensor<1x7x16x1xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 12, 1] } : tensor<7x12xf32> -> tensor<1x7x12x1xf32>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 2, 0], pads_end = [0, 0, 2, 0]} : tensor<1x7x12x1xf32> -> tensor<1x7x16x1xf32>
    return %1 : tensor<1x7x16x1xf32>

    // CHECK:               [[EXPAND:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 2], pads_end = [0, 2]} : tensor<7x12xf32> -> tensor<7x16xf32>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[EXPAND]]) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 16, 1]} : tensor<7x16xf32> -> tensor<1x7x16x1xf32>
    // CHECK:               return [[RESHAPE]] : tensor<1x7x16x1xf32>
}

// -----

#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

func.func @NoChangesShiftReshape(%arg0: tensor<1x1x12x8xf32>) -> tensor<4x12x2x1xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 4, 2, 1]} : tensor<1x1x12x8xf32> -> tensor<12x4x2x1xf32>
    %1 = IE.Transpose(%0) {order_value = #CNHW} : tensor<12x4x2x1xf32> -> tensor<4x12x2x1xf32>

    return %1 : tensor<4x12x2x1xf32>

    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape(%arg0) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 4, 2, 1]} : tensor<1x1x12x8xf32> -> tensor<12x4x2x1xf32>
    // CHECK:               [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE]]) {order_value = #map} : tensor<12x4x2x1xf32> -> tensor<4x12x2x1xf32>
    // CHECK:               return [[TRANSPOSE]] : tensor<4x12x2x1xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @NoChangesCollapseReshape(%arg0: tensor<1x2x12x7xf32>, %arg1: tensor<1x1x12x7xf32>, %arg2: tensor<1x1x12x6xf32>) 
                                    -> (tensor<7x24xf32>, tensor<12x7x16xf32>, tensor<12x3x16xf32>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [24, 7] } : tensor<1x2x12x7xf32> -> tensor<24x7xf32>
    %1 = IE.Transpose(%0) {order_value = #CN} : tensor<24x7xf32> -> tensor<7x24xf32>

    %2 = IE.AffineReshape(%arg1) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 7, 1] } : tensor<1x1x12x7xf32> -> tensor<12x7x1xf32>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0], pads_end = [0, 0, 15]}  : tensor<12x7x1xf32> -> tensor<12x7x16xf32>    
    
    %4 = IE.AffineReshape(%arg2) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 3, 2] } : tensor<1x1x12x6xf32> -> tensor<12x3x2xf32>
    %5 = IE.Expand(%4) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]}  : tensor<12x3x2xf32> -> tensor<12x3x16xf32>

    return %1, %3, %5 : tensor<7x24xf32>, tensor<12x7x16xf32>, tensor<12x3x16xf32>

    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [24, 7]} : tensor<1x2x12x7xf32> -> tensor<24x7xf32>
    // CHECK:               [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #map} : tensor<24x7xf32> -> tensor<7x24xf32>

    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape(%arg1) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 7, 1]} : tensor<1x1x12x7xf32> -> tensor<12x7x1xf32>
    // CHECK:               [[EXPAND0:%.*]] = IE.Expand([[RESHAPE1]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 15]} : tensor<12x7x1xf32> -> tensor<12x7x16xf32>   
    
    // CHECK:               [[RESHAPE2:%.*]] = IE.AffineReshape(%arg2) 
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 3, 2]} : tensor<1x1x12x6xf32> -> tensor<12x3x2xf32>
    // CHECK:               [[EXPAND1:%.*]] = IE.Expand([[RESHAPE2]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]} : tensor<12x3x2xf32> -> tensor<12x3x16xf32>
    // CHECK:               return [[TRANSPOSE]], [[EXPAND0]], [[EXPAND1]] : tensor<7x24xf32>, tensor<12x7x16xf32>, tensor<12x3x16xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @NoChangesExpandReshape(%arg0: tensor<6x12xf32>) -> tensor<1x2x12x3xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 3, 12] } : tensor<6x12xf32> -> tensor<1x2x3x12xf32>
    %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x2x3x12xf32> -> tensor<1x2x12x3xf32>
    return %1 : tensor<1x2x12x3xf32>

    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 3, 12]} : tensor<6x12xf32> -> tensor<1x2x3x12xf32>
    // CHECK:               [[TRANSPOSE:%.*]] = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x2x3x12xf32> -> tensor<1x2x12x3xf32>
    // CHECK:               return [[TRANSPOSE]] : tensor<1x2x12x3xf32>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeWithConcatWithOffsets
func.func @SwapAffineReshapeWithConcatWithOffsets(%arg0: tensor<1x1024x1x1xf16>, %arg1: tensor<1x1024x1x1xf16>) ->
                        tensor<2x1x1024xf16> {
     %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [1, 1, 1024]} : tensor<1x1024x1x1xf16> -> tensor<1x1x1024xf16>
    
      %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [1, 1, 1024]} : tensor<1x1024x1x1xf16> -> tensor<1x1x1024xf16>    

     %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0], [1, 0, 0]]} : tensor<1x1x1024xf16>, tensor<1x1x1024xf16> -> tensor<2x1x1024xf16>

     return %2 : tensor<2x1x1024xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16> -> tensor<2x1024x1x1xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [2, 1, 1024]}
     // CHECK-SAME:  tensor<2x1024x1x1xf16> -> tensor<2x1x1024xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeWithConcatWithOffsets
func.func @NotSwapAffineReshapeWithConcatWithOffsets(%arg0: tensor<1x40x80x2xf16>, %arg1: tensor<1x20x40x4xf16>) ->
                        tensor<1x60x1x160xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 40, 1, 160]} : tensor<1x40x80x2xf16> -> tensor<1x40x1x160xf16>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 20, 1, 160]} : tensor<1x20x40x4xf16> -> tensor<1x20x1x160xf16>

    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 40, 0, 0]]} : tensor<1x40x1x160xf16>, tensor<1x20x1x160xf16> -> tensor<1x60x1x160xf16>

    return %2 : tensor<1x60x1x160xf16>

    // CHECK:       IE.AffineReshape
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 40, 1, 160]}
    // CHECK-SAME:  tensor<1x40x80x2xf16> -> tensor<1x40x1x160xf16>
    // CHECK:       IE.AffineReshape
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 20, 1, 160]}
    // CHECK-SAME:  tensor<1x20x40x4xf16> -> tensor<1x20x1x160xf16>
    // CHECK:       IE.Concat
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 40, 0, 0]]} : tensor<1x40x1x160xf16>, tensor<1x20x1x160xf16> -> tensor<1x60x1x160xf16>
}

// -----

// CHECK-LABEL: @NoSwapAffineReshapeConcatDiffMapping
func.func @NoSwapAffineReshapeConcatDiffMapping(%arg0: tensor<1x70x1x28xf16>, %arg1: tensor<1x70x28x1xf16>) ->
                        tensor<1x70x56xf16> {

      %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 70, 28]} : tensor<1x70x1x28xf16> -> tensor<1x70x28xf16>
      %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 70, 28]} : tensor<1x70x28x1xf16> -> tensor<1x70x28xf16>
      %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0], [0, 0, 28]]} : tensor<1x70x28xf16>, tensor<1x70x28xf16> -> tensor<1x70x56xf16>

      return %2 : tensor<1x70x56xf16>

     // CHECK:       IE.AffineReshape(%arg0)
     // CHECK:       IE.AffineReshape(%arg1)
     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0], [0, 0, 28]]} 
     // CHECK-SAME:           tensor<1x70x28xf16>, tensor<1x70x28xf16> -> tensor<1x70x56xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatSameDims
func.func @SwapAffineReshapeConcatSameDims(%arg0: tensor<1x1x12x7xf16>, %arg1: tensor<1x1x12x7xf16>) ->
                        tensor<24x7x1x1xf16> {

      %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf16> -> tensor<12x7x1x1xf16>
      %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf16> -> tensor<12x7x1x1xf16>
      %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [12, 0, 0, 0]]} : tensor<12x7x1x1xf16>, tensor<12x7x1x1xf16> -> tensor<24x7x1x1xf16>

      return %2 : tensor<24x7x1x1xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 12, 0]]} 
     // CHECK-SAME:           tensor<1x1x12x7xf16>, tensor<1x1x12x7xf16> -> tensor<1x1x24x7xf16> 
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [24, 7, 1, 1]}
     // CHECK-SAME:  tensor<1x1x24x7xf16> -> tensor<24x7x1x1xf16>
}

// -----

// CHECK-LABEL: @NoPropagateAfiineReshapeDifferentOutput
func.func @NoPropagateAfiineReshapeDifferentOutput(%arg0: tensor<1x32x32x16xf16>, %arg1: tensor<1x16x16x20xf16>) -> tensor<1x21504xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 16384]} : tensor<1x32x32x16xf16> -> tensor<1x16384xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 5120]} : tensor<1x16x16x20xf16> -> tensor<1x5120xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0], [0, 16384]]} : tensor<1x16384xf16>, tensor<1x5120xf16> -> tensor<1x21504xf16>

    return %2 : tensor<1x21504xf16>
    // CHECK: IE.AffineReshape
    // CHECK: IE.AffineReshape
    // CHECK: IE.Concat
    // CHECK-SAME: tensor<1x16384xf16>, tensor<1x5120xf16> -> tensor<1x21504xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatOnNewOneDim
func.func @SwapAffineReshapeConcatOnNewOneDim(%arg0: tensor<1x68x128x128xf16>, %arg1: tensor<1x68x128x128xf16>) ->
                        tensor<1x4x68x128x128xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %3 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %4 = IE.Concat(%0, %1, %2, %3) {static_offsets = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 3, 0, 0, 0]]} : tensor<1x1x68x128x128xf16>, tensor<1x1x68x128x128xf16>, tensor<1x1x68x128x128xf16>, tensor<1x1x68x128x128xf16> -> tensor<1x4x68x128x128xf16>
    
    return %4 :tensor<1x4x68x128x128xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]} : tensor<1x68x128x128xf16>, tensor<1x68x128x128xf16>, tensor<1x68x128x128xf16>, tensor<1x68x128x128xf16> -> tensor<4x68x128x128xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 4, 68, 128, 128]}
     // CHECK-SAME:  tensor<4x68x128x128xf16> -> tensor<1x4x68x128x128xf16>
}

// -----

// CHECK-LABEL: @SwapWithSoftmax
func.func @SwapWithSoftmax(%arg0: tensor<1x24x16x1xf32>) -> tensor<1x1x24x16xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 24, 16]} : tensor<1x24x16x1xf32> -> tensor<1x1x24x16xf32>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x1x24x16xf32> -> tensor<1x1x24x16xf32>
    return %1: tensor<1x1x24x16xf32>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 24, 16]} : tensor<1x24x16x1xf32> -> tensor<1x1x24x16xf32>
    // CHECK:        return [[RESHAPE]] : tensor<1x1x24x16xf32>
}

// -----

// CHECK-LABEL: @SwapWithSoftmaxWithPadSize
func.func @SwapWithSoftmaxWithPadSize(%arg0: tensor<1x1504x1x1500xf16>) -> tensor<1x1504x1500x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1504, 1500, 1]} : tensor<1x1504x1x1500xf16> -> tensor<1x1504x1500x1xf16>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x1504x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    return %1: tensor<1x1504x1500x1xf16>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x1504x1x1500xf16> -> tensor<1x1504x1x1500xf16>
    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1504, 1500, 1]} : tensor<1x1504x1x1500xf16> -> tensor<1x1504x1500x1xf16>
    // CHECK: return [[RESHAPE]] : tensor<1x1504x1500x1xf16>
}

// -----

// CHECK-LABEL: @NotSwapWithSoftmaxWhenPadSizeButAxisChanged
func.func @NotSwapWithSoftmaxWhenPadSizeButAxisChanged(%arg0: tensor<1504x1x1500x1xf16>) -> tensor<1x1504x1500x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1504, 1500, 1]} : tensor<1504x1x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x1504x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    return %1: tensor<1x1504x1500x1xf16>

     // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
     // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1504, 1500, 1]} : tensor<1504x1x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[RESHAPE]]) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x1504x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    // CHECK: return [[SOFTMAX]] : tensor<1x1504x1500x1xf16>
}

// -----

// CHECK-LABEL: @NoSwapWithSoftmaxWhenAxisMerged
func.func @NoSwapWithSoftmaxWhenAxisMerged(%arg0: tensor<1x24x16x1xf32>) -> tensor<1x2x12x16xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 12, 16]} : tensor<1x24x16x1xf32> -> tensor<1x2x12x16xf32>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x2x12x16xf32> -> tensor<1x2x12x16xf32>
    return %1: tensor<1x2x12x16xf32>

    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 12, 16]} : tensor<1x24x16x1xf32> -> tensor<1x2x12x16xf32>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[RESHAPE]]) {axisInd = 2 : i64} : tensor<1x2x12x16xf32> -> tensor<1x2x12x16xf32>
    // CHECK:        return [[SOFTMAX]] : tensor<1x2x12x16xf32>
}

// -----

// CHECK-LABEL: @SwapWithSoftmaxWhenAxisMergedWithOnes
func.func @SwapWithSoftmaxWhenAxisMergedWithOnes(%arg0: tensor<1x24x16x1xf32>) -> tensor<1x24x1x16xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 24, 1, 16]} : tensor<1x24x16x1xf32> -> tensor<1x24x1x16xf32>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x24x1x16xf32> -> tensor<1x24x1x16xf32>
    return %1: tensor<1x24x1x16xf32>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 24, 1, 16]} : tensor<1x24x16x1xf32> -> tensor<1x24x1x16xf32>
    // CHECK:        return [[RESHAPE]] : tensor<1x24x1x16xf32>
}

// -----

// CHECK-LABEL: @NoSwapWithSoftmaxWhenAxisSplit
func.func @NoSwapWithSoftmaxWhenAxisSplit(%arg0: tensor<4x2x8x1xf32>) -> tensor<1x4x16x1xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 4, 16, 1]} : tensor<4x2x8x1xf32> -> tensor<1x4x16x1xf32>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x4x16x1xf32> -> tensor<1x4x16x1xf32>
    return %1: tensor<1x4x16x1xf32>

    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 4, 16, 1]} : tensor<4x2x8x1xf32> -> tensor<1x4x16x1xf32>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[RESHAPE]]) {axisInd = 2 : i64} : tensor<1x4x16x1xf32> -> tensor<1x4x16x1xf32>
    // CHECK:        return [[SOFTMAX]] : tensor<1x4x16x1xf32>
}

// -----

// CHECK-LABEL: @SwapWithSoftmaxWhenAxisSplitWithOnes
func.func @SwapWithSoftmaxWhenAxisSplitWithOnes(%arg0: tensor<4x8x1x2xf32>) -> tensor<1x4x8x2xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 4, 8, 2]} : tensor<4x8x1x2xf32> -> tensor<1x4x8x2xf32>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x4x8x2xf32> -> tensor<1x4x8x2xf32>
    return %1: tensor<1x4x8x2xf32>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<4x8x1x2xf32> -> tensor<4x8x1x2xf32>
    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 4, 8, 2]} : tensor<4x8x1x2xf32> -> tensor<1x4x8x2xf32>
    // CHECK:        return [[RESHAPE]] : tensor<1x4x8x2xf32>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeGelu
func.func @SwapAffineReshapeGelu(%arg0: tensor<1x2048x375x4xf16>) -> tensor<1x2048x1500x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]} : tensor<1x2048x375x4xf16> -> tensor<1x2048x1500x1xf16>
    %1 = IE.Gelu(%0) : tensor<1x2048x1500x1xf16> -> tensor<1x2048x1500x1xf16>
    return %1 : tensor<1x2048x1500x1xf16>

    // CHECK: [[GELU:%.*]] = IE.Gelu(%arg0) : tensor<1x2048x375x4xf16> -> tensor<1x2048x375x4xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[GELU]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]}
    // CHECK-SAME:          tensor<1x2048x375x4xf16> -> tensor<1x2048x1500x1xf16>
    // CHECK: return [[AFFINERESHAPE]] : tensor<1x2048x1500x1xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeGeluDueToDifferentRank
func.func @NotSwapAffineReshapeGeluDueToDifferentRank(%arg0: tensor<2048x375x4xf16>) -> tensor<1x2048x1500x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]} : tensor<2048x375x4xf16> -> tensor<1x2048x1500x1xf16>
    %1 = IE.Gelu(%0) : tensor<1x2048x1500x1xf16> -> tensor<1x2048x1500x1xf16>
    return %1 : tensor<1x2048x1500x1xf16>

    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]}
    // CHECK-SAME:          tensor<2048x375x4xf16> -> tensor<1x2048x1500x1xf16>

    // CHECK: [[GELU:%.*]] = IE.Gelu([[AFFINERESHAPE]]) : tensor<1x2048x1500x1xf16> -> tensor<1x2048x1500x1xf16>
    // CHECK: return [[GELU]] : tensor<1x2048x1500x1xf16>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: SwapConcatAffineReshape
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x3x40x40x81xf16, {order = #NCDHW}>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
// CHECK-SAME:     [[INPUT3:%arg[0-9]]]: tensor<1x4800x85xf16>)
func.func @SwapConcatAffineReshape(%input0: tensor<1x3x40x40x81xf16, {order = #NCDHW}>,
                                   %input1: tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
                                   %input2: tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
                                   %input3: tensor<1x4800x85xf16>)
    -> (tensor<1x9600x85xf16>) {
    %concat0 = IE.Concat(%input0, %input1, %input2) {
        static_offsets = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 81],
            [0, 0, 0, 0, 83]
        ]
    } : tensor<1x3x40x40x81xf16, {order = #NCDHW}>,
        tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
        tensor<1x3x40x40x2xf16, {order = #NCDHW}>
            -> tensor<1x3x40x40x85xf16, {order = #NCDHW}>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [1], [1], [1], [2]],
            shape_value = [1, 4800, 85]
        } : tensor<1x3x40x40x85xf16, {order = #NCDHW}> -> tensor<1x4800x85xf16>
    %concat1 = IE.Concat(%reshape, %input3) {
        static_offsets = [
            [0, 0, 0],
            [0, 4800, 0]
        ]
    } : tensor<1x4800x85xf16>,
        tensor<1x4800x85xf16>
            -> tensor<1x9600x85xf16>
    return %concat1 : tensor<1x9600x85xf16>

    // CHECK:                [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME:            {dim_mapping =
    // CHECK-SAME{LITERAL}:    [[0], [1], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:    shape_value = [1, 4800, 81]} :
    // CHECK-SAME{LITERAL}:    tensor<1x3x40x40x81xf16, {order = #NCDHW}> -> tensor<1x4800x81xf16>
    // CHECK:                [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME:            {dim_mapping =
    // CHECK-SAME{LITERAL}:    [[0], [1], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:    shape_value = [1, 4800, 2]} :
    // CHECK-SAME{LITERAL}:    tensor<1x3x40x40x2xf16, {order = #NCDHW}> -> tensor<1x4800x2xf16>
    // CHECK:                [[RESHAPE2:%.+]] = IE.AffineReshape([[INPUT2]])
    // CHECK-SAME:            {dim_mapping =
    // CHECK-SAME{LITERAL}:    [[0], [1], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:    shape_value = [1, 4800, 2]} :
    // CHECK-SAME{LITERAL}:    tensor<1x3x40x40x2xf16, {order = #NCDHW}> -> tensor<1x4800x2xf16>
    // CHECK:                [[CONCAT0:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]], [[RESHAPE2]]) {
    // CHECK-SAME{LITERAL}:    static_offsets = [[0, 0, 0], [0, 0, 81], [0, 0, 83]]} :
    // CHECK-SAME:             tensor<1x4800x81xf16>, tensor<1x4800x2xf16>, tensor<1x4800x2xf16> -> tensor<1x4800x85xf16>
    // CHECK:                [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[INPUT3]]) {
    // CHECK-SAME{LITERAL}:    static_offsets = [[0, 0, 0], [0, 4800, 0]]} :
    // CHECK-SAME:             tensor<1x4800x85xf16>, tensor<1x4800x85xf16> -> tensor<1x9600x85xf16>
    // CHECK:                return [[CONCAT1]] : tensor<1x9600x85xf16>
}

// -----

// CHECK-LABEL: ConcatAndCollapseReshape
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x1x6x7xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1x6x7xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<12x7xf32>)
func.func @ConcatAndCollapseReshape(%arg0: tensor<1x1x6x7xf32>,
                                    %arg1: tensor<1x1x6x7xf32>,
                                    %arg2: tensor<12x7xf32>) -> tensor<12x14xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 6, 0]
        ]
    } : tensor<1x1x6x7xf32>,
        tensor<1x1x6x7xf32>
            -> tensor<1x1x12x7xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [0], [0], [1]],
            shape_value = [12, 7]
        } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0],
            [0, 7]
        ]
    } : tensor<12x7xf32>,
        tensor<12x7xf32>
            -> tensor<12x14xf32>
    return %concat1 : tensor<12x14xf32>

    // CHECK:             [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [6, 7]} : tensor<1x1x6x7xf32> -> tensor<6x7xf32>
    // CHECK:             [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [6, 7]} : tensor<1x1x6x7xf32> -> tensor<6x7xf32>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]])
    // CHECK-SMAE{LITERAL}: {static_offsets = [[0, 0], [6, 0]]} : tensor<6x7xf32>, tensor<6x7xf32> -> tensor<12x7xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0], [0, 7]]} : tensor<12x7xf32>, tensor<12x7xf32> -> tensor<12x14xf32>
    // CHECK:             return [[CONCAT1]] : tensor<12x14xf32>
}

// -----

// CHECK-LABEL: ConcatAndExtendReshape
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<6x7xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<6x7xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x12x7x1xf32>)
func.func @ConcatAndExtendReshape(%arg0: tensor<6x7xf32>,
                                  %arg1: tensor<6x7xf32>,
                                  %arg2: tensor<1x12x7x1xf32>) -> tensor<1x12x14x1xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0],
            [6, 0]
        ]
    } : tensor<6x7xf32>,
        tensor<6x7xf32>
            -> tensor<12x7xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0, 1], [2, 3]],
            shape_value = [1, 12, 7, 1]
        } : tensor<12x7xf32> -> tensor<1x12x7x1xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 7, 0]
        ]
    } : tensor<1x12x7x1xf32>,
        tensor<1x12x7x1xf32>
            -> tensor<1x12x14x1xf32>
    return %concat1 : tensor<1x12x14x1xf32>

    // CHECK:             [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 6, 7, 1]} : tensor<6x7xf32> -> tensor<1x6x7x1xf32>
    // CHECK:             [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 6, 7, 1]} : tensor<6x7xf32> -> tensor<1x6x7x1xf32>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 6, 0, 0]]} : tensor<1x6x7x1xf32>, tensor<1x6x7x1xf32> -> tensor<1x12x7x1xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]} : tensor<1x12x7x1xf32>, tensor<1x12x7x1xf32> -> tensor<1x12x14x1xf32>
    // CHECK:             return [[CONCAT1]] : tensor<1x12x14x1xf32>
}

// -----

// CHECK-LABEL: ConcatAndShiftReshape
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x1x6x7xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1x6x7xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<12x7x1x1xf32>)
func.func @ConcatAndShiftReshape(%arg0: tensor<1x1x6x7xf32>,
                                 %arg1: tensor<1x1x6x7xf32>,
                                 %arg2: tensor<12x7x1x1xf32>) -> tensor<12x14x1x1xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 6, 0]
        ]
    } : tensor<1x1x6x7xf32>,
        tensor<1x1x6x7xf32>
            -> tensor<1x1x12x7xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [0], [0], [1, 2, 3]],
            shape_value = [12, 7, 1, 1]
        } : tensor<1x1x12x7xf32> -> tensor<12x7x1x1xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 7, 0, 0]
        ]
    } : tensor<12x7x1x1xf32>,
        tensor<12x7x1x1xf32>
            -> tensor<12x14x1x1xf32>
    return %concat1 : tensor<12x14x1x1xf32>

    // CHECK:             [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [6, 7, 1, 1]} : tensor<1x1x6x7xf32> -> tensor<6x7x1x1xf32>
    // CHECK:             [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [6, 7, 1, 1]} : tensor<1x1x6x7xf32> -> tensor<6x7x1x1xf32>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [6, 0, 0, 0]]} : tensor<6x7x1x1xf32>, tensor<6x7x1x1xf32> -> tensor<12x7x1x1xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 7, 0, 0]]} : tensor<12x7x1x1xf32>, tensor<12x7x1x1xf32> -> tensor<12x14x1x1xf32>
    // CHECK:             return [[CONCAT1]] : tensor<12x14x1x1xf32>
}

// -----

// CHECK-LABEL: NoSwapConcatAxisSplit
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x16x42xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x16x42xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x16x12x7xf32>)
func.func @NoSwapConcatAxisSplit(%arg0: tensor<1x16x42xf32>,
                                 %arg1: tensor<1x16x42xf32>,
                                 %arg2: tensor<1x16x12x7xf32>) -> tensor<1x16x12x14xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0],
            [0, 0, 42]
        ]
    } : tensor<1x16x42xf32>,
        tensor<1x16x42xf32>
            -> tensor<1x16x84xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [1], [2, 3]],
            shape_value = [1, 16, 12, 7]
        } : tensor<1x16x84xf32> -> tensor<1x16x12x7xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 0, 7]
        ]
    } : tensor<1x16x12x7xf32>,
        tensor<1x16x12x7xf32>
            -> tensor<1x16x12x14xf32>
    return %concat1 : tensor<1x16x12x14xf32>

    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0], [0, 0, 42]]} : tensor<1x16x42xf32>, tensor<1x16x42xf32> -> tensor<1x16x84xf32>
    // CHECK:             [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 16, 12, 7]} : tensor<1x16x84xf32> -> tensor<1x16x12x7xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[RESHAPE]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 7]]} : tensor<1x16x12x7xf32>, tensor<1x16x12x7xf32> -> tensor<1x16x12x14xf32>
    // CHECK:             return [[CONCAT1]] : tensor<1x16x12x14xf32>
}

// -----

// CHECK-LABEL: NoSwapConcatAxisMerged
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x16x6x7xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x16x6x7xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x16x84xf32>)
func.func @NoSwapConcatAxisMerged(%arg0: tensor<1x16x6x7xf32>,
                                  %arg1: tensor<1x16x6x7xf32>,
                                  %arg2: tensor<1x16x84xf32>) -> tensor<1x16x168xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 6, 0]
        ]
    } : tensor<1x16x6x7xf32>,
        tensor<1x16x6x7xf32>
            -> tensor<1x16x12x7xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [1], [2], [2]],
            shape_value = [1, 16, 84]
        } : tensor<1x16x12x7xf32> -> tensor<1x16x84xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0, 0],
            [0, 0, 84]
        ]
    } : tensor<1x16x84xf32>,
        tensor<1x16x84xf32>
            -> tensor<1x16x168xf32>
    return %concat1 : tensor<1x16x168xf32>

    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]} : tensor<1x16x6x7xf32>, tensor<1x16x6x7xf32> -> tensor<1x16x12x7xf32>
    // CHECK:             [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 16, 84]} : tensor<1x16x12x7xf32> -> tensor<1x16x84xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[RESHAPE]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0], [0, 0, 84]]} : tensor<1x16x84xf32>, tensor<1x16x84xf32> -> tensor<1x16x168xf32>
    // CHECK:             return [[CONCAT1:%.+]] : tensor<1x16x168xf32>
}

// -----

// CHECK-LABEL: @SwapSqueezeAffineReshapeConcat
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x1x1x256xf16>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1x1x256xf16>
func.func @SwapSqueezeAffineReshapeConcat(%arg0: tensor<1x1x1x256xf16>, %arg1: tensor<1x1x1x256xf16>) ->
                        tensor<2x256xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x1x256xf16> -> tensor<1x256xf16>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x1x256xf16> -> tensor<1x256xf16>

    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0], [1, 0]]} : tensor<1x256xf16>, tensor<1x256xf16> -> tensor<2x256xf16>

    return %2 : tensor<2x256xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<2x1x1x256xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [2, 256]} : tensor<2x1x1x256xf16> -> tensor<2x256xf16>
    // CHECK:       return [[RESHAPE]] : tensor<2x256xf16>
}

// CHECK-LABEL: @SwapBackConcatSqueezeAffineReshapeConcat
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x1x1x256xf16>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1x1x256xf16>
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<2x256xf16>
func.func @SwapBackConcatSqueezeAffineReshapeConcat(%arg0: tensor<1x1x1x256xf16>, %arg1: tensor<1x1x1x256xf16>, %arg2: tensor<2x256xf16>) ->
                        tensor<2x512xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<2x1x1x256xf16>

    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [2, 256]} : tensor<2x1x1x256xf16> -> tensor<2x256xf16>

    %2 = IE.Concat(%1, %arg2) {static_offsets = [[0, 0], [0, 256]]} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x512xf16>

    return %2 : tensor<2x512xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x1x256xf16> -> tensor<1x256xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x1x256xf16> -> tensor<1x256xf16>
    // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0], [1, 0]]} : tensor<1x256xf16>, tensor<1x256xf16> -> tensor<2x256xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CONCAT_0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0], [0, 256]]} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x512xf16>
    // CHECK:       return [[CONCAT_1]] : tensor<2x512xf16>
}

// -----

// CHECK-LABEL: @SwapUnsqueezeAffineReshapeConcat
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x256xf16>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x256xf16>
func.func @SwapUnsqueezeAffineReshapeConcat(%arg0: tensor<1x256xf16>, %arg1: tensor<1x256xf16>) ->
                        tensor<1x2x1x256xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x256xf16>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x256xf16>

    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x2x1x256xf16>

    return %2 : tensor<1x2x1x256xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0], [1, 0]]} : tensor<1x256xf16>, tensor<1x256xf16> -> tensor<2x256xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 1, 256]} : tensor<2x256xf16> -> tensor<1x2x1x256xf16>
    // CHECK:       return [[RESHAPE]] : tensor<1x2x1x256xf16>
}

// CHECK-LABEL: @SwapBackConcatUnsqueezeAffineReshapeConcat
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x256xf16>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x256xf16>
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x2x1x256xf16>
func.func @SwapBackConcatUnsqueezeAffineReshapeConcat(%arg0: tensor<1x256xf16>, %arg1: tensor<1x256xf16>, %arg2: tensor<1x2x1x256xf16>) ->
                        tensor<1x2x1x512xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0], [1, 0]]} : tensor<1x256xf16>, tensor<1x256xf16> -> tensor<2x256xf16>

    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 1, 256]} : tensor<2x256xf16> -> tensor<1x2x1x256xf16>

    %2 = IE.Concat(%1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 256]]} : tensor<1x2x1x256xf16>, tensor<1x2x1x256xf16> -> tensor<1x2x1x512xf16>

    return %2 : tensor<1x2x1x512xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x256xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x256xf16>
    // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x2x1x256xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CONCAT_0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 256]]} : tensor<1x2x1x256xf16>, tensor<1x2x1x256xf16> -> tensor<1x2x1x512xf16>
    // CHECK:       return [[CONCAT_1]] : tensor<1x2x1x512xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeSlice
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x3x144x1439xf16>
func.func @SwapAffineReshapeSlice(%arg0: tensor<1x3x144x1439xf16>) -> (tensor<1x3x9x16x478xf16>, tensor<1x3x9x16x956xf16>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 1439]} : tensor<1x3x144x1439xf16> -> tensor<1x3x9x16x1439xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0, 4] [1, 3, 9, 16, 478] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x16x478xf16>
    %2 = IE.Slice %0 [0, 0, 0, 0, 482] [1, 3, 9, 16, 956] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x16x956xf16>

    return %1, %2 : tensor<1x3x9x16x478xf16>, tensor<1x3x9x16x956xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT0]]
    // CHECK-SAME:      [0, 0, 0, 4] [1, 3, 144, 478] : tensor<1x3x144x1439xf16> to tensor<1x3x144x478xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 478]} : tensor<1x3x144x478xf16> -> tensor<1x3x9x16x478xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[INPUT0]]
    // CHECK-SAME:      [0, 0, 0, 482] [1, 3, 144, 956] : tensor<1x3x144x1439xf16> to tensor<1x3x144x956xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 956]} : tensor<1x3x144x956xf16> -> tensor<1x3x9x16x956xf16>
    // CHECK:       return [[RESHAPE_0]], [[RESHAPE_1]] : tensor<1x3x9x16x478xf16>, tensor<1x3x9x16x956xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeSliceOnMultiAxes
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x3x144x1439xf16>
func.func @NotSwapAffineReshapeSliceOnMultiAxes(%arg0: tensor<1x3x144x1439xf16>) -> (tensor<1x3x9x16x478xf16>, tensor<1x3x9x8x1439xf16>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 1439]} : tensor<1x3x144x1439xf16> -> tensor<1x3x9x16x1439xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0, 4] [1, 3, 9, 16, 478] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x16x478xf16>
    %2 = IE.Slice %0 [0, 0, 0, 0, 0] [1, 3, 9, 8, 1439] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x8x1439xf16>

    return %1, %2 : tensor<1x3x9x16x478xf16>, tensor<1x3x9x8x1439xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 1439]} : tensor<1x3x144x1439xf16> -> tensor<1x3x9x16x1439xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[RESHAPE]]
    // CHECK-SAME:      [0, 0, 0, 0, 4] [1, 3, 9, 16, 478] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x16x478xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[RESHAPE]]
    // CHECK-SAME:      [0, 0, 0, 0, 0] [1, 3, 9, 8, 1439] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x8x1439xf16>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x3x9x16x478xf16>, tensor<1x3x9x8x1439xf16>
}
