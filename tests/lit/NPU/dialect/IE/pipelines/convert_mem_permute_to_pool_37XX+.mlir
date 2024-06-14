//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-mem-permute-to-pool %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX


// CHECK-LABEL: @MemPermuteNHCWInNCHWOutNHCWPerm
func.func @MemPermuteNHCWInNCHWOutNHCWPerm(%arg0: tensor<1x32x48x224xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>}>)
        -> tensor<1x32x48x224xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
    } : tensor<1x32x48x224xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>}>
        -> tensor<1x32x48x224xf16>

    return %MEM_PERMUTE : tensor<1x32x48x224xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x32x48x224xf16, {order = #NHCW}>
    // CHECK-SAME:      -> tensor<1x32x48x224xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 224, 48, 32]
    // CHECK-SAME:  } inputs([[IN_LAYOUT_CAST]] : tensor<1x32x48x224xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x224x48x32xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x224x48x32xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x224x48x32xf16, {order = #NWHC}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW
    // CHECK-SAME:  } : tensor<1x224x48x32xf16, {order = #NWHC}> -> tensor<1x224x48x32xf16>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 32, 48, 224]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x224x48x32xf16>) -> tensor<1x32x48x224xf16>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x32x48x224xf16>
}

// -----

// CHECK-LABEL: @MemPermuteNCHWInNCHWOutNHWCPerm
func.func @MemPermuteNCHWInNCHWOutNHWCPerm(%arg0: tensor<1x32x48x64xf16>)
        -> tensor<1x48x64x32xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    } : tensor<1x32x48x64xf16> -> tensor<1x48x64x32xf16>

    return %MEM_PERMUTE : tensor<1x48x64x32xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x32x48x64xf16>
    // CHECK-SAME:      -> tensor<1x32x48x64xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 64, 32, 48]
    // CHECK-SAME:  } inputs([[IN_LAYOUT_CAST]] : tensor<1x32x48x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x64x32x48xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x32x48xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x32x48xf16, {order = #NWCH}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW
    // CHECK-SAME:  } : tensor<1x64x32x48xf16, {order = #NWCH}> -> tensor<1x64x32x48xf16>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 48, 64, 32]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x64x32x48xf16>) -> tensor<1x48x64x32xf16>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x48x64x32xf16>
}

// -----

// CHECK-LABEL: @MemPermuteNCHWInNCHWOutNHCWPerm
func.func @MemPermuteNCHWInNCHWOutNHCWPerm(%arg0: tensor<1x32x48x224xf16>)
        -> tensor<1x48x32x224xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
    } : tensor<1x32x48x224xf16> -> tensor<1x48x32x224xf16>

    return %MEM_PERMUTE : tensor<1x48x32x224xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x32x48x224xf16>
    // CHECK-SAME:      -> tensor<1x32x48x224xf16, {order = #NHWC}

    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 224, 32, 48]
    // CHECK-SAME:  } inputs([[IN_LAYOUT_CAST]] : tensor<1x32x48x224xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x224x32x48xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x224x32x48xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x224x32x48xf16, {order = #NWHC}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW
    // CHECK-SAME:  } : tensor<1x224x32x48xf16, {order = #NWHC}> -> tensor<1x224x32x48xf16>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 48, 32, 224]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x224x32x48xf16>) -> tensor<1x48x32x224xf16>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x48x32x224xf16>
}

// -----

// CHECK-LABEL: @MemPermuteNCWHInNHWCOutNWHCPerm
func.func @MemPermuteNCWHInNHWCOutNWHCPerm(%arg0: tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>}>)
        -> tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
    } : tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>}>
        -> tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    return %MEM_PERMUTE : tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x32x48x64xf16, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x32x48x64xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 48, 32, 64]
    // CHECK-SAME:  } inputs(%0 : tensor<1x32x48x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x48x32x64xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x32x64xf16, {order = #NCWH}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x48x32x64xf16, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x48x32x64xf16, {order = #NHWC}>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 32, 48, 64]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x48x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x32x48x64xf16, {order = #NHWC}>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x32x48x64xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @MemPermuteNCHWInNCHWOutNCWHPerm
func.func @MemPermuteNCHWInNCHWOutNCWHPerm(%arg0: tensor<1x8x1500x64xf16>) -> tensor<1x8x64x1500xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
    } : tensor<1x8x1500x64xf16> -> tensor<1x8x64x1500xf16>

    return %MEM_PERMUTE : tensor<1x8x64x1500xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x8x1500x64xf16>
    // CHECK-SAME:      -> tensor<1x8x1500x64xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 64, 8, 1500]
    // CHECK-SAME:  } inputs([[IN_LAYOUT_CAST]] : tensor<1x8x1500x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x64x8x1500xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x8x1500xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x8x1500xf16, {order = #NHCW}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW
    // CHECK-SAME:  } : tensor<1x64x8x1500xf16, {order = #NHCW}>
    // CHECK-SAME:      -> tensor<1x64x8x1500xf16>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 64, 1500]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x64x8x1500xf16>)
    // CHECK-SAME:      -> tensor<1x8x64x1500xf16>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x64x1500xf16>
}

// -----

// CHECK-LABEL: @MemPermuteWithMisalignedShape
func.func @MemPermuteWithMisalignedShape(%arg0: tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>}>)
        -> tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
    } : tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>}> -> tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    return %MEM_PERMUTE : tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x32x47x64xf16, {order = #NCWH}> -> tensor<1x32x47x64xf16, {order = #NHWC}>
    // CHECK:       [[IN_SHAPE_CAST0:%.*]] = IE.ShapeCast {shape = [1, 47, 32, 64]} inputs([[IN_LAYOUT_CAST]] : tensor<1x32x47x64xf16, {order = #NHWC}>) -> tensor<1x47x32x64xf16, {order = #NHWC}>
    // CHECK:       [[IN_SHAPE_CAST1:%.*]] = IE.ShapeCast {shape = [1, 16, 32, 188]} inputs([[IN_SHAPE_CAST0]] : tensor<1x47x32x64xf16, {order = #NHWC}>) -> tensor<1x16x32x188xf16, {order = #NHWC}>
    // CHECK:       [[MAX_POOL:%.*]] = IE.MaxPool([[IN_SHAPE_CAST1]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x32x188xf16, {order = #NHWC}> -> tensor<1x16x32x188xf16, {order = #NWCH}>
    // CHECK:       [[MID_LAYOUT_CAST:%.*]] = IE.LayoutCast([[MAX_POOL]]) {dst_order = #NHWC} : tensor<1x16x32x188xf16, {order = #NWCH}> -> tensor<1x16x32x188xf16, {order = #NHWC}>
    // CHECK:       [[MID_SHAPE_CAST0:%.*]] = IE.ShapeCast {shape = [1, 32, 64, 47]} inputs([[MID_LAYOUT_CAST]] : tensor<1x16x32x188xf16, {order = #NHWC}>) -> tensor<1x32x64x47xf16, {order = #NHWC}>
    // CHECK:       [[MAX_POOL2:%.*]] = IE.MaxPool([[MID_SHAPE_CAST0]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x32x64x47xf16, {order = #NHWC}> -> tensor<1x32x64x47xf16, {order = #NWHC}>
    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[MAX_POOL2]]) {dst_order = #NHWC} : tensor<1x32x64x47xf16, {order = #NWHC}> -> tensor<1x32x64x47xf16, {order = #NHWC}>
    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 32, 47, 64]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x32x64x47xf16, {order = #NHWC}>) -> tensor<1x32x47x64xf16, {order = #NHWC}>
    // CHECK:       return [[OUT_SHAPE_CAST]] : tensor<1x32x47x64xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SkipTrivialMemPermute
func.func @SkipTrivialMemPermute(%arg0: tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>)
        -> tensor<1x48x64x32xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    } : tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        -> tensor<1x48x64x32xf16>

    return %MEM_PERMUTE : tensor<1x48x64x32xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnsupportedDimN
func.func @UnsupportedDimN(%arg0: tensor<25x14x14x2304xf16>) -> tensor<25x14x2304x14xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<25x14x14x2304xf16> -> tensor<25x14x2304x14xf16>

    return %MEM_PERMUTE : tensor<25x14x2304x14xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} :
    // CHECK-SAME:  tensor<25x14x14x2304xf16> -> tensor<25x14x2304x14xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<25x14x2304x14xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithAlignChannel(%arg0: tensor<1x32x255x511xf16, {order = #NHWC}>) -> tensor<1x32x255x511xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x32x255x511xf16, {order = #NHWC}> -> tensor<1x32x255x511xf16>
    return %MEM_PERMUTE : tensor<1x32x255x511xf16>

    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:        kernel_size = [1, 1],
    // CHECK-SAME:        pads_begin = [0, 0],
    // CHECK-SAME:        pads_end = [0, 0],
    // CHECK-SAME:        rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:        strides = [1, 1]} : tensor<1x32x255x511xf16, {order = #NHWC}> -> tensor<1x32x255x511xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x32x255x511xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithUnalignedChannel
func.func @MemPermuteNHWCInNCHWOutWithUnalignedChannel(%arg0: tensor<1x3x256x512xf16, {order = #NHWC}>) -> tensor<1x3x256x512xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x256x512xf16, {order = #NHWC}> -> tensor<1x3x256x512xf16>
    return %MEM_PERMUTE : tensor<1x3x256x512xf16>

    // CHECK:       [[SHAPE_CAST_WC_IN_W:%.*]] = IE.ShapeCast {shape = [1, 16, 256, 96]} inputs(%arg0 : tensor<1x3x256x512xf16, {order = #NHWC}>) -> tensor<1x16x256x96xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.*]] = IE.MaxPool([[SHAPE_CAST_WC_IN_W]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x256x96xf16, {order = #NHWC}> -> tensor<1x16x256x96xf16, {order = #NWCH}>
    // CHECK:       [[LAYOUT_CAST_0:%.*]] = IE.LayoutCast([[MAXPOOL_0]]) {dst_order = #NHWC} : tensor<1x16x256x96xf16, {order = #NWCH}> -> tensor<1x16x256x96xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_WC_IN_H:%.*]] = IE.ShapeCast {shape = [1, 256, 512, 3]} inputs([[LAYOUT_CAST_0]] : tensor<1x16x256x96xf16, {order = #NHWC}>) -> tensor<1x256x512x3xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.*]] = IE.MaxPool([[SHAPE_CAST_WC_IN_H]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x256x512x3xf16, {order = #NHWC}> -> tensor<1x256x512x3xf16, {order = #NWCH}>
    // CHECK:       [[LAYOUT_CAST_1:%.*]] = IE.LayoutCast([[MAXPOOL_1]]) {dst_order = #NHWC} : tensor<1x256x512x3xf16, {order = #NWCH}> -> tensor<1x256x512x3xf16, {order = #NHWC}>
    // CHECK:       [[LAYOUT_CAST_2:%.*]] = IE.LayoutCast([[LAYOUT_CAST_1]]) {dst_order = #NCHW} : tensor<1x256x512x3xf16, {order = #NHWC}> -> tensor<1x256x512x3xf16>
    // CHECK:       [[RESULT:%.*]] = IE.ShapeCast {shape = [1, 3, 256, 512]} inputs([[LAYOUT_CAST_2]] : tensor<1x256x512x3xf16>) -> tensor<1x3x256x512xf16>
    // CHECK:       return [[RESULT]] : tensor<1x3x256x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithHCNotAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithHCNotAlignChannel(%arg0: tensor<1x3x255x512xf16, {order = #NHWC}>) -> tensor<1x3x255x512xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x255x512xf16, {order = #NHWC}> -> tensor<1x3x255x512xf16>
    return %MEM_PERMUTE : tensor<1x3x255x512xf16>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
    // CHECK-SAME:  tensor<1x3x255x512xf16, {order = #NHWC}> -> tensor<1x3x255x512xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x255x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel(%arg0: tensor<1x3x256x511xf16, {order = #NHWC}>) -> tensor<1x3x256x511xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    return %MEM_PERMUTE : tensor<1x3x256x511xf16>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
    // CHECK-SAME:  tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x256x511xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNWCHInNHWCOut
func.func @MemPermuteNWCHInNHWCOut(%arg0: tensor<1x16x48x289xf16, {order = #NWCH}>) -> tensor<1x16x48x289xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x48x289xf16, {order = #NWCH}> -> tensor<1x16x48x289xf16, {order = #NHWC}>
    return %MEM_PERMUTE : tensor<1x16x48x289xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x16x48x289xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x16x48x289xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 48, 289, 16]
    // CHECK-SAME:  } inputs([[IN_LAYOUT_CAST]] : tensor<1x16x48x289xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x48x289x16xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x289x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x289x16xf16>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x48x289x16xf16>
    // CHECK-SAME:      -> tensor<1x48x289x16xf16, {order = #NHWC}>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 48, 289]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x48x289x16xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x48x289xf16, {order = #NHWC}>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x16x48x289xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNWCHInNCHWOut
func.func @MemPermuteNWCHInNCHWOut(%arg0: tensor<1x16x48x289xf16, {order = #NWCH}>) -> tensor<1x16x48x289xf16, {order = #NCHW}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x48x289xf16, {order = #NWCH}> -> tensor<1x16x48x289xf16, {order = #NCHW}>
    return %MEM_PERMUTE : tensor<1x16x48x289xf16, {order = #NCHW}>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x16x48x289xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x16x48x289xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 48, 289, 16]
    // CHECK-SAME:  } inputs([[IN_LAYOUT_CAST]] : tensor<1x16x48x289xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x48x289x16xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x289x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x289x16xf16, {order = #NWCH}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW
    // CHECK-SAME:  } : tensor<1x48x289x16xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x48x289x16xf16>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 48, 289]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x48x289x16xf16>)
    // CHECK-SAME:      -> tensor<1x16x48x289xf16, {order = #NCHW}>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x16x48x289xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>


// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel(%arg0: tensor<1x3x256x511xf16, {order = #NHWC}>) -> tensor<1x3x256x511xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    return %MEM_PERMUTE : tensor<1x3x256x511xf16>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
    // CHECK-SAME:  tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x256x511xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @UnsupportedMemPermuteForUnalignedChannel
func.func @UnsupportedMemPermuteForUnalignedChannel(%arg0: tensor<1x4x20x36xf16, {order = #NHWC}>) -> tensor<1x4x20x36xf16, {order = #NCWH}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x4x20x36xf16, {order = #NHWC}> -> tensor<1x4x20x36xf16, {order = #NCWH}>
    return %MEM_PERMUTE : tensor<1x4x20x36xf16, {order = #NCWH}>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NWHC} :
    // CHECK-SAME:      tensor<1x4x20x36xf16, {order = #NHWC}> -> tensor<1x4x20x36xf16, {order = #NCWH}>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x4x20x36xf16, {order = #NCWH}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL: @MemPermuteDimWandDimHCAligned
func.func @MemPermuteDimWandDimHCAligned(%arg0: tensor<1x4x20x48xf16, {order = #NHWC}>) -> tensor<1x4x20x48xf16, {order = #NCWH}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x4x20x48xf16, {order = #NHWC}> -> tensor<1x4x20x48xf16, {order = #NCWH}>
    return %MEM_PERMUTE : tensor<1x4x20x48xf16, {order = #NCWH}>

    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 20, 12]} inputs(%arg0 : tensor<1x4x20x48xf16, {order = #NHWC}>) -> tensor<1x16x20x12xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.*]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x20x12xf16, {order = #NHWC}> -> tensor<1x16x20x12xf16, {order = #NWCH}>
    // CHECK:       [[LAYOUT_CAST_0:%.*]] = IE.LayoutCast([[MAXPOOL_0]]) {dst_order = #NHWC} : tensor<1x16x20x12xf16, {order = #NWCH}> -> tensor<1x16x20x12xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 48, 5]} inputs([[LAYOUT_CAST_0]] : tensor<1x16x20x12xf16, {order = #NHWC}>) -> tensor<1x16x48x5xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.*]] = IE.MaxPool([[SHAPE_CAST_0]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x48x5xf16, {order = #NHWC}> -> tensor<1x16x48x5xf16, {order = #NWCH}>
    // CHECK:       [[LAYOUT_CAST_1:%.*]] = IE.LayoutCast([[MAXPOOL_1]]) {dst_order = #NHWC} : tensor<1x16x48x5xf16, {order = #NWCH}> -> tensor<1x16x48x5xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_1:%.*]] = IE.ShapeCast {shape = [1, 48, 4, 20]} inputs([[LAYOUT_CAST_1]] : tensor<1x16x48x5xf16, {order = #NHWC}>) -> tensor<1x48x4x20xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_2:%.*]] = IE.MaxPool([[SHAPE_CAST_1]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x48x4x20xf16, {order = #NHWC}> -> tensor<1x48x4x20xf16, {order = #NHCW}>
    // CHECK:       [[LAYOUT_CAST_2:%.*]] = IE.LayoutCast([[MAXPOOL_2]]) {dst_order = #NHWC} : tensor<1x48x4x20xf16, {order = #NHCW}> -> tensor<1x48x4x20xf16, {order = #NHWC}>
    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LAYOUT_CAST_2]]) {dst_order = #NCWH} : tensor<1x48x4x20xf16, {order = #NHWC}> -> tensor<1x48x4x20xf16, {order = #NCWH}>
    // CHECK:       [[RET:%.*]] = IE.ShapeCast {shape = [1, 4, 20, 48]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x48x4x20xf16, {order = #NCWH}>) -> tensor<1x4x20x48xf16, {order = #NCWH}>
    // CHECK:       return [[RET]] : tensor<1x4x20x48xf16, {order = #NCWH}>
}
