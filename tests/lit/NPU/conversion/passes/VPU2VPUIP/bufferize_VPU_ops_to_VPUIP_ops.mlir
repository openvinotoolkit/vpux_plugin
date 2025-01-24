//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL:  func.func @ConstantLayer
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x2x2x2xf16>)
func.func @ConstantLayer(%input: tensor<1x2x2x2xf16>) -> tensor<1x2x2x2xf16> {
    %cmx = VPU.Copy(%input) {out_mem_space = @CMX_NN} : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16, {mem_space = @CMX_NN}>
    %ddr = VPU.Copy(%cmx) {out_mem_space = @DDR} : tensor<1x2x2x2xf16, {mem_space = @CMX_NN}> -> tensor<1x2x2x2xf16>
    return %ddr : tensor<1x2x2x2xf16>

    // CHECK: [[BUFFER_CMX:%.+]] = memref.alloc() : memref<1x2x2x2xf16, @CMX_NN>
    // CHECK: [[ARG_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x2x2x2xf16>) outputs([[BUFFER_CMX]] : memref<1x2x2x2xf16, @CMX_NN>) -> memref<1x2x2x2xf16, @CMX_NN>
    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x2x2x2xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.Copy inputs([[ARG_CMX]] : memref<1x2x2x2xf16, @CMX_NN>) outputs([[BUFFER_DDR]] : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
    // CHECK: return [[OUTPUT]] : memref<1x2x2x2xf16>
}

// -----
// CHECK-LABEL:  func.func @Expand
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @Expand(%input: tensor<1x3x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %output = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x3x4x4xf16> -> tensor<1x8x4x4xf16>
    return %output : tensor<1x8x4x4xf16>

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x4x4xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} inputs([[ARG]] : memref<1x3x4x4xf16>) outputs([[BUFFER_DDR]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>
    // CHECK: return [[OUTPUT]] : memref<1x8x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ExpandToSubviewWithoutTail
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x4x4x4xf16>)
func.func @ExpandToSubviewWithoutTail(%input: tensor<1x4x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x4x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x4x4xf16>
    // CHECK:       [[OUTPUT:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]}
    // CHECK-SAME:      inputs([[ARG]] : memref<1x4x4x4xf16>) outputs([[BUFFER_DDR]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>
    // CHECK:       return [[OUTPUT]] : memref<1x8x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ExpandToSubviewOnlyWithTail
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x5x4x4xf16>)
func.func @ExpandToSubviewOnlyWithTail(%input: tensor<1x5x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x5x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x4x4xf16>
    // CHECK:       [[OUTPUT:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]}
    // CHECK-SAME:     inputs([[ARG]] : memref<1x5x4x4xf16>) outputs([[BUFFER_DDR]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>
    // CHECK:       return [[OUTPUT]] : memref<1x8x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ExpandOverWidth
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @ExpandOverWidth(%input: tensor<1x3x4x4xf16>) -> tensor<1x3x4x9xf16> {
    %0 = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 5]} : tensor<1x3x4x4xf16> -> tensor<1x3x4x9xf16>
    return %0 : tensor<1x3x4x9xf16>

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x3x4x9xf16>
    // CHECK:       [[OUTPUT:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 5]}
    // CHECK-SAME:     inputs([[ARG]] : memref<1x3x4x4xf16>) outputs([[BUFFER_DDR]] : memref<1x3x4x9xf16>) -> memref<1x3x4x9xf16>
    // CHECK:       return [[OUTPUT]] : memref<1x3x4x9xf16>
}

// -----
// CHECK-LABEL:  func.func @ExpandOverHeight
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @ExpandOverHeight(%input: tensor<1x3x4x4xf16>) -> tensor<1x3x9x4xf16> {
    %0 = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]} : tensor<1x3x4x4xf16> -> tensor<1x3x9x4xf16>
    return %0 : tensor<1x3x9x4xf16>

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x3x9x4xf16>
    // CHECK:       [[OUTPUT:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]}
    // CHECK-SAME:     inputs([[ARG]] : memref<1x3x4x4xf16>) outputs([[BUFFER_DDR]] : memref<1x3x9x4xf16>) -> memref<1x3x9x4xf16>
    // CHECK:       return [[OUTPUT]] : memref<1x3x9x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ExpandPadsBeginFullCopy
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @ExpandPadsBeginFullCopy(%input: tensor<1x3x4x4xf16>) -> tensor<1x6x4x4xf16> {
    %0 = VPU.Expand(%input) {
        pads_begin = [0, 3, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x3x4x4xf16> -> tensor<1x6x4x4xf16>

    return %0 : tensor<1x6x4x4xf16>

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x6x4x4xf16>
    // CHECK:       [[OUTPUT:%.*]] = VPUIP.Expand {pads_begin = [0, 3, 0, 0], pads_end = [0, 0, 0, 0]}
    // CHECK-SAME:     inputs([[ARG]] : memref<1x3x4x4xf16>) outputs([[BUFFER_DDR]] : memref<1x6x4x4xf16>) -> memref<1x6x4x4xf16>
    // CHECK:       return [[OUTPUT]] : memref<1x6x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ExpandPadsBeginSliceCopy
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @ExpandPadsBeginSliceCopy(%input: tensor<1x3x4x4xf16>) -> tensor<1x5x4x4xf16> {
    %0 = VPU.Expand(%input) {
        pads_begin = [0, 2, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x3x4x4xf16> -> tensor<1x5x4x4xf16>

    return %0 : tensor<1x5x4x4xf16>

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x5x4x4xf16>

    // CHECK:       [[OUTPUT:%.*]] = VPUIP.Expand {pads_begin = [0, 2, 0, 0], pads_end = [0, 0, 0, 0]}
    // CHECK-SAME:     inputs([[ARG]] : memref<1x3x4x4xf16>) outputs([[BUFFER_DDR]] : memref<1x5x4x4xf16>) -> memref<1x5x4x4xf16>
    // CHECK:       return [[OUTPUT]] : memref<1x5x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ExpandPadsBeginCopiesWithTail
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @ExpandPadsBeginCopiesWithTail(%input: tensor<1x3x4x4xf16>) -> tensor<1x11x4x4xf16> {
    %0 = VPU.Expand(%input) {
        pads_begin = [0, 8, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x3x4x4xf16> -> tensor<1x11x4x4xf16>

    return %0 : tensor<1x11x4x4xf16>

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x11x4x4xf16>
    // CHECK:       [[OUTPUT:%.*]] = VPUIP.Expand {pads_begin = [0, 8, 0, 0], pads_end = [0, 0, 0, 0]}
    // CHECK-SAME:     inputs([[ARG]] : memref<1x3x4x4xf16>) outputs([[BUFFER_DDR]] : memref<1x11x4x4xf16>) -> memref<1x11x4x4xf16>
    // CHECK:       return [[OUTPUT]] : memref<1x11x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ExpandBeginPadsWithEndPads
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @ExpandBeginPadsWithEndPads(%input: tensor<1x3x4x4xf16>) -> tensor<1x9x4x4xf16> {
    %0 = VPU.Expand(%input) {
        pads_begin = [0, 3, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x3x4x4xf16> -> tensor<1x9x4x4xf16>

    return %0 : tensor<1x9x4x4xf16>

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x9x4x4xf16>
    // CHECK:       [[OUTPUT:%.*]] = VPUIP.Expand {pads_begin = [0, 3, 0, 0], pads_end = [0, 3, 0, 0]}
    // CHECK-SAME:     inputs([[ARG]] : memref<1x3x4x4xf16>) outputs([[BUFFER_DDR]] : memref<1x9x4x4xf16>) -> memref<1x9x4x4xf16>
    // CHECK:       return [[OUTPUT]] : memref<1x9x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @Reshape
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x640x640xf16>)
func.func @Reshape(%input: tensor<1x3x640x640xf16>) -> tensor<1x640x3x640xf16> {
    %output = VPU.Reshape(%input) {shape_value = [1, 640, 3, 640]} : tensor<1x3x640x640xf16> -> tensor<1x640x3x640xf16>
    return %output : tensor<1x640x3x640xf16>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.GenericReshape inputs([[ARG]] : memref<1x3x640x640xf16>) -> memref<1x640x3x640xf16>
    // CHECK: eturn [[OUTPUT]] : memref<1x640x3x640xf16>
}

// -----
// CHECK-LABEL:  func.func @Slice
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x8x60x60xf16>)
func.func @Slice(%input: tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>) {
    %slice0 = VPU.Slice %input [0, 2, 0, 0] [1, 4, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x4x60x60xf16>
    %slice1 = VPU.Slice %input [0, 4, 0, 0] [1, 2, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x2x60x60xf16>
    return %slice0, %slice1 : tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>

    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ARG]] [0, 2, 0, 0] [1, 4, 60, 60] : memref<1x8x60x60xf16> to memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>
    // CHECK: [[BUFFER_DDR_1:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
    // CHECK: [[OUTPUT_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1]] : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>) outputs([[BUFFER_DDR_1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ARG]] [0, 4, 0, 0] [1, 2, 60, 60] : memref<1x8x60x60xf16> to memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>
    // CHECK: [[BUFFER_DDR_2:%.+]] = memref.alloc() : memref<1x2x60x60xf16>
    // CHECK: [[OUTPUT_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>) outputs([[BUFFER_DDR_2]] : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16>
    // CHECK: return [[OUTPUT_1]], [[OUTPUT_2]] : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
}

// -----
// CHECK-LABEL:  func.func @Split
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x96x2x48xf16>)
func.func @Split(%input: tensor<1x96x2x48xf16>) -> tensor<1x48x2x48xf16> {
    %output:2 = VPU.Split(%input) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x96x2x48xf16> -> tensor<1x48x2x48xf16>, tensor<1x48x2x48xf16>
    return %output#0 : tensor<1x48x2x48xf16>

    // CHECK: [[BUFFER_DDR_1:%.+]] = memref.alloc() : memref<1x48x2x48xf16>
    // CHECK: [[BUFFER_DDR_2:%.+]] = memref.alloc() : memref<1x48x2x48xf16>
    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ARG]] [0, 0, 0, 0] [1, 48, 2, 48] : memref<1x96x2x48xf16> to memref<1x48x2x48xf16, {order = #NCHW, strides = [9216, 96, 48, 1]}>
    // CHECK: [[OUTPUT_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1]] : memref<1x48x2x48xf16, {order = #NCHW, strides = [9216, 96, 48, 1]}>) outputs([[BUFFER_DDR_1]] : memref<1x48x2x48xf16>) -> memref<1x48x2x48xf16>
    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ARG]] [0, 48, 0, 0] [1, 48, 2, 48] : memref<1x96x2x48xf16> to memref<1x48x2x48xf16, {order = #NCHW, strides = [9216, 96, 48, 1]}>
    // CHECK: [[OUTPUT_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x48x2x48xf16, {order = #NCHW, strides = [9216, 96, 48, 1]}>) outputs([[BUFFER_DDR_2]] : memref<1x48x2x48xf16>) -> memref<1x48x2x48xf16>
    // CHECK: return [[OUTPUT_1]] : memref<1x48x2x48xf16>

}

// -----
// CHECK-LABEL:  func.func @Concat
// CHECK-SAME:       ([[ARG0:%.+]]: memref<1x16x16x16xf16>, [[ARG1:%.+]]: memref<1x16x16x16xf16>)
func.func @Concat(%input0: tensor<1x16x16x16xf16>, %input1: tensor<1x16x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %output = VPU.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16> -> tensor<1x32x16x16xf16>
    return %output : tensor<1x32x16x16xf16>

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x32x16x16xf16>
    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 0, 0] [1, 16, 16, 16] : memref<1x32x16x16xf16> to memref<1x16x16x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}>
    // CHECK: [[PART_1:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x16x16x16xf16>) outputs([[SUBVIEW_1]] : memref<1x16x16x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}>) -> memref<1x16x16x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}>
    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 16, 0, 0] [1, 16, 16, 16] : memref<1x32x16x16xf16> to memref<1x16x16x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}>
    // CHECK: [[PART_2:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<1x16x16x16xf16>) outputs([[SUBVIEW_2]] : memref<1x16x16x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}>) -> memref<1x16x16x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.ConcatView inputs([[PART_1]], [[PART_2]] : memref<1x16x16x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}>, memref<1x16x16x16xf16, {order = #NCHW, strides = [8192, 256, 16, 1]}>) outputs(%alloc : memref<1x32x16x16xf16>) -> memref<1x32x16x16xf16>
    // CHECK: return [[OUTPUT]] : memref<1x32x16x16xf16>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteCast
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x320x1x1xf16>)
func.func @PermuteCast(%input: tensor<1x320x1x1xf16>) -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    %output = VPU.PermuteCast(%input) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>
    return %output :  tensor<1x320x1x1xf16, {order = #NHWC}>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[ARG]] : memref<1x320x1x1xf16>) -> memref<1x320x1x1xf16, #NHWC>
    // CHECK: return [[OUTPUT]] : memref<1x320x1x1xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16, 0.013744638480392158:128>

// CHECK-LABEL: @QuantizeCast
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x32x256x256x!qElemType, #NHWC>)
func.func @QuantizeCast(%input: tensor<1x32x256x256x!qElemType1, {order = #NHWC}>) -> (tensor<1x32x256x256x!qElemType, {order = #NHWC}>) {
    %output = VPU.QuantizeCast(%input) {dstElemType = !qElemType} : tensor<1x32x256x256x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
    return %output : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.QuantizeCast inputs([[ARG]] : memref<1x32x256x256x!qElemType, #NHWC>) -> memref<1x32x256x256x!qElemType1, #NHWC>
    // CHECK: return [[OUTPUT]] : memref<1x32x256x256x!qElemType1, #NHWC>

}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @StorageElementTable
func.func @StorageElementTable() -> tensor<1x1x16x16xi32, {order = #NHWC}> {
    %output = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16], seDepth = 1 : i64, seSize = 64 : i64} -> tensor<1x1x16x16xi32, {order = #NHWC}>
    return %output : tensor<1x1x16x16xi32, {order = #NHWC}>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16], seDepth = 1 : i64, seSize = 64 : i64} -> memref<1x1x16x16xi32, #NHWC>
    // CHECK: return [[OUTPUT]] : memref<1x1x16x16xi32, #NHWC>
}

// -----
// CHECK-LABEL: @ShapeCast
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x32x32xf16>)
func.func @ShapeCast(%input: tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16> {
    %output = VPU.ShapeCast {shape = [1, 16, 16, 12]} inputs(%input : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    return %output : tensor<1x16x16x12xf16>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 16, 12]} inputs([[ARG]] : memref<1x3x32x32xf16>) -> memref<1x16x16x12xf16>
    // CHECK: return [[OUTPUT]] : memref<1x16x16x12xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @LayoutCast
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x32x32xf16, #NWCH>)
func.func @LayoutCast(%input: tensor<1x3x32x32xf16, {order = #NWCH}>) -> tensor<1x3x32x32xf16, {order = #NHWC}> {
    %output = VPU.LayoutCast(%input) {dst_order = #NHWC} : tensor<1x3x32x32xf16, {order = #NWCH}> -> tensor<1x3x32x32xf16, {order = #NHWC}>
    return %output : tensor<1x3x32x32xf16, {order = #NHWC}>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[ARG]] : memref<1x3x32x32xf16, #NWCH>) -> memref<1x3x32x32xf16, #NHWC>
    // CHECK: return [[OUTPUT]] : memref<1x3x32x32xf16, #NHWC>
}

// -----

// CHECK-LABEL: @M2ITask
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x256x256xui8>)
func.func @M2ITask(%input: tensor<1x3x256x256xui8>) -> tensor<1x3x224x224xui8> {
    %0 = VPU.M2I.Task(%input) {axes = [2, 3], do_csc = false, do_norm = false, inFmt = #VPU.m2i_color_fmt<PL_RGB24>, outFmt = #VPU.m2i_color_fmt<PL_RGB24>, sizes = [224, 224]} -> tensor<1x3x224x224xui8>
    return %0 : tensor<1x3x224x224xui8>

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x3x224x224xui8>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.M2ITask {do_csc = false, do_norm = false, inFmt = #VPU.m2i_color_fmt<PL_RGB24>, outFmt = #VPU.m2i_color_fmt<PL_RGB24>, scale_factor_x = 149797 : ui32, scale_factor_y = 149797 : ui32} inputs([[ARG]] : memref<1x3x256x256xui8>) outputs([[BUFFER_DDR]] : memref<1x3x224x224xui8>) -> memref<1x3x224x224xui8>
    // CHECK: return [[OUTPUT]] : memref<1x3x224x224xui8>
}

// -----

// CHECK-LABEL:  func.func @StridedSlice1Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice1Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x40x40x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x40x5xf16>
    return %output : tensor<3x40x40x5xf16>

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [3, 40, 40, 5] [1, 1, 1, 3] : memref<3x40x40x15xf16> to memref<3x40x40x5xf16, {order = #NCHW, strides = [24000, 600, 15, 3]}>
    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<3x40x40x5xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<3x40x40x5xf16, {order = #NCHW, strides = [24000, 600, 15, 3]}>) outputs([[ALLOC]] : memref<3x40x40x5xf16>) -> memref<3x40x40x5xf16>
    // CHECK: return [[OUTPUT]] : memref<3x40x40x5xf16>
}

// -----

// CHECK-LABEL: @M2ITask
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x288x256x1xui8>)
func.func @M2ITask(%input: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xui8> {
    %0 = VPU.M2I.Task(%input) {axes = [2, 3], chroma_out_reverse_channels, do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<PL_RGB24>, sizes = [168, 224]} -> tensor<1x3x168x224xui8>
    return %0 : tensor<1x3x168x224xui8>

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x3x168x224xui8>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.M2ITask {chroma_out_reverse_channels, do_csc = true, do_norm = false, inFmt = #VPU.m2i_color_fmt<SP_NV12_8>, outFmt = #VPU.m2i_color_fmt<PL_RGB24>, scale_factor_x = 149797 : ui32, scale_factor_y = 149797 : ui32} inputs([[ARG]] : memref<1x288x256x1xui8>) outputs([[BUFFER_DDR]] : memref<1x3x168x224xui8>) -> memref<1x3x168x224xui8>
    // CHECK: return [[OUTPUT]] : memref<1x3x168x224xui8>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedTensor = !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!OutputDistributedTensor = !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK: func.func @DistributedCast({{[^:]+}}:
// CHECK-SAME: !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN,
// CHECK-SAME: {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>)
// CHECK-SAME: !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN,
// CHECK-SAME: {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
func.func @DistributedCast(%arg0: !InputDistributedTensor) -> !OutputDistributedTensor {
    %1 = VPU.DistributedCast(%arg0 : !InputDistributedTensor) -> !OutputDistributedTensor
    return %1 : !OutputDistributedTensor

    // CHECK: VPUIP.DistributedCast
    // CHECK-SAME: inputs({{[^:]+}} : !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>) ->
    // CHECK-SAME: !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:  return
    // CHECK-SAME: !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME: {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x4x8x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x1x1x512xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

// CHECK: func.func @SliceSparseTensor({{[^:]+}}: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, {{[^:]+}}: memref<1x32x16x16xi1, #NHWC, @CMX_NN>)
// CHECK-SAME: !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>
func.func @SliceSparseTensor(%arg0: tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>, %arg1: tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>) -> !OutputSparseTensor {
    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1) {is_weights} -> !InputSparseTensor
    %output = VPU.Slice %input_sparse [0, 0, 0, 0] [1, 4, 8, 16]: !InputSparseTensor to !OutputSparseTensor
    return %output: !OutputSparseTensor

    // CHECK: [[SUBVIEW_RES:%.+]] = VPUIP.SubView {{%.+}} [0, 0, 0, 0] [1, 4, 8, 16] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK-SAME:                                                                      to
    // CHECK-SAME:                                                                     !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                         sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK: [[ALLOC_DATA_BUF:%.+]] = memref.alloc() : memref<1x4x8x16xf16, #NHWC, @CMX_NN>
    // CHECK: [[ALLOC_SM_BUF:%.+]] = memref.alloc() : memref<1x1x1x512xi1, #NHWC, @CMX_NN>
    // CHECK: [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]]) {is_weights} -> !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK: {{%.+}} = VPUIP.Copy inputs([[SUBVIEW_RES]] : !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                              sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                 outputs([[ALLOC_SPARSE_BUF]] : !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                 ->
    // CHECK-SAME:                 !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK: return
    // CHECK-SAME: !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2}
>

!InputSMTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2}
>

!InputSparseTensorDistributed = !VPU.SparseTensor<
    data=!InputTensorDistributed, sparsity_map=!InputSMTensorDistributed, is_weights
>

!OutputTensorDistributed = !VPU.DistributedTensor<
    1x4x8x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSMTensorDistributed = !VPU.DistributedTensor<
    1x1x1x512xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSparseTensorDistributed = !VPU.SparseTensor<
    data=!OutputTensorDistributed, sparsity_map=!OutputSMTensorDistributed, is_weights
>

// CHECK-LABEL: @SliceSparseDistributedTensor
func.func @SliceSparseDistributedTensor(%arg0: !InputTensorDistributed, %arg1: !InputSMTensorDistributed) -> !OutputSparseTensorDistributed {
    %st = VPU.GroupSparseTensor(%arg0, %arg1) {is_weights} -> !InputSparseTensorDistributed
    %output = VPU.Slice %st [0, 0, 0, 0] [1, 4, 8, 16]: !InputSparseTensorDistributed to !OutputSparseTensorDistributed
    return %output: !OutputSparseTensorDistributed

    // CHECK:       [[SUBVIEW_RES:%.+]] = VPUIP.SubView {{%.+}} [0, 0, 0, 0] [1, 4, 8, 16] :
    // CHECK-SAME:                           !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                              sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                              is_weights> to
    // CHECK-SAME:                           !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                               sparsity_map=!VPUIP.DistributedBuffer<1x1x1x512xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                               is_weights>

    // CHECK:       [[ALLOC_DATA_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x8x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ALLOC_SM_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x512xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]]) {is_weights} -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x4x8x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                                                               sparsity_map=!VPUIP.DistributedBuffer<1x1x1x512xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                                                               is_weights>

    // CHECK:       %{{.+}} = VPUIP.NCEClusterTiling inputs([[SUBVIEW_RES]] as [[ARG2:%.+]]: !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                               sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                   outputs([[ALLOC_SPARSE_BUF]] as [[ARG3:%.+]]: !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:            -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x4x8x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                   sparsity_map=!VPUIP.DistributedBuffer<1x1x1x512xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                   is_weights>
    // CHECK-SAME:   {
    // CHECK:           {{%.+}} = VPUIP.Copy inputs([[ARG2]] : !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                 sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                           outputs([[ARG3]] : !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                           -> !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK:        }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!InputSMTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!InputSparseTensorDistributed = !VPU.SparseTensor<
    data=!InputTensorDistributed, sparsity_map=!InputSMTensorDistributed
>

!OutputTensorDistributed = !VPU.DistributedTensor<
    1x32x8x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 4, 16], [1, 32, 4, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 32, 5, 16], [1, 32, 5, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!OutputSMTensorDistributed = !VPU.DistributedTensor<
    1x32x8x16xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 4, 16], [1, 32, 4, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 32, 5, 16], [1, 32, 5, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!OutputSparseTensorDistributed = !VPU.SparseTensor<
    data=!OutputTensorDistributed, sparsity_map=!OutputSMTensorDistributed
>

// CHECK-LABEL: @SliceSparseExplicitDistributedBuf
func.func @SliceSparseExplicitDistributedBuf(%arg0: !InputTensorDistributed, %arg1: !InputSMTensorDistributed) -> !OutputSparseTensorDistributed {
    %st = VPU.GroupSparseTensor(%arg0, %arg1) -> !InputSparseTensorDistributed
    %output = VPU.Slice %st [0, 0, 8, 0] [1, 32, 8, 16]: !InputSparseTensorDistributed to !OutputSparseTensorDistributed
    return %output: !OutputSparseTensorDistributed

    // CHECK:       [[SUBVIEW_RES:%.+]] = VPUIP.SubView {{%.+}} [0, 0, 8, 0] [1, 32, 8, 16] :
    // CHECK-SAME:             !VPUIP.SparseBuffer<
    // CHECK-SAME:                  data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]}
    // CHECK-SAME:                  sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]}
    // CHECK-SAME:          to !VPUIP.SparseBuffer<
    // CHECK-SAME:                  data=!VPUIP.DistributedBuffer<1x32x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 4, 16], [1, 32, 4, 16]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 5, 16], [1, 32, 5, 16]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]}
    // CHECK-SAME:                  sparsity_map=!VPUIP.DistributedBuffer<1x32x8x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 4, 16], [1, 32, 4, 16]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 5, 16], [1, 32, 5, 16]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]}

    // CHECK:       [[ALLOC_DATA_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x32x8x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                 {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 32, 4, 16], [1, 32, 4, 16]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 32, 5, 16], [1, 32, 5, 16]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]}

    // CHECK:       [[ALLOC_SM_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:           -> !VPUIP.DistributedBuffer<1x32x8x16xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                 {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 32, 4, 16], [1, 32, 4, 16]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 32, 5, 16], [1, 32, 5, 16]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]}
    // CHECK:       [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]])

    // CHECK:       VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[SUBVIEW_RES]] as [[ARG2:%.+]]: !VPUIP.SparseBuffer<data=memref<1x32x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                  sparsity_map=memref<1x32x8x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>)
    // CHECK-SAME:      outputs([[ALLOC_SPARSE_BUF]] as [[ARG3:%.+]]: !VPUIP.SparseBuffer<data=memref<1x32x8x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                                                                        sparsity_map=memref<1x32x8x16xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<
    // CHECK-SAME:                 data=!VPUIP.DistributedBuffer<1x32x8x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                   {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 4, 16], [1, 32, 4, 16]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 5, 16], [1, 32, 5, 16]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]}
    // CHECK-SAME:                 sparsity_map=!VPUIP.DistributedBuffer<1x32x8x16xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                   {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 4, 16], [1, 32, 4, 16]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 5, 16], [1, 32, 5, 16]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]}
    // CHECK:               VPUIP.Copy inputs([[ARG2]] : !VPUIP.SparseBuffer<data=memref<1x32x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                          sparsity_map=memref<1x32x8x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>)
    // CHECK-SAME:                      outputs([[ARG3]] : !VPUIP.SparseBuffer<data=memref<1x32x8x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                                                            sparsity_map=memref<1x32x8x16xi1, #NHWC, @CMX_NN>>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputTensorDistributed = !VPU.DistributedTensor<
    1x64x8x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!OutputTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @Concat
func.func @Concat(%arg0: !InputTensorDistributed, %arg1: !InputTensorDistributed) -> !OutputTensorDistributed {
    %output = VPU.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]}: !InputTensorDistributed, !InputTensorDistributed -> !OutputTensorDistributed
    return %output : !OutputTensorDistributed

    // CHECK:       [[ALLOC_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC_BUF]] [0, 0, 0, 0] [1, 64, 8, 16] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}> to !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       [[CLUSTER_COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[SUBVIEW0]] as [[ARG2:%.+]]: memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG1]] : memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ARG2]] : memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)


    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC_BUF]] [0, 0, 8, 0] [1, 64, 8, 16] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}> to !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       [[CLUSTER_COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs(%arg1 as [[ARG3:%.+]]: memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[SUBVIEW1]] as [[ARG4:%.+]]: memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG3]] : memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ARG4]] : memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)


    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:       inputs([[CLUSTER_COPY0]], [[CLUSTER_COPY1]] : !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>, !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:       outputs([[ALLOC_BUF]] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       return [[CONCAT]] :
    // CHECK-SAME:       !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x32x24x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x32x24x16xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

// CHECK-LABEL: @ConcatSparseTensor
func.func @ConcatSparseTensor(%arg0: tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>, %arg1: tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>,
                            %arg2: tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>, %arg3: tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>) -> !OutputSparseTensor {
    %st1 = VPU.GroupSparseTensor(%arg0, %arg1) {is_weights} -> !InputSparseTensor
    %st2 = VPU.GroupSparseTensor(%arg2, %arg3) {is_weights} -> !InputSparseTensor
    %res = VPU.Concat(%st1, %st2) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]}: !InputSparseTensor, !InputSparseTensor -> !OutputSparseTensor
    return %res : !OutputSparseTensor

    // CHECK:       [[ALLOC_DATA_BUF:%.+]] = memref.alloc() : memref<1x32x24x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[ALLOC_SM_BUF:%.+]] = memref.alloc() : memref<1x32x24x16xi1, #NHWC, @CMX_NN>
    // CHECK:       [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]]) {is_weights} ->
    // CHECK-SAME:                             !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       [[SUBVIEW_1_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 0, 0] [1, 32, 16, 16] :
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK-SAME:                              to
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                  sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       [[COPY_1_RES:%.+]] = VPUIP.Copy inputs({{%.+}} : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                  outputs([[SUBVIEW_1_RES]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                  sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>,  is_weights>)
    // CHECK-SAME:                                  -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                         sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       [[SUBVIEW_2_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 8, 0] [1, 32, 16, 16] :
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK-SAME:                              to
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                  sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       [[COPY_2_RES:%.+]] = VPUIP.Copy inputs({{%.+}} : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                  outputs([[SUBVIEW_2_RES]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                  sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                  -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                         sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       {{%.+}} = VPUIP.ConcatView inputs([[COPY_1_RES]], [[COPY_2_RES]] :
    // CHECK-SAME:                                    !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                        sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>,
    // CHECK-SAME:                                    !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                        sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                             outputs([[ALLOC_SPARSE_BUF]] : !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                             -> !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 }
>

!InputSMTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED",  num_tiles = [1, 1, 1, 2], num_clusters = 2}
>

!InputSparseTensorDistributed = !VPU.SparseTensor<
    data=!InputTensorDistributed,
    sparsity_map=!InputSMTensorDistributed,
    is_weights
>

!OutputTensorDistributed = !VPU.DistributedTensor<
    1x32x32x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64 }
>

!OutputSMTensorDistributed = !VPU.DistributedTensor<
    1x32x32x16xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64 }
>

!OutputSparseTensorDistributed = !VPU.SparseTensor<
    data=!OutputTensorDistributed,
    sparsity_map=!OutputSMTensorDistributed,
    is_weights
>

// CHECK-LABEL: @ConcatSparseDistributedTensor
func.func @ConcatSparseDistributedTensor(%arg0: !InputTensorDistributed, %arg1: !InputSMTensorDistributed,
                                 %arg2: !InputTensorDistributed, %arg3: !InputSMTensorDistributed) -> !OutputSparseTensorDistributed {
    %st1 = VPU.GroupSparseTensor(%arg0, %arg1) {is_weights} -> !InputSparseTensorDistributed
    %st2 = VPU.GroupSparseTensor(%arg2, %arg3) {is_weights} -> !InputSparseTensorDistributed

    %res = VPU.Concat(%st1, %st2) {static_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]]}: !InputSparseTensorDistributed, !InputSparseTensorDistributed -> !OutputSparseTensorDistributed
    return %res : !OutputSparseTensorDistributed

    // CHECK:       [[ALLOC_IN0_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer(%arg0, %arg1) {is_weights}
    // CHECK-SAME:         -> !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:              is_weights>

    // CHECK:       [[ALLOC_IN1_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer(%arg2, %arg3) {is_weights}
    // CHECK-SAME:         -> !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:              is_weights>

    // CHECK:       [[ALLOC_DATA_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}

    // CHECK:       [[ALLOC_SM_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}

    // CHECK:       [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]]) {is_weights}
    // CHECK-SAME:         -> !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:              is_weights>

    // CHECK:       [[SUBVIEW_1_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 0, 0] [1, 32, 16, 16] :
    // CHECK-SAME:            !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:              is_weights>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:              is_weights>

    // CHECK:       [[COPY_1_RES:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[ALLOC_IN0_SPARSE_BUF]] as [[ARG2:[^:]+]]:
    // CHECK-SAME:                      !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                                          sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:          outputs([[SUBVIEW_1_RES]] as [[ARG3:[^:]+]]:
    // CHECK-SAME:                      !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                          sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<
    // CHECK-SAME:               data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:               sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:               is_weights> {

    // CHECK:                {{%.+}} = VPUIP.Copy
    // CHECK-SAME:               inputs([[ARG2]] : !VPUIP.SparseBuffer<
    // CHECK-SAME:                                   data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                                   sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:               outputs([[ARG3]] : !VPUIP.SparseBuffer<
    // CHECK-SAME:                                    data=memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                    sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:               -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                 sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       [[SUBVIEW_2_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 16, 0] [1, 32, 16, 16] :
    // CHECK-SAME:            !VPUIP.SparseBuffer<
    // CHECK-SAME:               data=!VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:               sparsity_map=!VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}
    // CHECK-SAME:               is_weights>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<
    // CHECK-SAME:               data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                      {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:               sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                      {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:               is_weights>

    // CHECK:       [[COPY_2_RES:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[ALLOC_IN1_SPARSE_BUF]] as [[ARG4:[^:]+]]:
    // CHECK-SAME:             !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:         outputs([[SUBVIEW_2_RES]] as [[ARG5:[^:]+]]:
    // CHECK-SAME:             !VPUIP.SparseBuffer<
    // CHECK-SAME:                 data=memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                 sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAme:         -> !VPUIP.SparseBuffer<
    // CHECK-SAME:               data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:               sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:               is_weights> {

    // CHECK:                {{%.+}} = VPUIP.Copy
    // CHECK-SAME:               inputs([[ARG4]] : !VPUIP.SparseBuffer<
    // CHECK-SAME:                                    data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                                    sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:               outputs([[ARG5]] : !VPUIP.SparseBuffer<
    // CHECK-SAME:                                    data=memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                    sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:             -> !VPUIP.SparseBuffer<
    // CHECK-SAME:                   data=memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                   sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       {{%.+}} = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_1_RES]], [[COPY_2_RES]] :
    // CHECK-SAME:          !VPUIP.SparseBuffer<
    // CHECK-SAME:             data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                     {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:             sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:             is_weights>,
    // CHECK-SAME:          !VPUIP.SparseBuffer<
    // CHECK-SAME:             data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                     {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:             sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:             is_weights>)
    // CHECK-SAME:       outputs([[ALLOC_SPARSE_BUF]] :
    // CHECK-SAME:          !VPUIP.SparseBuffer<
    // CHECK-SAME:             data=!VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:             sparsity_map=!VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:             is_weights>)
    // CHECK-SAME:     -> !VPUIP.SparseBuffer<
    // CHECK-SAME:           data=!VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:           sparsity_map=!VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:           is_weights>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputBufferDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!OutputBufferDistributed = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 64, 8, 16], [1, 64, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!InputTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

!OutputTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    compute_shapes = [[1, 64, 8, 16], [1, 64, 8, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    memory_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
}>

// CHECK-LABEL: @ConcatWithExplicitDistributedAttr
func.func @ConcatWithExplicitDistributedAttr(%arg0: !InputTensorDistributed, %arg1: !InputTensorDistributed) -> !OutputTensorDistributed {
    %output = VPU.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]}: !InputTensorDistributed, !InputTensorDistributed -> !OutputTensorDistributed
    return %output : !OutputTensorDistributed

    // CHECK:       [[ALLOC_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 8, 16], [1, 64, 8, 16]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC_BUF]] [0, 0, 0, 0] [1, 32, 16, 16] :
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 8, 16], [1, 64, 8, 16]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]

    // CHECK:       [[CLUSTER_COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[SUBVIEW0]] as [[ARG2:%.+]]: memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]

    // CHECK:       VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ARG2]] : memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)


    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC_BUF]] [0, 32, 0, 0] [1, 32, 16, 16] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME           to !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]

    // CHECK:       [[CLUSTER_COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs(%arg1 as [[ARG3:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[SUBVIEW1]] as [[ARG4:%.+]]: memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN,
    // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG3]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ARG4]] : memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)


    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:       inputs([[CLUSTER_COPY0]], [[CLUSTER_COPY1]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
    // CHECK-SAME           !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 8, 16], [1, 32, 8, 16]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 10, 16], [1, 32, 10, 16]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]
    // CHECK-SAME:       outputs([[ALLOC_BUF]] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 64, 8, 16], [1, 64, 8, 16]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 64, 10, 16], [1, 64, 10, 16]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]] }
>

!InputSMTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",  num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]] }
>

!InputSparseTensorDistributed = !VPU.SparseTensor<
    data=!InputTensorDistributed,
    sparsity_map=!InputSMTensorDistributed
>

!OutputTensorDistributed = !VPU.DistributedTensor<
    1x32x32x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 32, 8], [1, 32, 32, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    memory_shapes = [[1, 32, 32, 10], [1, 32, 32, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]] }
>

!OutputSMTensorDistributed = !VPU.DistributedTensor<
    1x32x32x16xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 32, 8], [1, 32, 32, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    memory_shapes = [[1, 32, 32, 10], [1, 32, 32, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]] }
>

!OutputSparseTensorDistributed = !VPU.SparseTensor<
    data=!OutputTensorDistributed,
    sparsity_map=!OutputSMTensorDistributed
>

// CHECK-LABEL: @SparseConcatWithExplicitDistributedAttr
func.func @SparseConcatWithExplicitDistributedAttr(%arg0: !InputTensorDistributed, %arg1: !InputSMTensorDistributed,
                                 %arg2: !InputTensorDistributed, %arg3: !InputSMTensorDistributed) -> !OutputSparseTensorDistributed {
    %st1 = VPU.GroupSparseTensor(%arg0, %arg1) -> !InputSparseTensorDistributed
    %st2 = VPU.GroupSparseTensor(%arg2, %arg3) -> !InputSparseTensorDistributed
    %res = VPU.Concat(%st1, %st2) {static_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]]}: !InputSparseTensorDistributed, !InputSparseTensorDistributed -> !OutputSparseTensorDistributed
    return %res : !OutputSparseTensorDistributed

    // CHECK:       [[ALLOC_IN0_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer(%arg0, %arg1)
    // CHECK-SAME:         -> !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64


    // CHECK:       [[ALLOC_IN1_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer(%arg2, %arg3)
    // CHECK-SAME:         -> !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64


    // CHECK:       [[ALLOC_DATA_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 32, 8], [1, 32, 32, 8]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 32, 10], [1, 32, 32, 10]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>

    // CHECK:       [[ALLOC_SM_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 32, 8], [1, 32, 32, 8]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 32, 10], [1, 32, 32, 10]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>

    // CHECK:       [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]])
    // CHECK-SAME:         -> !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 32, 8], [1, 32, 32, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 32, 10], [1, 32, 32, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64


    // CHECK:       [[SUBVIEW_0_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 0, 0] [1, 32, 16, 16] :
    // CHECK-SAME:            !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME:         to !VPUIP.SparseBuffer<
    // CHECK-SAME:              data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:              sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64


    // CHECK:       [[COPY_0_RES:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[ALLOC_IN0_SPARSE_BUF]]
    // CHECK-SAME:          outputs([[SUBVIEW_0_RES]]
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<
    // CHECK-SAME:               data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:               sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-NEXT:      VPUIP.Copy

    // CHECK:       [[SUBVIEW_1_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 16, 0] [1, 32, 16, 16] :
    // CHECK-SAME:            !VPUIP.SparseBuffer<
    // CHECK-SAME:               data=!VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME:               sparsity_map=!VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME:         to !VPUIP.SparseBuffer<
    // CHECK-SAME:               data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:               sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64


    // CHECK:       [[COPY_1_RES:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[ALLOC_IN1_SPARSE_BUF]]
    // CHECK-SAME:         outputs([[SUBVIEW_1_RES]]
    // CHECK-SAme:         -> !VPUIP.SparseBuffer<
    // CHECK-SAME:               data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:               sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-NEXT:      VPUIP.Copy

    // CHECK:       VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_0_RES]], [[COPY_1_RES]] :
    // CHECK-SAME:          !VPUIP.SparseBuffer<
    // CHECK-SAME:             data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:             sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64

    // CHECK-SAME:          !VPUIP.SparseBuffer<
    // CHECK-SAME:             data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 16, 8], [1, 32, 16, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 16, 10], [1, 32, 16, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:             sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, {order = #NHWC, strides = [16384, 1, 512, 32]}, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64

    // CHECK-SAME:       outputs([[ALLOC_SPARSE_BUF]] :
    // CHECK-SAME:          !VPUIP.SparseBuffer<
    // CHECK-SAME:             data=!VPUIP.DistributedBuffer<1x32x32x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 32, 32, 8], [1, 32, 32, 8]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 8]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 32, 32, 10], [1, 32, 32, 10]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 6]]}>
    // CHECK-SAME:             sparsity_map=!VPUIP.DistributedBuffer<1x32x32x16xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                  {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>, sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x16x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>, sparsity_map=tensor<1x1x1x4096xi1, {order = #NHWC, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @SplitSparseTensor
func.func @SplitSparseTensor(%arg0: tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>, %arg1: tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>) -> (!OutputSparseTensor, !OutputSparseTensor) {
    %st = VPU.GroupSparseTensor(%arg0, %arg1) -> !InputSparseTensor
    %parts:2 = VPU.Split(%st) {num_splits = 2, axis_value = 1} : !InputSparseTensor -> !OutputSparseTensor, !OutputSparseTensor
    return %parts#0, %parts#1 : !OutputSparseTensor, !OutputSparseTensor

    // CHECK:      [[BUF_1_PART:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:      [[MAP_FOR_1_PART:%.+]] = memref.alloc() : memref<1x1x1x4096xi1, #NHWC, @CMX_NN>
    // CHECK:      [[SPARSE_BUF_FOR_1_PART:%.+]] = VPUIP.GroupSparseBuffer([[BUF_1_PART]], [[MAP_FOR_1_PART]]) -> !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>

    // CHECK:      [[BUF_FOR_2_PART:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:      [[MAP_FOR_2_PART:%.+]] = memref.alloc() : memref<1x1x1x4096xi1, #NHWC, @CMX_NN>
    // CHECK:      [[SPARSE_BUF_FOR_2_PART:%.+]] = VPUIP.GroupSparseBuffer([[BUF_FOR_2_PART]], [[MAP_FOR_2_PART]]) -> !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>

    // CHECK:      [[SUBVIEW_1:%.+]] = VPUIP.SubView {{%.+}} [0, 0, 0, 0] [1, 16, 16, 16]
    // CHECK-SAME:    : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:    to !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>, sparsity_map=memref<1x16x16x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>
    // CHECK:      [[RES_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:    inputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>, sparsity_map=memref<1x16x16x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>)
    // CHECK-SAME:    outputs([[SPARSE_BUF_FOR_1_PART]] : !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:    -> !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>

    // CHECK:      [[SUBVIEW_2:%.+]] = VPUIP.SubView {{%.+}} [0, 16, 0, 0] [1, 16, 16, 16]
    // CHECK-SAME:    : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:    to !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>, sparsity_map=memref<1x16x16x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>
    // CHECK:      [[RES_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:    inputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>, sparsity_map=memref<1x16x16x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>)
    // CHECK-SAME:    outputs([[SPARSE_BUF_FOR_2_PART]] : !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:    -> !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>

    // CHECK:      return [[RES_1]], [[RES_2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x32x16x16xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x16x32x16xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x16xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @ReshapeSparseTensor
func.func @ReshapeSparseTensor(%arg0: tensor<1x32x16x16xf16, {order = #NCHW, mem_space = @CMX_NN}>, %arg1: tensor<1x32x16x16xi1, {order = #NCHW, mem_space = @CMX_NN}>) -> !OutputSparseTensor {
    %st = VPU.GroupSparseTensor(%arg0, %arg1) -> !InputSparseTensor
    %res = VPU.Reshape(%st) {shape_value = [1, 16, 32, 16]} : !InputSparseTensor -> !OutputSparseTensor
    return %res : !OutputSparseTensor

    // CHECK:      [[RES:%.+]] = VPUIP.GenericReshape inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x16x32x16xf16, @CMX_NN>, sparsity_map=memref<1x16x32x16xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<32x16x16x1xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<32x16x16x1xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x32x16x16xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @AffineReshapeSparseTensor
func.func @AffineReshapeSparseTensor(%arg0: tensor<32x16x16x1xf16, {order = #NCHW, mem_space = @CMX_NN}>, %arg1: tensor<32x16x16x1xi1, {order = #NCHW, mem_space = @CMX_NN}>) -> !OutputSparseTensor {
    %st = VPU.GroupSparseTensor(%arg0, %arg1) -> !InputSparseTensor
    %res = VPU.AffineReshape(%st) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 32, 16, 16]} : !InputSparseTensor -> !OutputSparseTensor
    return %res : !OutputSparseTensor

    // CHECK:      [[RES:%.+]] = VPUIP.GenericReshape inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<32x16x16x1xf16, @CMX_NN>, sparsity_map=memref<32x16x16x1xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x3x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x3x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x16x16x12xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x16x12xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @ShapeCastSparseTensor
func.func @ShapeCastSparseTensor(%arg0: tensor<1x3x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, %arg1: tensor<1x3x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>) -> !OutputSparseTensor {
    %st = VPU.GroupSparseTensor(%arg0, %arg1) -> !InputSparseTensor
    %res = VPU.ShapeCast {shape = [1, 16, 16, 12]} inputs(%st : !InputSparseTensor) -> !OutputSparseTensor
    return %res : !OutputSparseTensor

    // CHECK:      [[RES:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 16, 12]} inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<1x3x32x32xf16, @CMX_NN>, sparsity_map=memref<1x3x32x32xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x16x16x12xf16, @CMX_NN>, sparsity_map=memref<1x16x16x12xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x16x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x16x32x32xf16, {order = #NHWC, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x32xi1, {order = #NHWC, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @LayoutCastSparseTensor
func.func @LayoutCastSparseTensor(%arg0: tensor<1x16x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, %arg1: tensor<1x16x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>) -> !OutputSparseTensor {
    %st = VPU.GroupSparseTensor(%arg0, %arg1) -> !InputSparseTensor
    %res = VPU.LayoutCast(%st) {dst_order = #NHWC} : !InputSparseTensor -> !OutputSparseTensor
    return %res : !OutputSparseTensor

    // CHECK:      [[RES:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<1x16x32x32xf16, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x16x32x32xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, #NHWC, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<16x32x32xf16, {order = #CHW, mem_space = @CMX_NN}>, sparsity_map=tensor<16x32x32xi1, {order = #CHW, mem_space = @CMX_NN}>
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x16x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @UnsqueezeSparseTensor
func.func @UnsqueezeSparseTensor(%arg0: !InputSparseTensor) -> !OutputSparseTensor {
    %res = VPU.Unsqueeze(%arg0) { axes_value = [0] } : !InputSparseTensor -> !OutputSparseTensor
    return %res : !OutputSparseTensor

    // CHECK:      [[RES:%.+]] = VPUIP.GenericReshape inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<16x32x32xf16, @CMX_NN>, sparsity_map=memref<16x32x32xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x16x32x32xf16, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>

!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x16x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<16x32x32xf16, {order = #CHW, mem_space = @CMX_NN}>, sparsity_map=tensor<16x32x32xi1, {order = #CHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @SqueezeSparseTensor
func.func @SqueezeSparseTensor(%arg0: tensor<1x16x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, %arg1: tensor<1x16x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>) -> !OutputSparseTensor {
    %st = VPU.GroupSparseTensor(%arg0, %arg1) -> !InputSparseTensor
    %res = VPU.Squeeze(%st) { axes_value = [0] } : !InputSparseTensor -> !OutputSparseTensor
    return %res : !OutputSparseTensor

    // CHECK:      [[RES:%.+]] = VPUIP.GenericReshape inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<1x16x32x32xf16, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<16x32x32xf16, @CMX_NN>, sparsity_map=memref<16x32x32xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

!DataTensor = tensor<1x16x32x32xf16>
!SMTensor = tensor<1x16x32x32xi1>
!SETensor = tensor<1x1x32x32xi32>
!SparseTensor = !VPU.SparseTensor<data=tensor<1x16x32x32xf16>, sparsity_map=tensor<1x16x32x32xi1>, storage_element_table=tensor<1x1x32x32xi32>>

// CHECK-LABEL: @UngroupSparseTensor
// CHECK-SAME:  ([[ARG0:%.+]]: !VPUIP.SparseBuffer<data=memref<1x16x32x32xf16>, sparsity_map=memref<1x16x32x32xi1>, storage_element_table=memref<1x1x32x32xi32>>)
func.func @UngroupSparseTensor(%arg0: !SparseTensor) -> (!DataTensor, !SMTensor, !SETensor) {
    %data, %sm, %se = VPU.UngroupSparseTensor(%arg0) {resultSegmentSizes = array<i32: 1, 1, 1>} -> !DataTensor, !SMTensor, !SETensor
    return %data, %sm, %se : !DataTensor, !SMTensor, !SETensor

    // CHECK:  [[DATA:%.+]], [[SM:%.+]], [[SE:%.+]] = VPUIP.UngroupSparseBuffer([[ARG0]]) {resultSegmentSizes = array<i32: 1, 1, 1>} -> memref<1x16x32x32xf16>, memref<1x16x32x32xi1>, memref<1x1x32x32xi32>
    // CHECK:  return [[DATA]], [[SM]], [[SE]]
}
