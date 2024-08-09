//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: FuseConsecutiveSubViews
func.func @FuseConsecutiveSubViews() -> tensor<320x2816x1x1xf16> {
    %cst = const.Declare tensor<320x2816x1x1xf16> = dense<0.0> : tensor<4096x11008x1x1xf16>,
        [#const.SubView<[0, 8192, 0, 0], [4096, 2816, 1, 1]>, #const.SubView<[0, 0, 0, 0], [320, 2816, 1, 1]>]
    return %cst : tensor<320x2816x1x1xf16>

    // CHECK: [#const.SubView<[0, 8192, 0, 0], [320, 2816, 1, 1]>]
}

// -----

// CHECK-LABEL: FuseConsecutiveAdds
func.func @FuseConsecutiveAdds() -> tensor<16x32x1x1xf16> {
    %cst = const.Declare tensor<16x32x1x1xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Add<1.000000e+00 : f64>, #const.Add<2.000000e+00 : f64>]
    return %cst : tensor<16x32x1x1xf16>

    // CHECK: [#const.Add<3.000000e+00 : f64>]
}

// -----

// CHECK-LABEL: FuseConsecutiveRescales
func.func @FuseConsecutiveRescales() -> tensor<16x32x1x1xf16> {
    %cst = const.Declare tensor<16x32x1x1xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Rescale<2.000000e+00 : f64>, #const.Rescale<3.000000e+00 : f64>]
    return %cst : tensor<16x32x1x1xf16>

    // CHECK: [#const.Rescale<6.000000e+00 : f64>]
}

// -----

// CHECK-LABEL: FuseConsecutiveReshapes
func.func @FuseConsecutiveReshapes() -> tensor<16x8x1x4xf16> {
    %cst = const.Declare tensor<16x8x1x4xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reshape<[16, 8, 2, 2]>, #const.Reshape<[16, 8, 1, 4]>]
    return %cst : tensor<16x8x1x4xf16>

    // CHECK: [#const.Reshape<[16, 8, 1, 4]>]
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseConsecutiveReorders
func.func @FuseConsecutiveReorders() -> tensor<16x32x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1> : tensor<16x32x1x1xui8>, [#const.Reorder<#NHCW>, #const.Reorder<#NHWC>]
    return %cst : tensor<16x32x1x1xf16, {order = #NHWC}>

    // CHECK: [#const.Reorder<#NHWC>]
}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SwapReorderAndSubView
func.func @SwapReorderAndSubView() -> tensor<2x1x1x1xf16, { order = #NHWC }> {
    %cst = const.Declare tensor<2x1x1x1xf16, { order = #NHWC }> = dense<0.0> : tensor<4x3x2x1xf16>,
        [#const.Reorder<#NHWC>, #const.SubView<[2, 2, 1, 0], [2, 1, 1, 1]>]
    return %cst : tensor<2x1x1x1xf16, { order = #NHWC }>

    // CHECK: [#const.SubView<[2, 2, 1, 0], [2, 1, 1, 1]>, #const.Reorder<#NHWC>]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: SwapTransposeAndSubView
func.func @SwapTransposeAndSubView() -> tensor<4x3x1x2xf16, { order = #HCNW }> {
    %cst = const.Declare tensor<4x3x1x2xf16, { order = #HCNW }> = dense<0.0> : tensor<8x6x4x2xf16, { order = #HCNW }>,
        [#const.Transpose<#NHWC>, #const.SubView<[4, 1, 1, 4], [4, 3, 1, 2]>]
    return %cst : tensor<4x3x1x2xf16, { order = #HCNW }>

    // CHECK: [#const.SubView<[4, 4, 1, 1], [4, 2, 3, 1]>, #const.Transpose<#NHWC>]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: SwapMemPermAndSubViewMultipleInstances
func.func @SwapMemPermAndSubViewMultipleInstances() -> tensor<10x100x32x10xf16, { order = #NHWC }> {
    %cst = const.Declare tensor<10x100x32x10xf16, { order = #NHWC }> = dense<0.0> : tensor<4096x11008x1x1xf16>,
        [
            #const.Transpose<#HCNW>,
            #const.SubView<[0, 8192, 0, 0], [1, 2816, 4096, 1]>,
            #const.SubView<[0, 0, 0, 0], [1, 2816, 320, 1]>,
            #const.Reorder<#NHWC>,
            #const.SubView<[0, 1000, 0, 0], [1, 1816, 320, 1]>,
            #const.SubView<[0, 0, 0, 0], [1, 1000, 320, 1]>,
            #const.Reshape<[10, 100, 32, 10]>
        ]
    return %cst : tensor<10x100x32x10xf16, { order = #NHWC }>

    // CHECK: [#const.SubView<[0, 9192, 0, 0], [320, 1000, 1, 1]>, #const.Transpose<#map>, #const.Reorder<#NHWC>, #const.Reshape<[10, 100, 32, 10]>]
}

// -----

// CHECK-LABEL: SwapAddAndSubView
func.func @SwapAddAndSubView() -> tensor<8x16x1x1xf16> {
    %cst = const.Declare tensor<8x16x1x1xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Add<1.000000e+00 : f64>, #const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>]
    return %cst : tensor<8x16x1x1xf16>

    // CHECK: [#const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>, #const.Add<1.000000e+00 : f64>]
}

// -----

// CHECK-LABEL: SwapRescaleAndSubView
func.func @SwapRescaleAndSubView() -> tensor<8x16x1x1xf16> {
    %cst = const.Declare tensor<8x16x1x1xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Rescale<2.000000e+00 : f64>, #const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>]
    return %cst : tensor<8x16x1x1xf16>

    // CHECK: [#const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>, #const.Rescale<2.000000e+00 : f64>]
}

// -----

// CHECK-LABEL: SwapConvertElemTypeAndSubView
func.func @SwapConvertElemTypeAndSubView() -> tensor<8x16x1x1xf16> {
    %cst = const.Declare tensor<8x16x1x1xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf32>, [#const.ConvertElemType<f16>, #const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>]
    return %cst : tensor<8x16x1x1xf16>

    // CHECK: [#const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>, #const.ConvertElemType<f16>]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:0, {1.000000e+00:128,2.000000e+00:128}>
// CHECK: !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u8:f16:0, {2.000000e+00:128}>

// CHECK-LABEL: SwapQuantizeTransformationsAndSubView
func.func @SwapQuantizeTransformationsAndSubView() -> (tensor<8x16x1x1xf16>, tensor<1x16x1x1xf16>) {
    %cst = const.Declare tensor<8x16x1x1xf16> = dense<1> : tensor<16x32x1x1xui8>, [#const.QuantCast<!qElemType>, #const.Dequantize, #const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>]
    %cst_peraxis = const.Declare tensor<1x16x1x1xf16> = dense<1> : tensor<2x32x1x1xui8>, [#const.QuantCast<!qElemType1>, #const.Dequantize, #const.SubView<[1, 16, 0, 0], [1, 16, 1, 1]>]
    return %cst, %cst_peraxis : tensor<8x16x1x1xf16>, tensor<1x16x1x1xf16>

    // CHECK: [#const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>, #const.QuantCast<!qElemType>, #const.Dequantize]
    // CHECK: [#const.SubView<[1, 16, 0, 0], [1, 16, 1, 1]>, #const.QuantCast<!qElemType1>, #const.Dequantize]
}

// -----

// SwapReshapeAndSubView
func.func @SwapReshapeAndSubView() -> (tensor<8x8x1x1xf16>, tensor<1x16x3x4xf16>) {
    %cst1 = const.Declare tensor<8x8x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x8xf16>, [#const.Reshape<[16, 8, 1, 1]>, #const.SubView<[0, 0, 0, 0], [8, 8, 1, 1]>]
    %cst2 = const.Declare tensor<1x16x3x4xf16> = dense<1.000000e+00> : tensor<1x16x48x1xf16>, [#const.Reshape<[1, 16, 12, 4]>, #const.SubView<[0, 0, 9, 0], [1, 16, 3, 4]>]
    return %cst1, %cst2: tensor<8x8x1x1xf16>, tensor<1x16x3x4xf16>

    // CHECK: tensor<1x16x1x8xf16>, [#const.SubView<[0, 0, 0, 0], [1, 8, 1, 8]>, #const.Reshape<[8, 8, 1, 1]>]
    // CHECK: tensor<1x16x48x1xf16>, [#const.Reshape<[1, 16, 12, 4]>, #const.SubView<[0, 0, 9, 0], [1, 16, 3, 4]>]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e+00:128,2.000000e+00:128,3.000000e+00:128,4.000000e+00:128}>
!qElemType1 = !quant.uniform<u8:f16:0, {1.000000e+00:128,2.000000e+00:128,3.000000e+00:128,4.000000e+00:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {1.000000e+00:128,2.000000e+00:128,3.000000e+00:128}>
// CHECK: !qElemType = !quant.uniform<u8:f16:0, {1.000000e+00:128,2.000000e+00:128,3.000000e+00:128}>
// CHECK: !qElemType1 = !quant.uniform<u8:f16:1, {1.000000e+00:128,2.000000e+00:128,3.000000e+00:128}>

// CHECK-LABEL: SwapChangeShapeAndElemTypeAndSubView
func.func @SwapChangeShapeAndElemTypeAndSubView() -> tensor<3x2x1x1x!qElemType2> {
    %cst = const.Declare tensor<3x2x1x1x!qElemType2> = dense<1> : tensor<1x4x1x2xui8>, [#const.QuantCast<!qElemType>, #const.ChangeShapeAndElemType<[4, 2, 1, 1], !qElemType1>, #const.SubView<[0, 0, 0, 0], [3, 2, 1, 1]>]
    return %cst: tensor<3x2x1x1x!qElemType2>

    // CHECK: [#const.SubView<[0, 0, 0, 0], [1, 3, 1, 2]>, #const.QuantCast<!qElemType1>, #const.ChangeShapeAndElemType<[3, 2, 1, 1], !qElemType>]
}

// -----

// CHECK-LABEL: SwapRelocateWeightsTableAndSubView
func.func @SwapRelocateWeightsTableAndSubView() -> (tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>) {
    %cst = const.Declare tensor<2x1x1x4xsi32> = dense<1> : tensor<4x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[100], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=64 : i64>,
         #const.SubView<[0, 0, 0, 0], [2, 1, 1, 4]>]
    %cst_offset = const.Declare tensor<2x1x1x4xsi32> = dense<1> : tensor<4x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[100], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=64 : i64>,
         #const.SubView<[1, 0, 0, 0], [2, 1, 1, 4]>]
    %cst_sparse = const.Declare tensor<2x1x1x4xsi32> = dense<1> : tensor<4x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[100], sparsityPtr=200 : i64, offsets=[0], weightsTableSize=64 : i64, weightsElemBitSize=16 : i64,
                                     weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<[1, 2, 3, 4]> : tensor<4xi64>, alignment = 16 : i64>>,
         #const.SubView<[2, 0, 0, 0], [2, 1, 1, 4]>]
    %cst_sparse_already_offset = const.Declare tensor<2x1x1x4xsi32> = dense<1> : tensor<4x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[100], sparsityPtr=200 : i64, offsets=[0], weightsTableSize=64 : i64, weightsElemBitSize=16 : i64,
                                     weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<[1, 2, 3, 4]> : tensor<4xi64>, alignment = 16 : i64>,
                                     channelOffset=2 : i64>,
         #const.SubView<[2, 0, 0, 0], [2, 1, 1, 4]>]
    %cst_segmented = const.Declare tensor<2x1x1x4xsi32> = dense<1> : tensor<8x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[100, 200, 300, 400], sparsityPtr=16777215 : i64, offsets=[0, 2, 4, 6], weightsTableSize=128 : i64, weightsElemBitSize=16 : i64>,
         #const.SubView<[4, 0, 0, 0], [2, 1, 1, 4]>]
    %cst_segmented_sparse = const.Declare tensor<2x1x1x4xsi32> = dense<1> : tensor<8x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[100, 200, 300, 400], sparsityPtr=200 : i64, offsets=[0, 2, 4, 6], weightsTableSize=128 : i64, weightsElemBitSize=16 : i64,
                                     weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi64>, alignment = 16 : i64>>,
         #const.SubView<[4, 0, 0, 0], [2, 1, 1, 4]>]

    // The subview does not cover the values of a single cluster
    %cst_segmented_noswap = const.Declare tensor<2x1x1x4xsi32> = dense<1> : tensor<8x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[100, 200, 300, 400], sparsityPtr=16777215 : i64, offsets=[0, 2, 4, 6], weightsTableSize=128 : i64, weightsElemBitSize=16 : i64>,
         #const.SubView<[3, 0, 0, 0], [2, 1, 1, 4]>]

    return %cst, %cst_offset, %cst_sparse, %cst_sparse_already_offset, %cst_segmented, %cst_segmented_sparse, %cst_segmented_noswap
        : tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>, tensor<2x1x1x4xsi32>

    // CHECK:      [[CST:%.+]] = const.Declare
    // CHECK-SAME:     [#const.SubView<[0, 0, 0, 0], [2, 1, 1, 4]>,
    // CHECK-SAME:      #const.RelocateWeightsTable<weightsPtr=[100], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=32 : i64, channelOffset=0 : i64>]
    // CHECK:      [[CST_OFFSET:%.+]] = const.Declare
    // CHECK-SAME:     [#const.SubView<[1, 0, 0, 0], [2, 1, 1, 4]>,
    // CHECK-SAME:      #const.RelocateWeightsTable<weightsPtr=[100], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=32 : i64, channelOffset=1 : i64>]
    // CHECK:      [[CST_SPARSE:%.+]] = const.Declare
    // CHECK:          [#const.SubView<[2, 0, 0, 0], [2, 1, 1, 4]>,
    // CHECK-SAME:      #const.RelocateWeightsTable<weightsPtr=[100], sparsityPtr=200 : i64, offsets=[0], weightsTableSize=32 : i64, weightsElemBitSize=16 : i64,
    // CHECK-SAME:      weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<[3, 4]> : tensor<2xi64>, alignment = 16 : i64>, channelOffset=2 : i64>]
    // CHECK:      [[CST_SPARSE_ALREADY_OFFSET:%.+]] = const.Declare
    // CHECK:          [#const.SubView<[2, 0, 0, 0], [2, 1, 1, 4]>,
    // CHECK-SAME:      #const.RelocateWeightsTable<weightsPtr=[100], sparsityPtr=200 : i64, offsets=[0], weightsTableSize=32 : i64, weightsElemBitSize=16 : i64,
    // CHECK-SAME:      weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<[3, 4]> : tensor<2xi64>, alignment = 16 : i64>, channelOffset=4 : i64>]
    // CHECK:      [[CST_SEGMENTED:%.+]] = const.Declare
    // CHECK:          [#const.SubView<[4, 0, 0, 0], [2, 1, 1, 4]>,
    // CHECK-SAME:      #const.RelocateWeightsTable<weightsPtr=[300], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=32 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64>]
    // CHECK:      [[CST_SEGMENTED_SPARSE:%.+]] = const.Declare
    // CHECK:          [#const.SubView<[4, 0, 0, 0], [2, 1, 1, 4]>,
    // CHECK-SAME:      #const.RelocateWeightsTable<weightsPtr=[300], sparsityPtr=200 : i64, offsets=[0], weightsTableSize=32 : i64, weightsElemBitSize=16 : i64,
    // CHECK-SAME:      weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<[5, 6]> : tensor<2xi64>, alignment = 16 : i64>, channelOffset=0 : i64>]
    // CHECK:      [[CST_SEGMENTED_NOSWAP:%.+]] = const.Declare
    // CHECK:          [#const.RelocateWeightsTable<weightsPtr=[100, 200, 300, 400], sparsityPtr=16777215 : i64, offsets=[0, 2, 4, 6], weightsTableSize=128 : i64, weightsElemBitSize=16 : i64>,
    // CHECK-SAME:      #const.SubView<[3, 0, 0, 0], [2, 1, 1, 4]>]
    // CHECK:      return [[CST]], [[CST_OFFSET]], [[CST_SPARSE]], [[CST_SPARSE_ALREADY_OFFSET]], [[CST_SEGMENTED]], [[CST_SEGMENTED_SPARSE]], [[CST_SEGMENTED_NOSWAP]]
}

// -----

// CHECK-LABEL: SwapMultipleTransformationsAndSubView
func.func @SwapMultipleTransformationsAndSubView() -> tensor<8x16x1x1xf16> {
    %cst = const.Declare tensor<8x16x1x1xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [
        #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>, #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>,
        #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>, #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>,
        #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>, #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>,
        #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>, #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>,
        #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>, #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>,
        #const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>]
    return %cst : tensor<8x16x1x1xf16>

    // CHECK:          [#const.SubView<[0, 16, 0, 0], [8, 16, 1, 1]>,
    // CHECK-COUNT-10:  #const.Add<1.000000e+00 : f64>, #const.Rescale<2.000000e+00 : f64>
    // CHECK-NOT:       #const
    // CHECK-SAME:     ]
}

// -----

// CHECK-LABEL: SwapAddAndReshape
func.func @SwapAddAndReshape() -> tensor<16x8x2x2xf16> {
    %cst = const.Declare tensor<16x8x2x2xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Add<1.000000e+00 : f64>, #const.Reshape<[16, 8, 2, 2]>]
    return %cst : tensor<16x8x2x2xf16>

    // CHECK: [#const.Reshape<[16, 8, 2, 2]>, #const.Add<1.000000e+00 : f64>]
}

// -----

// CHECK-LABEL: SwapRescaleAndReshape
func.func @SwapRescaleAndReshape() -> tensor<16x8x2x2xf16> {
    %cst = const.Declare tensor<16x8x2x2xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Rescale<2.000000e+00 : f64>, #const.Reshape<[16, 8, 2, 2]>]
    return %cst : tensor<16x8x2x2xf16>

    // CHECK: [#const.Reshape<[16, 8, 2, 2]>, #const.Rescale<2.000000e+00 : f64>]
}

// -----

// CHECK-LABEL: SwapConvertElemTypeAndReshape
func.func @SwapConvertElemTypeAndReshape() -> tensor<16x8x2x2xf16> {
    %cst = const.Declare tensor<16x8x2x2xf16> = dense<1.000000e+00> : tensor<16x32x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[16, 8, 2, 2]>]
    return %cst : tensor<16x8x2x2xf16>

    // CHECK: [#const.Reshape<[16, 8, 2, 2]>, #const.ConvertElemType<f16>]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:1, {1.000000e+00:128,2.000000e+00:128,3.000000e+00:128}>
!qElemType2 = !quant.uniform<u8:f16:2, {1.000000e+00:128,2.000000e+00:128,3.000000e+00:128}>
// CHECK: !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u8:f16:1, {1.000000e+00:128,2.000000e+00:128,3.000000e+00:128}>
// CHECK: !qElemType2 = !quant.uniform<u8:f16:2, {1.000000e+00:128,2.000000e+00:128,3.000000e+00:128}>

// CHECK-LABEL: SwapQuantizeTransformationsAndReshape
func.func @SwapQuantizeTransformationsAndReshape() -> (tensor<16x8x2x2xf16>, tensor<1x2x3x1xf16>, tensor<2x1x3x1xf16>, tensor<3x2x1x1xf16>) {
    %cst_pertensor = const.Declare tensor<16x8x2x2xf16> = dense<1> : tensor<16x32x1x1xui8>, [#const.QuantCast<!qElemType>, #const.Dequantize, #const.Reshape<[16, 8, 2, 2]>]
    // CHECK: tensor<16x32x1x1xui8>, [#const.QuantCast<!qElemType>, #const.Reshape<[16, 8, 2, 2]>, #const.Dequantize]

    %cst_peraxis1 = const.Declare tensor<1x2x3x1xf16> = dense<1> : tensor<2x3x1x1xui8>, [#const.QuantCast<!qElemType1>, #const.Dequantize, #const.Reshape<[1, 2, 3, 1]>]
    %cst_peraxis2 = const.Declare tensor<2x1x3x1xf16> = dense<1> : tensor<1x2x3x1xui8>, [#const.QuantCast<!qElemType2>, #const.Dequantize, #const.Reshape<[2, 1, 3, 1]>]
    %cst_peraxis3 = const.Declare tensor<3x2x1x1xf16> = dense<1> : tensor<1x3x1x2xui8>, [#const.QuantCast<!qElemType1>, #const.Dequantize, #const.Reshape<[3, 2, 1, 1]>]
    return %cst_pertensor, %cst_peraxis1, %cst_peraxis2, %cst_peraxis3 : tensor<16x8x2x2xf16>, tensor<1x2x3x1xf16>, tensor<2x1x3x1xf16>, tensor<3x2x1x1xf16>

    // CHECK: tensor<2x3x1x1xui8>, [#const.QuantCast<!qElemType1>, #const.ChangeShapeAndElemType<[1, 2, 3, 1], !qElemType2>, #const.Dequantize]
    // CHECK: tensor<1x2x3x1xui8>, [#const.QuantCast<!qElemType2>, #const.ChangeShapeAndElemType<[2, 1, 3, 1], !qElemType2>, #const.Dequantize]
    // CHECK: tensor<1x3x1x2xui8>, [#const.QuantCast<!qElemType1>, #const.ChangeShapeAndElemType<[3, 2, 1, 1], !qElemType3>, #const.Dequantize]
}

// -----

// CHECK-LABEL: DoNotOptimizeWithoutTransformations
func.func @DoNotOptimizeWithoutTransformations() -> tensor<320x2816x1x1xf16> {
    %cst = const.Declare tensor<320x2816x1x1xf16> = dense<0.0> : tensor<320x2816x1x1xf16>
    return %cst : tensor<320x2816x1x1xf16>

    // CHECK: const.Declare tensor<320x2816x1x1xf16> = dense<0.000000e+00> : tensor<320x2816x1x1xf16>
}

// -----

// CHECK-LABEL: @EraseTiledInfo
#C = affine_map<(d0) -> (d0)>

func.func @EraseTiledInfo() -> memref<8xf32> {
    %0 = const.Declare memref<8xf32, {order = #C, strides = [1]}> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    %1 = IERT.SubView %0 [0] [8] :
        memref<8xf32, {order = #C, strides = [1]}> to
        memref<8xf32>
    return %1 : memref<8xf32>
    // CHECK: [[CST:%.+]] = const.Declare memref<8xf32> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    // CHECK: return %cst : memref<8xf32>
}

// -----

// CHECK-LABEL: @EraseTiledInfoCopy
#C = affine_map<(d0) -> (d0)>

func.func @EraseTiledInfoCopy(%arg0: memref<8xf32, {order = #C}>) -> memref<8xf32, {order = #C}> {
    %0 = const.Declare memref<8xf32, {order = #C, strides = [1]}> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    %1 = IERT.Copy
        inputs(%0 : memref<8xf32, {order = #C, strides = [1]}>)
        outputs(%arg0: memref<8xf32, {order = #C}>)
        -> memref<8xf32, {order = #C}>
    return %1 : memref<8xf32, {order = #C}>
    // CHECK: [[CST:%.+]] = const.Declare memref<8xf32> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    // CHECK: [[VAR1:%.+]] = IERT.Copy inputs([[CST]] : memref<8xf32>) outputs(%arg0 : memref<8xf32, {order = #C}>) -> memref<8xf32, {order = #C}>
    // CHECK: return [[VAR1]] : memref<8xf32, {order = #C}>
}
