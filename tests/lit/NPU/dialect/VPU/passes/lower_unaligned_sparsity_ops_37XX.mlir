//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --lower-sparsity-ops="fake-sparsify=false" %s | FileCheck %s
// REQUIRES: arch-VPUX37XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!inputType = tensor<1x4x120x110x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>
!smType = tensor<1x4x120x110xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!inputType, sparsity_map=!smType>
!weightsType = tensor<64x1x1x16xf16, {order = #NHWC}>
!outputType = tensor<1x64x120x110x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>

func.func @LowerSparsifyOpUniformQuantUnalignedShape(%arg0: !inputType, %wt: tensor<64x1x1x4xsi32>, %weights: !weightsType) -> !outputType {
    %0 = VPU.Sparsify(%arg0) : !inputType -> !sparseType
    %1 = VPU.NCE.CompressConvolution(%0, %weights, %wt) {
            cm_sp_pattern = 15 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [64, 4, 1, 1],
            strides = [1, 1]
        } -> !outputType

    return %1 : !outputType

    // CHECK:       [[VAL0:%.+]] = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK-SAME:    : tensor<1x4x120x110x!qElemType, {order = #NHWC}> -> tensor<1x16x120x110x!qElemType, {order = #NHWC}>
    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[SPARSE_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<16x16x1x1x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                           #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:    : tensor<16x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[SPARSE_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) {
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:        strides = [1, 1]
    // CHECK-SAME:    } -> !VPU.SparseTensor<data=tensor<1x16x120x110x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x120x110xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Slice [[VAL1]] [0, 0, 0, 0] [1, 4, 120, 110]
    // CHECK-SAME:    : !VPU.SparseTensor<data=tensor<1x16x120x110x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x120x110xi1, {order = #NHWC}>> to
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x4x120x110x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x4x120x110xi1, {order = #NHWC}>>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.CompressConvolution([[VAL2]], %arg2, %arg1)
    // CHECK:       return [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!inputType = tensor<1x4x31x31xf16, {order = #NHWC}>
!smType = tensor<1x4x31x31xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!inputType, sparsity_map=!smType>
!weightsType = tensor<64x1x1x16xf16, {order = #NHWC}>
!outputType = tensor<1x64x31x31xf16, {order = #NHWC}>

func.func @LowerSparsifyOpFloatUnalignedShape(%arg0: !inputType, %wt: tensor<64x1x1x4xsi32>, %weights: !weightsType) -> !outputType {
    %0 = VPU.Sparsify(%arg0) : !inputType -> !sparseType
    %1 = VPU.NCE.CompressConvolution(%0, %weights, %wt) {
            cm_sp_pattern = 15 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [64, 4, 1, 1],
            strides = [1, 1]
        } -> !outputType

    return %1 : !outputType

    // CHECK:       [[VAL0:%.+]] = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK-SAME:    : tensor<1x4x31x31xf16, {order = #NHWC}> -> tensor<1x16x31x31xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[SPARSE_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                           #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:    : tensor<16x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[SPARSE_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) {
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:        strides = [1, 1]
    // CHECK-SAME:    } -> !VPU.SparseTensor<data=tensor<1x16x31x31xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x31x31xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Slice [[VAL1]] [0, 0, 0, 0] [1, 4, 31, 31]
    // CHECK-SAME:    : !VPU.SparseTensor<data=tensor<1x16x31x31xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x31x31xi1, {order = #NHWC}>> to
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x4x31x31xf16, {order = #NHWC}>, sparsity_map=tensor<1x4x31x31xi1, {order = #NHWC}>>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.CompressConvolution([[VAL2]], %arg2, %arg1)
    // CHECK:       return [[VAL3]]
}
