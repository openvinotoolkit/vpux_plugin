//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --lower-sparsity-ops="fake-sparsify=false" %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16xf16, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

func.func @LowerSparsifyOpF16(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.Sparsify(%arg0) : !defaultType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[SPARSE_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                           #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[SPARSE_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

func.func @LowerSparsifyOpUniformQuant(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.Sparsify(%arg0) : !defaultType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[SPARSE_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<16x16x1x1x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                           #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<1> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:    : tensor<16x1x1x4xsi32>
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[SPARSE_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) {
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:        strides = [1, 1]
    // CHECK-SAME:    } -> !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16:1, {1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}>
!defaultType = tensor<1x16x16x16x!qElemType, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK:  !qElemType = !quant.uniform<u8:f16:1, {1.000000e+00,2.000000e+00,3.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

func.func @LowerSparsifyOpPerAxisQuant(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.Sparsify(%arg0) : !defaultType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[VAL0:%.+]] = VPU.QuantizeCast(%arg0) {dstElemType = !qElemType1} : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
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
    // CHECK-SAME:    } -> !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.QuantizeCast([[VAL1]]) {dstElemType = !qElemType}
    // CHECK-SAME:     : !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]], %arg2, %arg1)
    // CHECK:       return [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16xf16, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

func.func @LowerDesparsifyOpF16(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !sparseType
    %1 = VPU.Desparsify(%0) : !sparseType -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %arg2, %arg1)
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL0]])
    // CHECK-SAME:                   op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:                   clamp_low = -2147483648 : i64
    // CHECK-SAME:                   clamp_high = 2147483647 : i64
    // CHECK-SAME:                   quant_scale = [5.000000e-01]
    // CHECK-NOT:                    !VPU.SparseTensor

    // CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

func.func @LowerDesparsifyOpQuantUniform(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !sparseType
    %1 = VPU.Desparsify(%0) : !sparseType -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %arg2, %arg1)
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL0]])
    // CHECK-SAME:                   op_type = #VPU.eltwise_type<ADD>
    // CHECK:                        quant_mult = [16384]
    // CHECK-SAME:                   quant_shift = [29]
    // CHECK-SAME:                   quant_post_shift = 0
    // CHECK-NOT:                    !VPU.SparseTensor

    // CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16:1, {1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}>
!defaultType = tensor<1x16x16x16x!qElemType, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK:  !qElemType = !quant.uniform<u8:f16:1, {1.000000e+00,2.000000e+00,3.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

func.func @LowerDesparsifyOpPerAxisQuant(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !sparseType
    %1 = VPU.Desparsify(%0) : !sparseType -> !defaultType

    return %1 : !defaultType

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %arg2, %arg1)
    // CHECK:       [[VAL1:%.+]] = VPU.QuantizeCast(%0) {dstElemType = !qElemType1}
    // CHECK-SAME:       : !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Eltwise([[VAL1]], [[VAL1]])
    // CHECK-SAME:                   op_type = #VPU.eltwise_type<ADD>
    // CHECK:                        quant_mult = [16384]
    // CHECK-SAME:                   quant_shift = [29]
    // CHECK-SAME:                   quant_post_shift = 0
    // CHECK-NOT:                    !VPU.SparseTensor
    // CHECK:       [[VAL3:%.+]] = VPU.QuantizeCast([[VAL2]]) {dstElemType = !qElemType} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // CHECK:       return [[VAL3]]
}
