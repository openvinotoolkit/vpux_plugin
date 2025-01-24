//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --optimize-sparsify-desparsify-pairs="act-sparsity-profile=S0" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeMultipleConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.Sparsify(%1) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %3 = VPU.Desparsify(%2) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %4 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %5 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %6 = VPU.NCE.Eltwise(%4, %5) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %7 = VPU.Sparsify(%6) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %8 = VPU.Desparsify(%7) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %9 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %10 = VPU.NCE.Convolution(%9, %weights, %wt) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %11 = VPU.Sparsify(%10) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %12 = VPU.Desparsify(%11) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %8, %12 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Eltwise([[VAL2]], [[VAL2]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.Sparsify([[VAL3]])
    // CHECK:       [[VAL5:%.+]] = VPU.Desparsify([[VAL4]]

    // CHECK:       [[VAL6:%.+]] = VPU.NCE.Convolution([[VAL2]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL7:%.+]] = VPU.Sparsify([[VAL6]])
    // CHECK:       [[VAL8:%.+]] = VPU.Desparsify([[VAL7]]

    // CHECK:       return [[VAL5]], [[VAL8]]
}

// -----

// Sparsify -> NCEConv -> Sparsify -> Desparsify -> Sparsify -> NCEEltwise -> Sparsify -> Desparsify -> [result 0]
//                                               |           |>
//                                               |> Sparsify |
//                                               |
//                                               |> MaxPool -> [output 1]
// could be simplified to:
// Sparsify -> NCEConv -> Sparsify -> Desparsify -> MaxPool -> [output 1]
//                                 |
//                                 |> NCEEltwise -> Sparsify -> Desparsify -> [result 0]
// However, this optimization is avoided in order to avoid standalone Desparsify operations as much as possible.
// NCEConv's output pair of Sparsify->Desparsify will be later simplified, similar to NCEEltwise's input Sparsify ops.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DoNotOptimizeMultipleMixedConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.Sparsify(%1) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %3 = VPU.Desparsify(%2) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %4 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %5 = VPU.Sparsify(%3) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %6 = VPU.NCE.Eltwise(%4, %5) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %7 = VPU.Sparsify(%6) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %8 = VPU.Desparsify(%7) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %9 = VPU.MaxPool(%3) {
            kernel_size = [3, 3],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %8, %9 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
    // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]

    // CHECK:       [[VAL4:%.+]] = VPU.Sparsify([[VAL3]])
    // CHECK:       [[VAL5:%.+]] = VPU.Sparsify([[VAL3]])
    // CHECK:       [[VAL6:%.+]] = VPU.NCE.Eltwise([[VAL4]], [[VAL5]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL7:%.+]] = VPU.Sparsify([[VAL6]])
    // CHECK:       [[VAL8:%.+]] = VPU.Desparsify([[VAL7]]

    // CHECK:       [[VAL9:%.+]] = VPU.MaxPool([[VAL3]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL8]], [[VAL9]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!PreConcatType = tensor<1x8x16x16xf16, {order = #NHWC}>
!PostConcatType = tensor<1x16x16x16xf16, {order = #NHWC}>
!DefaultType = tensor<1x16x16x16xf16, {order = #NHWC}>

func.func @OptimizeConcat(%arg0: !PreConcatType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (!DefaultType, !DefaultType) {
    %0 = VPU.Sparsify(%arg0) : !PreConcatType -> !VPU.SparseTensor<data=!PreConcatType>
    %1 = VPU.Desparsify(%0) : !VPU.SparseTensor<data=!PreConcatType> -> !PreConcatType
    %2 = VPU.Desparsify(%0) : !VPU.SparseTensor<data=!PreConcatType> -> !PreConcatType
    %3 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 8, 0, 0]]} : !PreConcatType, !PreConcatType -> !PostConcatType
    %4 = VPU.Sparsify(%3) : !PostConcatType -> !VPU.SparseTensor<data=!PostConcatType>
    %5 = VPU.Sparsify(%3) : !PostConcatType -> !VPU.SparseTensor<data=!PostConcatType>

    %6 = VPU.NCE.Convolution(%4, %weights, %wt) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !DefaultType
    %7 = VPU.NCE.Convolution(%5, %weights, %wt) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !DefaultType


    return %6, %7 : !DefaultType, !DefaultType

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK-NOT:   VPU.Desparsify
    // CHECK-NOT:   VPU.Desparsify
    // CHECK:       [[VAL1:%.+]] = VPU.Concat([[VAL0]], [[VAL0]])
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    // CHECK-NOT:   VPU.Sparsify
    // CHECK-NOT:   VPU.Sparsify

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL2]], [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!PreConcatType = tensor<1x8x16x16xf16, {order = #NHWC}>
!PostConcatType = tensor<1x16x16x16xf16, {order = #NHWC}>
!DefaultType = tensor<1x16x16x16xf16, {order = #NHWC}>

func.func @OptimizeConcatMixedConsumers(%arg0: !PreConcatType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (!DefaultType, !PostConcatType) {
    %0 = VPU.Sparsify(%arg0) : !PreConcatType -> !VPU.SparseTensor<data=!PreConcatType>
    %1 = VPU.Desparsify(%0) : !VPU.SparseTensor<data=!PreConcatType> -> !PreConcatType
    %2 = VPU.Desparsify(%0) : !VPU.SparseTensor<data=!PreConcatType> -> !PreConcatType
    %3 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 8, 0, 0]]} : !PreConcatType, !PreConcatType -> !PostConcatType
    %4 = VPU.Sparsify(%3) : !PostConcatType -> !VPU.SparseTensor<data=!PostConcatType>

    %5 = VPU.NCE.Convolution(%4, %weights, %wt) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> !DefaultType
    %6 = VPU.MaxPool(%3) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : !PostConcatType -> !PostConcatType


    return %5, %6 : !DefaultType, !PostConcatType

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK:       [[VAL1:%.+]] = VPU.Desparsify([[VAL0]])
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL0]])

    // CHECK:       [[VAL3:%.+]] = VPU.Concat([[VAL1]], [[VAL2]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.Sparsify([[VAL3]])
    // CHECK-NOT:   VPU.Desparsify

    // CHECK:       [[VAL5:%.+]] = VPU.NCE.Convolution([[VAL4]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL6:%.+]] = VPU.MaxPool([[VAL3]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]], [[VAL6]]
}
