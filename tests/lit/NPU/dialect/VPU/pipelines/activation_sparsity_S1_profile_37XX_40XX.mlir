//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --enable-act-sparsity="enable-activation-sparsity=true act-sparsity-profile=S1" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SingleOp(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1],
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ChainedOps(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1],
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1],
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.MaxPool(%1) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1],
        opaque_ppe = #VPU.PPEStub<>
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-SAME:      !VPU.SparseTensor
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       [[VAL3:%.+]] = VPU.MaxPool([[VAL2]])
    // CHECK:       return [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SparseNonSparseSparseChain(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1],
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = VPU.MaxPool(%0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1],
        opaque_ppe = #VPU.PPEStub<>
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {
            op_type = #VPU.eltwise_type<ADD>,
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       [[VAL2:%.+]] = VPU.MaxPool([[VAL1]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Eltwise([[VAL2]], [[VAL2]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       return [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @Resnet50Pattern(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1],
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1],
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %weights, %wt) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1],
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %3 = VPU.NCE.Eltwise(%0, %2) {
            op_type = #VPU.eltwise_type<ADD>,
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %4 = VPU.NCE.Eltwise(%3, %3) {
            op_type = #VPU.eltwise_type<AND>,
            opaque_ppe = #VPU.PPEStub<>
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-SAME:      !VPU.SparseTensor
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-SAME:      !VPU.SparseTensor
    // CHECK:       [[VAL4:%.+]] = VPU.NCE.Convolution([[VAL3]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       [[VAL5:%.+]] = VPU.NCE.Eltwise([[VAL2]], [[VAL4]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       [[VAL6:%.+]] = VPU.NCE.Eltwise([[VAL5]], [[VAL5]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<AND>
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       return [[VAL6]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!PreConcatType = tensor<1x16x16x16xf16, {order = #NHWC}>
!PostConcatType = tensor<1x32x16x16xf16, {order = #NHWC}>

func.func @GooglenetLikePattern(%arg0: !PreConcatType,
                        %wt: tensor<16x1x1x4xsi32>,
                        %weights: tensor<16x16x1x1xf16, {order = #NHWC}>,
                        %weights2: tensor<32x32x1x1xf16, {order = #NHWC}>,
                        %wt2: tensor<32x1x1x4xsi32>) -> (!PostConcatType, !PostConcatType, !PostConcatType) {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = #VPU.eltwise_type<AND>,
        opaque_ppe = #VPU.PPEStub<>
      } -> !PreConcatType

    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          rawFilterShape = [16, 16, 1, 1],
          strides = [1, 1],
          opaque_ppe = #VPU.PPEStub<>
      } -> !PreConcatType

    %2 = VPU.NCE.Convolution(%0, %weights, %wt) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1],
        opaque_ppe = #VPU.PPEStub<>
      } -> !PreConcatType

    %3 = VPU.NCE.Eltwise(%2, %2) {
        op_type = #VPU.eltwise_type<ADD>,
        opaque_ppe = #VPU.PPEStub<>
      } -> !PreConcatType

    %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : !PreConcatType, !PreConcatType -> !PostConcatType

    %5 = VPU.NCE.Eltwise(%4, %4) {
        op_type = #VPU.eltwise_type<AND>,
        opaque_ppe = #VPU.PPEStub<>
      } -> !PostConcatType

    %6 = VPU.NCE.Convolution(%4, %weights2, %wt2) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [32, 32, 1, 1],
        strides = [1, 1],
        opaque_ppe = #VPU.PPEStub<>
      } -> !PostConcatType

    %7 = VPU.MaxPool(%4) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1],
        opaque_ppe = #VPU.PPEStub<>
      } : !PostConcatType -> !PostConcatType

    return %5, %6, %7 : !PostConcatType, !PostConcatType, !PostConcatType

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Eltwise(%arg0, %arg0)
    // CHECK-SAME:      op_type = #VPU.eltwise_type<AND>
    // CHECK-SAME:      !VPU.SparseTensor
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-SAME:      !VPU.SparseTensor
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Eltwise([[VAL2]], [[VAL2]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      !VPU.SparseTensor
    // CHECK:       [[VAL4:%.+]] = VPU.Concat([[VAL1]], [[VAL3]])
    // CHECK:       [[VAL5:%.+]] = VPU.Desparsify([[VAL4]])
    // CHECK:       [[VAL6:%.+]] = VPU.NCE.Eltwise([[VAL5]], [[VAL5]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<AND>
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       [[VAL7:%.+]] = VPU.NCE.Convolution([[VAL4]], %arg3, %arg4)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK:       [[VAL8:%.+]] = VPU.MaxPool([[VAL5]])
    // CHECK:       return [[VAL6]], [[VAL7]], [[VAL8]]
}
