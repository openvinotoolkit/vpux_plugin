//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FuseConstDivideToMatMul
module @FuseConstDivideToMatMul {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x3x24xf16>
        DataInfo "input" : tensor<1x64x3x24xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK-LABEL: @main
    func.func @main(%arg0: tensor<1x3x64x24xf16>, %arg1: tensor<1x3x64x24xf16>) -> tensor<1x3x64x64xf16> {
        %cst_0 = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
        %cst_1 = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>
        %cst_16 = const.Declare tensor<1xf16> = dense<-8.01463317> : tensor<1xf16>
        %cst_17 = const.Declare tensor<1xf16> = dense<7.95201873> : tensor<1xf16>
        %cst_18 = const.Declare tensor<1xf16> = dense<2.460000e+02> : tensor<1xf16>
        %cst_fq = IE.FakeQuantize(%cst_18, %cst_0, %cst_1, %cst_16, %cst_17) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
        } : tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>

        %28 = IE.MatMul(%arg0, %arg1) {transpose_b}
            : tensor<1x3x64x24xf16>, tensor<1x3x64x24xf16> -> tensor<1x3x64x64xf16>

        %29 = IE.Divide(%28, %cst_fq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x3x64x64xf16>, tensor<1xf16> -> tensor<1x3x64x64xf16>

        return %29 : tensor<1x3x64x64xf16>

        // CHECK: [[CONV0:%.+]] = IE.Convolution
        // CHECK-SAME:  static_scale = 0.135327876 : f32
        // CHECK: [[CONV0_RESHAPE:%.+]] = IE.AffineReshape([[CONV0]])
        // CHECK-SAME: -> tensor<1x64x64x1xf16, {order = #NHWC}>
        // CHECK: [[CONV0_PERMUTE:%.+]] = IE.PermuteCast([[CONV0_RESHAPE]])

        // CHECK: [[CONV1:%.+]] = IE.Convolution
        // CHECK-SAME:  static_scale = 0.135327876 : f32
        // CHECK: [[CONV1_RESHAPE:%.+]] = IE.AffineReshape([[CONV1]])
        // CHECK-SAME: -> tensor<1x64x64x1xf16, {order = #NHWC}>
        // CHECK: [[CONV1_PERMUTE:%.+]] = IE.PermuteCast([[CONV1_RESHAPE]])

        // CHECK: [[CONV2:%.+]] = IE.Convolution
        // CHECK-SAME:  static_scale = 0.135327876 : f32
        // CHECK: [[CONV2_RESHAPE:%.+]] = IE.AffineReshape([[CONV2]])
        // CHECK-SAME: -> tensor<1x64x64x1xf16, {order = #NHWC}>
        // CHECK: [[CONV2_PERMUTE:%.+]] = IE.PermuteCast([[CONV2_RESHAPE]])


        // CHECK: [[CONV0_RES:%.+]] = IE.AffineReshape([[CONV0_PERMUTE]])
        // CHECK: [[CONV1_RES:%.+]] = IE.AffineReshape([[CONV1_PERMUTE]])
        // CHECK: [[CONV2_RES:%.+]] = IE.AffineReshape([[CONV2_PERMUTE]])
        // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0_RES]], [[CONV1_RES]], [[CONV2_RES]])
        // CHECK-NEXT: return [[CONCAT]]
    }
}
