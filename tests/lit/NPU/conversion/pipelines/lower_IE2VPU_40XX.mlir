//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --mlir-elide-elementsattrs-if-larger 8 --lower-IE-to-VPU %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf32>,
                    [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = IE.PermuteQuantize(%0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
                    tensor<1x3x62x64xf16> -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = IE.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = IE.Convolution(%2, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
                    tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %3 : tensor<1x48x60x60xf16>

        // CHECK-DAG:       [[MAP:%.+]] = const.Declare tensor<48x1x1x4xsi32>
        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[NCE_PERM:%.+]] = VPU.NCE.Permute([[EXPAND]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[SLICE]], [[WEIGHTS]], [[MAP]])
        // CHECK-SAME:        {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [48, 16, 3, 3], strides = [1, 1]}
        // CHECK-SAME:         -> tensor<1x48x60x60xf16>
        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        return %0 : tensor<1x48x60x60xf16>

        // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        // CHECK: return [[SOFTMAX]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16> {
        %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        %1 = call @foo1(%0) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        %2 = call @foo2(%1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %2 : tensor<1x48x60x60xf16>

        // CHECK: [[CONVERT:%.+]] = VPU.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        // CHECK: [[FOO1_RES:%.+]] = call @foo1([[CONVERT]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: return [[FOO2_RES]] : tensor<1x48x60x60xf16>
    }
}
