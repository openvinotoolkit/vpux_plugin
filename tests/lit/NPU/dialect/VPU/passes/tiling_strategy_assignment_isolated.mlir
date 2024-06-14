//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX
// TODO: #-81889 restore arch-VPUX37XX and arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16:3, {0.1:127, 0.2:127, 0.3:127, 0.4:127, 0.5:127, 0.6:127, 0.7:127, 0.8:127}>

// // 1x16x4x8xf16 + weights_table + act_window + profiling buffer
IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 4800 bytes of @CMX_NN
}

// CHECK-LABEL: func.func @MultiAxesAndPerAxisQuant
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x8x8x!qElemType, {order = #NHWC}>
func.func @MultiAxesAndPerAxisQuant(
        %input: tensor<1x32x8x8x!qElemType, {order = #NHWC}>)
            -> tensor<1x32x8x8x!qElemType, {order = #NHWC}> {
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%input, %weights_table, %activation_window) {
        activation_window_channel_length = 54 : i64,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
    } -> tensor<1x32x8x8x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x32x8x8x!qElemType, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK-SAME:      = dense<1> : tensor<32x1x1x4xsi32>

    // CHECK-DAG:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-SAME:      = dense<1> : tensor<1x1x1x16xui8>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool(%arg0, [[WEIGHTS_TABLE]] , [[ACTIVATION_WINDOW]] )
    // CHECK-SAME:          activation_window_channel_length = 54 : i64,
    // CHECK-SAME:          kernel_size = [3, 3],
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          strides = [1, 1],
    // CHECK-SAME:          tilingStrategy = [1, 1, 8, 1]}
    // CHECK-SAME:      -> tensor<1x32x8x8x!qElemType, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<1x32x8x8x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1000000 bytes of @CMX_NN
}

// CHECK-LABEL: func.func @SplitNCEConvOverHAndAlignW
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x3x224x224xf16>
func.func @SplitNCEConvOverHAndAlignW(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x64x111x111xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x1x1x32xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[64, 1, 1, 27]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 5]>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x32xui8> = dense<10> : tensor<1x1x1x32xui8>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table, %activation_window ) {
        activation_window_channel_length = 81 : i64,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [64, 3, 3, 3],
        strides = [2, 2]
    } -> tensor<1x64x111x111xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    return %0 : tensor<1x64x111x111xf16, {order = #NHWC}>

    // CHECK-DAG:       [[ACT_WIN:%.+]] = const.Declare tensor<1x1x1x32xui8> = dense<10> : tensor<1x1x1x32xui8>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<64x1x1x32xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[64, 1, 1, 27]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 5]>, #const.Reorder<#NHWC>]

    // CHECK:           [[CONV_0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTS_TABLE]], [[ACT_WIN]] )
    // CHECK-SAME:          {activation_window_channel_length = 81 : i64,
    // CHECK-SAME:           pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [64, 3, 3, 3], strides = [2, 2], tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:              -> tensor<1x64x111x111xf16, {order = #NHWC}>

    // CHECK:           return [[CONV_0]] : tensor<1x64x111x111xf16, {order = #NHWC}>
}
