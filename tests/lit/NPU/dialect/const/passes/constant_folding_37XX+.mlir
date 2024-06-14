//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --constant-folding-pipe %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func.func @ConstFoldWithSwizzlingSubByte(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<true> : tensor<100x1x1x384xi1>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
  %buf = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // CHECK:      VPURT.DeclareVirtualBarrier
  // CHECK-DAG:      [[CST:%.+]] = const.Declare memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<true> : tensor<40960x1x1x1xi1>
  // CHECK-NOT:    [#const.SwizzleConstant<5 : i64, 3 : i64>]
  // CHECK:      [[BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA
  // CHECK-SAME    inputs([[CST]] : memref<40960x1x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK-SAME    outputs([[BUF]] : memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufferDdr = memref<512x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = memref<512x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

// Swizzling transformation needs to use always state of constant buffer which is an input for this transformation
func.func @ConstFoldWithSwizzlingWhereInputIsDifferentThanRawStorageValue(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {

  %cst = const.Declare memref<512x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<1.000000e+00> : tensor<32x1x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!quant.uniform<u8<0:254>:f16, 1.000000e+00>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>>, #const.Reshape<[32, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>, #const.SwizzleConstant<5 : i64, 3 : i64>]

  %buf = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task {
    %0 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // CHECK:      const.Declare memref<512x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> =
  // CHECK-SAME    dense<"0x010000000000000000000000000000000
  // CHECK-NOT:    [#const.SwizzleConstant<5 : i64, 3 : i64>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @ConstFoldWithSwizzlingWhereContentShapeIsDifferentFromOpShape() -> memref<768x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> {

  %cst = const.Declare memref<768x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<[[[[1], [2], [3]]]]> : tensor<1x1x3x1xui8>, [#const.Reshape<[768, 1, 1, 1]>, #const.Broadcast<3, 1>, #const.SwizzleConstant<5 : i64, 3 : i64>]

  return %cst: memref<768x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>

  // CHECK:      [[CST:%.+]] = const.Declare memref<768x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
  // CHECK-NOT:  [#const.SwizzleConstant<5 : i64, 3 : i64>]
}
