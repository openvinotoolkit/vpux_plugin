//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --weights-dequantize-to-fake-quantize="enable-wd-blockarg-input=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @BlockArgMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xui8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @BlockArgMultToFakeQuantize(%input: tensor<1x4x28x28xui8>) -> tensor<1x4x28x28xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%input) { dstElemType = f32 } : tensor<1x4x28x28xui8> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.275000e+02> : tensor<1x1x1x1xf32>

  // CHECK:    [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28xui8> -> tensor<1x4x28x28xf32>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<1x4x28x28xf32>

  // CHECK:    [[ADD:%.+]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @BlockArgSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @BlockArgSubToFakeQuantize(%input: tensor<1x4x28x28xsi8>) -> tensor<1x4x28x28xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.407326102> : tensor<1x1x1x1xf16>
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%input) { dstElemType = f16 } : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf16>
  %1 = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  return %2 : tensor<1x4x28x28xf16>

  // CHECK-DAG:    [[CST:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<4.072270e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[IN_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[IN_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-2.280000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<2.700000e+01> : tensor<1x1x1x1xf16>

  // CHECK:    [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf16>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME:-> tensor<1x4x28x28xf16>

  // CHECK:    [[ADD:%.+]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @BlockArgMultSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @BlockArgMultSubToFakeQuantize(%input: tensor<1x4x28x28xsi8>) -> tensor<1x4x28x28xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%input) { dstElemType = f32 } : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf32>
  %1 = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %3 = IE.Add(%2, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  return %3 : tensor<1x4x28x28xf32>

  // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.140000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.350000e+01> : tensor<1x1x1x1xf32>

  // CHECK:    [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf32>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<1x4x28x28xf32>

  // CHECK:    [[ADD:%.+]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @BlockArgMultiConvertMultSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @BlockArgMultiConvertMultSubToFakeQuantize(%input: tensor<1x4x28x28xsi8>) -> tensor<1x4x28x28xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert_0 = IE.Convert(%input) { dstElemType = ui8 } : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xui8>
  %convert_1 = IE.Convert(%convert_0) { dstElemType = f16 } : tensor<1x4x28x28xui8> -> tensor<1x4x28x28xf16>
  %convert_2 = IE.Convert(%convert_1) { dstElemType = f32 } : tensor<1x4x28x28xf16> -> tensor<1x4x28x28xf32>

  %1 = IE.Subtract(%convert_2, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %3 = IE.Add(%2, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  return %3 : tensor<1x4x28x28xf32>

  // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.140000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.350000e+01> : tensor<1x1x1x1xf32>

  // CHECK:    [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf32>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<1x4x28x28xf32>

  // CHECK:    [[ADD:%.+]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @BlockArgUI4MultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xui4>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @BlockArgUI4MultToFakeQuantize(%input: tensor<1x4x28x28xui4>) -> tensor<1x4x28x28xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%input) { dstElemType = f32 } : tensor<1x4x28x28xui4> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<7.500000e+00> : tensor<1x1x1x1xf32>

  // CHECK:    [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28xui4> -> tensor<1x4x28x28xf32>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
  // CHECK-SAME: -> tensor<1x4x28x28xf32>

  // CHECK:    [[ADD:%.+]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @BlockArgSI4SubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xsi4>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @BlockArgSI4SubToFakeQuantize(%input: tensor<1x4x28x28xsi4>) -> tensor<1x4x28x28xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.407326102> : tensor<1x1x1x1xf16>
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%input) { dstElemType = f16 } : tensor<1x4x28x28xsi4> -> tensor<1x4x28x28xf16>
  %1 = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  return %2 : tensor<1x4x28x28xf16>

  // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.072270e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-8.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<7.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.080000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-9.300000e+01> : tensor<1x1x1x1xf16>

  // CHECK:    [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x4x28x28xsi4> -> tensor<1x4x28x28xf16>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
  // CHECK-SAME: -> tensor<1x4x28x28xf16>
  // CHECK:    [[ADD:%.+]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @DontBlockArgConvertNoMultNoSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @DontBlockArgConvertNoMultNoSubToFakeQuantize(%input: tensor<1x4x28x28xsi8>) -> tensor<1x4x28x28xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.407326102> : tensor<1x1x1x1xf16>
  %convert = IE.Convert(%input) { dstElemType = f16 } : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf16>
  %0 = IE.Add(%convert, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  return %0 : tensor<1x4x28x28xf16>

  // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.072270e-01> : tensor<1x1x1x1xf16>
  // CHECK:    [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf16>
  // CHECK:    [[ADD:%.+]] = IE.Add([[CONV]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

  // CHECK: return [[ADD]]
}


// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @BlockArgWithTransposeSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x48xsi8>
// CHECK-SAME: -> tensor<1x4x48x28xf16>
func.func @BlockArgWithTransposeSubToFakeQuantize(%input: tensor<1x4x28x48xsi8>) -> tensor<1x4x48x28xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.407326102> : tensor<1x1x1x1xf16>
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%input) { dstElemType = f16 } : tensor<1x4x28x48xsi8> -> tensor<1x4x28x48xf16>
  %transpose = IE.Transpose(%convert) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x4x28x48xf16> -> tensor<1x4x48x28xf16>
  %1 = IE.Subtract(%transpose, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x48x28xf16>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x48x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x48x28xf16>

  return %2 : tensor<1x4x48x28xf16>

  // CHECK-DAG:    [[CST:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<4.072270e-01> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[IN_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[IN_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-2.280000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<2.700000e+01> : tensor<1x1x1x1xf16>

  // CHECK:    [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x4x28x48xsi8> -> tensor<1x4x28x48xf16>
  // CHECK:    [[TRANSPOSE:%.+]] = IE.Transpose([[CONV]]) {order_value = #NCWH} : tensor<1x4x28x48xf16> -> tensor<1x4x48x28xf16>
  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME:-> tensor<1x4x48x28xf16>

  // CHECK:    [[ADD:%.+]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

  // CHECK: return [[ADD]]
}
