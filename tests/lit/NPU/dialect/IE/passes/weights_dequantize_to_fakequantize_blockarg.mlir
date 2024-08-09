//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --weights-dequantize-to-fake-quantize="enable-wd-blockarg-input=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @BlockArgMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x4x28x28xui8>
func.func @BlockArgMultToFakeQuantize(%input: tensor<1x4x28x28xui8>) -> tensor<1x4x28x28xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%input) { dstElemType = f32 } : tensor<1x4x28x28xui8> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  //CHECK-DAG:    [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.275000e+02> : tensor<1x1x1x1xf32>

  //CHECK:    [[CONV:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28xui8> -> tensor<1x4x28x28xf32>
  //CHECK:    [[FQ:%.*]] = IE.FakeQuantize([[CONV]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  //CHECK:    [[ADD:%.*]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  //CHECK:    return [[ADD]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @BlockArgSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x4x28x28xsi8>
func.func @BlockArgSubToFakeQuantize(%input: tensor<1x4x28x28xsi8>) -> tensor<1x4x28x28xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.407326102> : tensor<1x1x1x1xf16>
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%input) { dstElemType = f16 } : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf16>
  %1 = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  return %2 : tensor<1x4x28x28xf16>

  //CHECK-DAG:    [[CST:%.*]]  = const.Declare tensor<1x1x1x1xf16> = dense<4.072270e-01> : tensor<1x1x1x1xf16>
  //CHECK-DAG:    [[CST_0:%.*]]  = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
  //CHECK-DAG:    [[CST_1:%.*]]  = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
  //CHECK-DAG:    [[CST_2:%.*]]  = const.Declare tensor<1x1x1x1xf16> = dense<-2.280000e+02> : tensor<1x1x1x1xf16>
  //CHECK-DAG:    [[CST_3:%.*]]  = const.Declare tensor<1x1x1x1xf16> = dense<2.700000e+01> : tensor<1x1x1x1xf16>

  //CHECK:    [[CONV:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf16>
  //CHECK:    [[FQ:%.*]] = IE.FakeQuantize([[CONV]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>
  //CHECK:    [[ADD:%.*]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  //CHECK:    return [[ADD]] : tensor<1x4x28x28xf16>
}

// -----

// CHECK-LABEL: @BlockArgMultSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x4x28x28xsi8>
func.func @BlockArgMultSubToFakeQuantize(%input: tensor<1x4x28x28xsi8>) -> tensor<1x4x28x28xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%input) { dstElemType = f32 } : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf32>
  %1 = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %3 = IE.Add(%2, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  return %3 : tensor<1x4x28x28xf32>

  //CHECK-DAG:    [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.140000e+02> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.350000e+01> : tensor<1x1x1x1xf32>

  //CHECK:    [[CONV:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf32>
  //CHECK:    [[FQ:%.*]] = IE.FakeQuantize([[CONV]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  //CHECK:    [[ADD:%.*]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  //CHECK:    return [[ADD]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @BlockArgMultiConvertMultSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x4x28x28xsi8>
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

  //CHECK-DAG:    [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.140000e+02> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.350000e+01> : tensor<1x1x1x1xf32>

  //CHECK:    [[CONV:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf32>
  //CHECK:    [[FQ:%.*]] = IE.FakeQuantize([[CONV]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  //CHECK:    [[ADD:%.*]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  //CHECK:    return [[ADD]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @BlockArgUI4MultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x4x28x28xui4>
func.func @BlockArgUI4MultToFakeQuantize(%input: tensor<1x4x28x28xui4>) -> tensor<1x4x28x28xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%input) { dstElemType = f32 } : tensor<1x4x28x28xui4> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  //CHECK-DAG:    [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1x1xf32>
  //CHECK-DAG:    [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<7.500000e+00> : tensor<1x1x1x1xf32>

  //CHECK:    [[CONV:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x28x28xui4> -> tensor<1x4x28x28xf32>
  //CHECK:    [[FQ:%.*]] = IE.FakeQuantize([[CONV]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  //CHECK:    [[ADD:%.*]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  //CHECK:    return [[ADD]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @BlockArgSI4SubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.*]]:  tensor<1x4x28x28xsi4>
func.func @BlockArgSI4SubToFakeQuantize(%input: tensor<1x4x28x28xsi4>) -> tensor<1x4x28x28xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.407326102> : tensor<1x1x1x1xf16>
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%input) { dstElemType = f16 } : tensor<1x4x28x28xsi4> -> tensor<1x4x28x28xf16>
  %1 = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  return %2 : tensor<1x4x28x28xf16>

  //CHECK-DAG:    [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.072270e-01> : tensor<1x1x1x1xf16>
  //CHECK-DAG:    [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-8.000000e+00> : tensor<1x1x1x1xf16>
  //CHECK-DAG:    [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.000000e+00> : tensor<1x1x1x1xf16>
  //CHECK-DAG:    [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.080000e+02> : tensor<1x1x1x1xf16>
  //CHECK-DAG:    [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-9.300000e+01> : tensor<1x1x1x1xf16>

  //CHECK:    [[CONV:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x4x28x28xsi4> -> tensor<1x4x28x28xf16>
  //CHECK:    [[FQ:%.*]] = IE.FakeQuantize([[CONV]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>
  //CHECK:    [[ADD:%.*]] = IE.Add([[FQ]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  //CHECK:    return [[ADD]] : tensor<1x4x28x28xf16>
}

// -----

// CHECK-LABEL: @DontBlockArgConvertNoMultNoSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x4x28x28xsi8>
func.func @DontBlockArgConvertNoMultNoSubToFakeQuantize(%input: tensor<1x4x28x28xsi8>) -> tensor<1x4x28x28xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.407326102> : tensor<1x1x1x1xf16>
  %convert = IE.Convert(%input) { dstElemType = f16 } : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf16>
  %0 = IE.Add(%convert, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  return %0 : tensor<1x4x28x28xf16>

  //CHECK:    [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.072270e-01> : tensor<1x1x1x1xf16>
  //CHECK:    [[CONV:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x4x28x28xsi8> -> tensor<1x4x28x28xf16>
  //CHECK:    [[ADD:%.*]] = IE.Add([[CONV]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x28x28xf16>

  //CHECK:    return [[ADD]] : tensor<1x4x28x28xf16>
}
