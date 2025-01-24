//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-sw-layers-to-Affine %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @SingleLayer {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1000xf16>
  }
  func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %alloc = memref.alloc() : memref<1x1x1x1000xf16>
    %0 = IERT.GenericReshape inputs(%arg0 : memref<1x1000xf16>) outputs(%alloc : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    %alloc_0 = memref.alloc() : memref<1x1x1x1000xf16>
    %1 = IERT.Cos inputs(%0 : memref<1x1x1x1000xf16>) outputs(%alloc_0 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    %alloc_1 = memref.alloc() : memref<1x1000xf16>
    %2 = IERT.GenericReshape inputs(%1 : memref<1x1x1x1000xf16>) outputs(%alloc_1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    %3 = VPUIP.Copy inputs(%2 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    return %3 : memref<1x1000xf16>
  }
}

// CHECK: func.func @generated_GenericReshape1([[ARG0:%.+]]: !IERT.PackedParams) {
// CHECK: [[EP0_0:%.+]] = IERT.ExtractParam([[ARG0]] : !IERT.PackedParams, 0) -> memref<1x1x1x1000xf16>
// CHECK: [[EP0_1:%.+]] = IERT.ExtractParam([[ARG0]] : !IERT.PackedParams, 1) -> memref<1x1000xf16>
// CHECK: affine.for [[ARG0_1:%.+]] = 0 to 1 {
// CHECK: affine.for [[ARG0_2:%.+]] = 0 to 1000 {
// CHECK: [[AL0:%.+]] = affine.load [[EP0_0]][[[ARG0_1]], [[ARG0_1]], [[ARG0_1]], [[ARG0_2]]] : memref<1x1x1x1000xf16>
// CHECK: affine.store [[AL0]], [[EP0_1]][[[ARG0_1]], [[ARG0_2]]] : memref<1x1000xf16>
// CHECK: return

// CHECK: func.func @generated_GenericReshape0([[ARG1:%.+]]: !IERT.PackedParams) {
// CHECK: [[EP1_0:%.+]] = IERT.ExtractParam([[ARG1]] : !IERT.PackedParams, 0) -> memref<1x1000xf16>
// CHECK: [[EP1_1:%.+]] = IERT.ExtractParam([[ARG1]] : !IERT.PackedParams, 1) -> memref<1x1x1x1000xf16>
// CHECK: affine.for [[ARG1_1:%.+]] = 0 to 1 {
// CHECK: affine.for [[ARG1_2:%.+]] = 0 to 1 {
// CHECK: affine.for [[ARG1_3:%.+]] = 0 to 1 {
// CHECK: affine.for [[ARG1_4:%.+]] = 0 to 1000 {
// CHECK: [[AL1:%.+]] = affine.load [[EP1_0]][[[ARG1_1]], [[ARG1_4]]] : memref<1x1000xf16>
// CHECK: affine.store [[AL1]], [[EP1_1]][[[ARG1_1]], [[ARG1_2]], [[ARG1_3]], [[ARG1_4]]] : memref<1x1x1x1000xf16>
// CHECK: return

// CHECK: func.func @generated_Cos0([[ARG2:%.+]]: !IERT.PackedParams) {
// CHECK: [[EP2_0:%.+]] = IERT.ExtractParam([[ARG2]] : !IERT.PackedParams, 0) -> memref<1x1x1x1000xf16>
// CHECK: [[EP2_1:%.+]] = IERT.ExtractParam([[ARG2]] : !IERT.PackedParams, 1) -> memref<1x1x1x1000xf16>
// CHECK: affine.for [[ARG2_1:%.+]] = 0 to 1 {
// CHECK: affine.for [[ARG2_2:%.+]] = 0 to 1 {
// CHECK: affine.for [[ARG2_3:%.+]] = 0 to 1 {
// CHECK: affine.for [[ARG2_4:%.+]] = 0 to 1000 {
// CHECK: [[AL2:%.+]] = affine.load [[EP2_0]][[[ARG2_1]], [[ARG2_2]], [[ARG2_3]], [[ARG2_4]]] : memref<1x1x1x1000xf16>
// CHECK: [[COS:%.+]] = math.cos [[AL2]] : f16
// CHECK: affine.store [[COS]], [[EP2_1]][[[ARG2_1]], [[ARG2_2]], [[ARG2_3]], [[ARG2_4]]] : memref<1x1x1x1000xf16>
// CHECK: return

// CHECK: func.func @main([[ARG3_0:%.+]]: memref<1x1000xf16>, [[ARG3_1:%.+]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
// CHECK: [[PM0:%.+]] = IERT.PackMemrefs([[ARG3_0]], {{.+}} : memref<1x1000xf16>, memref<1x1x1x1000xf16>) -> !IERT.PackedParams
// CHECK: [[EC0:%.+]] = IERT.ExtendedCall @VPU.SW::@generated_GenericReshape0([[PM0]]) : (!IERT.PackedParams) -> memref<1x1x1x1000xf16>
// CHECK: [[PM1:%.+]] = IERT.PackMemrefs([[EC0]], {{.+}} : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>) -> !IERT.PackedParams
// CHECK: [[EC1:%.+]] = IERT.ExtendedCall @VPU.SW::@generated_Cos0([[PM1]]) : (!IERT.PackedParams) -> memref<1x1x1x1000xf16>
// CHECK: [[PM2:%.+]] = IERT.PackMemrefs([[EC1]], {{.+}} : memref<1x1x1x1000xf16>, memref<1x1000xf16>) -> !IERT.PackedParams
// CHECK: [[EC2:%.+]] = IERT.ExtendedCall @VPU.SW::@generated_GenericReshape1([[PM2]]) : (!IERT.PackedParams) -> memref<1x1000xf16>
// CHECK: return {{.+}} : memref<1x1000xf16>
