//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ShaveCodeGen %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

module @SingleLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "cos" : tensor<1x1000xf16>
    }

func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = IE.Cos(%arg0) : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>
}

}

// CHECK: llvm.func @generated_Cos0([[ARG0:%.+]]: !llvm.ptr<struct<(ptr<f16>, ptr<f16>, i32, array<4 x i32>, array<4 x i32>)>>) {
// CHECK: [[VAL2:%.+]] = llvm.mlir.constant(1 : index) : i32
// CHECK: [[VAL3:%.+]] = llvm.getelementptr [[ARG0]][[[VAL1:.+]]] : (!llvm.ptr<struct<(ptr<f16>, ptr<f16>, i32, array<4 x i32>, array<4 x i32>)>>) -> !llvm.ptr<struct<(ptr<f16>, ptr<f16>, i32, array<4 x i32>, array<4 x i32>)>>
// CHECK: [[VAL4:%.+]] = llvm.load [[VAL3]] : !llvm.ptr<struct<(ptr<f16>, ptr<f16>, i32, array<4 x i32>, array<4 x i32>)>>
// CHECK: [[VAL37:%.+]] = llvm.intr.cos([[VAL36:%.+]]) : (f16) -> f16
// CHECK: llvm.return

// CHECK: IE.CNNNetwork entryPoint : @main inputsInfo : {

// CHECK: func.func @main([[ARGB0:%.+]]: memref<1x1000xf16, @DDR>, %arg1: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR> {
// CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_GenericReshape0 inputs([[VAL6:%.+]] as [[ARG2:%.+]]: memref<1x1000xf16>) outputs([[VAL7:%.+]] as [[ARG3:%.+]]: memref<1x1x1x1000xf16>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
// CHECK: [[VAL10:%.+]] = IERT.PackMemrefs([[ARG2]], [[ARG3]] : memref<1x1000xf16>, memref<1x1x1x1000xf16>) -> !IERT.PackedParams
// CHECK: VPUIP.SW.Kernel.run([[VAL10]]) : !IERT.PackedParams
// CHECK: }
// CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_Cos0 inputs([[VAL7:%.+]] as [[ARG2:%.+]]: memref<1x1x1x1000xf16>) outputs([[VAL8:%.+]] as [[ARG3:%.+]]: memref<1x1x1x1000xf16>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
// CHECK: [[VAL10:%.+]] = IERT.PackMemrefs([[ARG2]], [[ARG3]] : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>) -> !IERT.PackedParams
// CHECK: VPUIP.SW.Kernel.run([[VAL10]]) : !IERT.PackedParams
// CHECK: }
// CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_GenericReshape1 inputs([[VAL8:%.+]] as [[ARG2:%.+]]: memref<1x1x1x1000xf16>) outputs([[VAL9:%.+]] as [[ARG3:%.+]]: memref<1x1000xf16>) on tile 0 -> memref<1x1000xf16, [@CMX_NN, 0]>{
// CHECK: [[VAL10:%.+]] = IERT.PackMemrefs([[ARG2]], [[ARG3]] : memref<1x1x1x1000xf16>, memref<1x1000xf16>) -> !IERT.PackedParams
// CHECK: VPUIP.SW.Kernel.run([[VAL10]]) : !IERT.PackedParams
// CHECK: }
// CHECK: return [[ARG1:%.+]] : memref<1x1000xf16, @DDR>

// -----

module @SingleLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "hswish" : tensor<1x1000xf16>
    }

func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = IE.HSwish(%arg0) : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>
}

}

// CHECK: llvm.func @generated_HSwish0([[ARG0:%.+]]: !llvm.ptr<struct<(ptr<f16>, ptr<f16>, i32, array<4 x i32>, array<4 x i32>)>>) {
// CHECK: [[VAL2:%.+]] = llvm.mlir.constant(1 : index) : i32
// CHECK: [[VAL3:%.+]] = llvm.getelementptr [[ARG0]][[[VAL1:.+]]] : (!llvm.ptr<struct<(ptr<f16>, ptr<f16>, i32, array<4 x i32>, array<4 x i32>)>>) -> !llvm.ptr<struct<(ptr<f16>, ptr<f16>, i32, array<4 x i32>, array<4 x i32>)>>
// CHECK: [[VAL4:%.+]] = llvm.load [[VAL3]] : !llvm.ptr<struct<(ptr<f16>, ptr<f16>, i32, array<4 x i32>, array<4 x i32>)>>
// CHECK: [[VAL34:%.+]] = llvm.fdiv [[VAL33:%.+]], [[VAL0:%.+]]  : f16
// CHECK: llvm.return

// CHECK: IE.CNNNetwork entryPoint : @main inputsInfo : {

// CHECK: func.func @main([[ARGB0:%.+]]: memref<1x1000xf16, @DDR>, %arg1: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR> {
// CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_GenericReshape0 inputs([[VAL6:%.+]] as [[ARG2:%.+]]: memref<1x1000xf16>) outputs([[VAL7:%.+]] as [[ARG3:%.+]]: memref<1x1x1x1000xf16>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
// CHECK: [[VAL10:%.+]] = IERT.PackMemrefs([[ARG2]], [[ARG3]] : memref<1x1000xf16>, memref<1x1x1x1000xf16>) -> !IERT.PackedParams
// CHECK: VPUIP.SW.Kernel.run([[VAL10]]) : !IERT.PackedParams
// CHECK: }
// CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_HSwish0 inputs([[VAL7:%.+]] as [[ARG2:%.+]]: memref<1x1x1x1000xf16>) outputs([[VAL8:%.+]] as [[ARG3:%.+]]: memref<1x1x1x1000xf16>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
// CHECK: [[VAL10:%.+]] = IERT.PackMemrefs([[ARG2]], [[ARG3]] : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>) -> !IERT.PackedParams
// CHECK: VPUIP.SW.Kernel.run([[VAL10]]) : !IERT.PackedParams
// CHECK: }
// CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@generated_GenericReshape1 inputs([[VAL8:%.+]] as [[ARG2:%.+]]: memref<1x1x1x1000xf16>) outputs([[VAL9:%.+]] as [[ARG3:%.+]]: memref<1x1000xf16>) on tile 0 -> memref<1x1000xf16, [@CMX_NN, 0]>{
// CHECK: [[VAL10:%.+]] = IERT.PackMemrefs([[ARG2]], [[ARG3]] : memref<1x1x1x1000xf16>, memref<1x1000xf16>) -> !IERT.PackedParams
// CHECK: VPUIP.SW.Kernel.run([[VAL10]]) : !IERT.PackedParams
// CHECK: }
// CHECK: return [[ARG1:%.+]] : memref<1x1000xf16, @DDR>
