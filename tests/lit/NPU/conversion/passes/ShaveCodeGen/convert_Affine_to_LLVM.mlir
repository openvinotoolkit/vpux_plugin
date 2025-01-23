//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-Affine-to-LLVM %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @SingleLayer {
  module @VPU.SW {
    func.func @generated_GenericReshape1(%arg0: !IERT.PackedParams) {
      %0 = IERT.ExtractParam(%arg0 : !IERT.PackedParams, 0) -> memref<1x1x1x1000xf16>
      %1 = IERT.ExtractParam(%arg0 : !IERT.PackedParams, 1) -> memref<1x1000xf16>
      affine.for %arg1 = 0 to 1 {
        affine.for %arg2 = 0 to 1000 {
          %2 = affine.load %0[%arg1, %arg1, %arg1, %arg2] : memref<1x1x1x1000xf16>
          affine.store %2, %1[%arg1, %arg2] : memref<1x1000xf16>
        }
      }
      return
    }
    func.func @generated_GenericReshape0(%arg0: !IERT.PackedParams) {
      %0 = IERT.ExtractParam(%arg0 : !IERT.PackedParams, 0) -> memref<1x1000xf16>
      %1 = IERT.ExtractParam(%arg0 : !IERT.PackedParams, 1) -> memref<1x1x1x1000xf16>
      affine.for %arg1 = 0 to 1 {
        affine.for %arg2 = 0 to 1 {
          affine.for %arg3 = 0 to 1 {
            affine.for %arg4 = 0 to 1000 {
              %2 = affine.load %0[%arg1, %arg4] : memref<1x1000xf16>
              affine.store %2, %1[%arg1, %arg2, %arg3, %arg4] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return
    }
    func.func @generated_Cos0(%arg0: !IERT.PackedParams) {
      %0 = IERT.ExtractParam(%arg0 : !IERT.PackedParams, 0) -> memref<1x1x1x1000xf16>
      %1 = IERT.ExtractParam(%arg0 : !IERT.PackedParams, 1) -> memref<1x1x1x1000xf16>
      affine.for %arg1 = 0 to 1 {
        affine.for %arg2 = 0 to 1 {
          affine.for %arg3 = 0 to 1 {
            affine.for %arg4 = 0 to 1000 {
              %2 = affine.load %0[%arg1, %arg2, %arg3, %arg4] : memref<1x1x1x1000xf16>
              %3 = math.cos %2 : f16
              affine.store %3, %1[%arg1, %arg2, %arg3, %arg4] : memref<1x1x1x1000xf16>
            }
          }
        }
      }
      return
    }
  }
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1000xf16>
  }
  func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %alloc = memref.alloc() : memref<1x1x1x1000xf16>
    %0 = IERT.PackMemrefs(%arg0, %alloc : memref<1x1000xf16>, memref<1x1x1x1000xf16>) -> !IERT.PackedParams
    %1 = IERT.ExtendedCall @VPU.SW::@generated_GenericReshape0(%0) : (!IERT.PackedParams) -> memref<1x1x1x1000xf16>
    %alloc_0 = memref.alloc() : memref<1x1x1x1000xf16>
    %2 = IERT.PackMemrefs(%1, %alloc_0 : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>) -> !IERT.PackedParams
    %3 = IERT.ExtendedCall @VPU.SW::@generated_Cos0(%2) : (!IERT.PackedParams) -> memref<1x1x1x1000xf16>
    %alloc_1 = memref.alloc() : memref<1x1000xf16>
    %4 = IERT.PackMemrefs(%3, %alloc_1 : memref<1x1x1x1000xf16>, memref<1x1000xf16>) -> !IERT.PackedParams
    %5 = IERT.ExtendedCall @VPU.SW::@generated_GenericReshape1(%4) : (!IERT.PackedParams) -> memref<1x1000xf16>
    %6 = VPUIP.Copy inputs(%5 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    return %6 : memref<1x1000xf16>
  }
}

// CHECK: llvm.func @generated_GenericReshape1
// CHECK: = llvm.mlir.constant
// CHECK: = llvm.getelementptr
// CHECK: = llvm.load
// CHECK: llvm.return

// CHECK: llvm.func @generated_GenericReshape0
// CHECK: = llvm.mlir.constant
// CHECK: = llvm.getelementptr
// CHECK: = llvm.load
// CHECK: llvm.return

// CHECK: llvm.func @generated_Cos0
// CHECK: = llvm.mlir.constant
// CHECK: = llvm.getelementptr
// CHECK: = llvm.load
// CHECK: llvm.return
