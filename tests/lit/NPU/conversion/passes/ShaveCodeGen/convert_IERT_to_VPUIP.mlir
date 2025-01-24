//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-IERT-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @SingleLayer {
  module @VPU.SW {
    llvm.func @generated_Cos0(%arg0: !llvm.ptr) {
      %0 = llvm.mlir.constant(7500 : index) : i32
      %1 = llvm.mlir.constant(150000 : index) : i32
      %2 = llvm.mlir.constant(150 : index) : i32
      %3 = llvm.mlir.constant(50 : index) : i32
      %4 = llvm.mlir.constant(20 : index) : i32
      %5 = llvm.mlir.constant(1 : index) : i32
      %6 = llvm.mlir.constant(0 : index) : i32
      %7 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %8 = llvm.getelementptr %arg0[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %9 = llvm.load %8 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      llvm.br ^bb1(%6 : i32)
    ^bb1(%10: i32):  // 2 preds: ^bb0, ^bb11
      %11 = llvm.icmp "slt" %10, %5 : i32
      llvm.cond_br %11, ^bb2, ^bb12
    ^bb2:  // pred: ^bb1
      llvm.br ^bb3(%6 : i32)
    ^bb3(%12: i32):  // 2 preds: ^bb2, ^bb10
      %13 = llvm.icmp "slt" %12, %4 : i32
      llvm.cond_br %13, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      llvm.br ^bb5(%6 : i32)
    ^bb5(%14: i32):  // 2 preds: ^bb4, ^bb9
      %15 = llvm.icmp "slt" %14, %3 : i32
      llvm.cond_br %15, ^bb6, ^bb10
    ^bb6:  // pred: ^bb5
      llvm.br ^bb7(%6 : i32)
    ^bb7(%16: i32):  // 2 preds: ^bb6, ^bb8
      %17 = llvm.icmp "slt" %16, %2 : i32
      llvm.cond_br %17, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %18 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %19 = llvm.mul %10, %1  : i32
      %20 = llvm.mul %12, %0  : i32
      %21 = llvm.add %19, %20  : i32
      %22 = llvm.mul %14, %2  : i32
      %23 = llvm.add %21, %22  : i32
      %24 = llvm.add %23, %16  : i32
      %25 = llvm.getelementptr %18[%24] : (!llvm.ptr, i32) -> !llvm.ptr, f16
      %26 = llvm.load %25 : !llvm.ptr -> f16
      %27 = llvm.intr.cos(%26)  : (f16) -> f16
      %28 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i32, array<4 x i32>, array<4 x i32>)>
      %29 = llvm.mul %10, %1  : i32
      %30 = llvm.mul %12, %0  : i32
      %31 = llvm.add %29, %30  : i32
      %32 = llvm.mul %14, %2  : i32
      %33 = llvm.add %31, %32  : i32
      %34 = llvm.add %33, %16  : i32
      %35 = llvm.getelementptr %28[%34] : (!llvm.ptr, i32) -> !llvm.ptr, f16
      llvm.store %27, %35 : f16, !llvm.ptr
      %36 = llvm.add %16, %5  : i32
      llvm.br ^bb7(%36 : i32)
    ^bb9:  // pred: ^bb7
      %37 = llvm.add %14, %5  : i32
      llvm.br ^bb5(%37 : i32)
    ^bb10:  // pred: ^bb5
      %38 = llvm.add %12, %5  : i32
      llvm.br ^bb3(%38 : i32)
    ^bb11:  // pred: ^bb3
      %39 = llvm.add %10, %5  : i32
      llvm.br ^bb1(%39 : i32)
    ^bb12:  // pred: ^bb1
      llvm.return
    }
  }
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x20x50x150xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x20x50x150xf16>
  }
  func.func @main(%arg0: memref<1x20x50x150xf16>, %arg1: memref<1x20x50x150xf16>) -> memref<1x20x50x150xf16> {
    %alloc_0 = memref.alloc() : memref<1x20x50x150xf16>
    %0 = IERT.PackMemrefs(%arg0, %alloc_0 : memref<1x20x50x150xf16>, memref<1x20x50x150xf16>) -> !IERT.PackedParams
    %1 = IERT.ExtendedCall @VPU.SW::@generated_Cos0(%0) : (!IERT.PackedParams) -> memref<1x20x50x150xf16>
    %2 = IERT.Copy inputs(%1 : memref<1x20x50x150xf16>) outputs(%arg1 : memref<1x20x50x150xf16>) -> memref<1x20x50x150xf16>
    return %2 : memref<1x20x50x150xf16>
  }
}

// CHECK: [[RESULTS2:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME: resultSegmentSizes = array<i32: 1, 0, 0>}
// CHECK-SAME: @VPU.SW::@generated_Cos0
// CHECK-SAME: inputs([[VAL2:%.+]] as [[ARG2:%.+]]: memref<1x20x50x150xf16>) outputs([[ALLOC4:%.+]] as [[ARG3:%.+]]: memref<1x20x50x150xf16>)
// CHECK-SAME: -> memref<1x20x50x150xf16, [@CMX_NN, 0]>
// CHECK: [[VAL2:%.+]] = IERT.PackMemrefs([[ARG2]], [[ARG3]] : memref<1x20x50x150xf16>, memref<1x20x50x150xf16>) -> !IERT.PackedParams
// CHECK: VPUIP.SW.Kernel.run([[VAL2]]) : !IERT.PackedParams
