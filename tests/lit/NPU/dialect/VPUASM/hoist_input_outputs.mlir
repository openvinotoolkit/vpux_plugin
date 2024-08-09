//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" -allow-unregistered-dialect --hoist-input-outputs %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @hoistIO {

IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1xui8>
} outputsInfo : {
    DataInfo "prob" : tensor<2xf16>
    DataInfo "age_conv3" : tensor<3xf32>
} profilingOutputsInfo : {
    DataInfo "profilingOutput" {
    } : tensor<4xui64>
}

func.func @main(%arg0: memref<1xui8>, %arg1: memref<2xf16>, %arg2: memref<3xf32>, %arg3: memref<4xui64>) -> (memref<2xf16>, memref<3xf32>, memref<4xui64>) {

  "foo"(%arg0, %arg1) : (memref<1xui8>,  memref<2xf16>) -> ()
  "bar"(%arg2, %arg3) : (memref<3xf32>, memref<4xui64>) -> ()
  "foobar"(%arg0, %arg1, %arg2, %arg3) : (memref<1xui8>,  memref<2xf16>, memref<3xf32>, memref<4xui64>) -> ()

  VPUMI40XX.OpRanges
}
}

//CHECK:      VPUASM.IOBindings inputDeclarations
//CHECK-NEXT:   VPUASM.DeclareBuffer @data_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1xui8> :  swizzling(0)>
//CHECK-NEXT: outputDeclarations
//CHECK-NEXT:   VPUASM.DeclareBuffer @prob_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<2xf16> :  swizzling(0)>
//CHECK-NEXT:   VPUASM.DeclareBuffer @age_conv3_buffDecl !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<3xf32> :  swizzling(0)>
//CHECK-NEXT: profilingBuffDeclarations
//CHECK-NEXT:   VPUASM.DeclareBuffer @profilingOutput_buffDecl !VPUASM.Buffer< "ProfilingOutput"[0] <0> : memref<4xui64> :  swizzling(0)>

//CHECK: func.func @main() {
//CHECK: [[ARG0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1xui8>
//CHECK: [[ARG1:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<2xf16>
//CHECK: [[ARG2:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<3xf32>
//CHECK: [[ARG3:%.*]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<4xui64>
//CHECK: "foo"([[ARG0]], [[ARG1]])
//CHECK: "bar"([[ARG2]], [[ARG3]])
//CHECK: "foobar"([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]])

// CHECK: VPUMI40XX.OpRanges
