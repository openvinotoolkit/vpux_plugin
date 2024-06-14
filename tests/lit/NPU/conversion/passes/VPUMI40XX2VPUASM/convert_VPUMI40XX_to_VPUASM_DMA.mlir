//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
IE.ExecutorResource 1 of @DMA_NN
IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @nndma_1d_to_1d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_1d_to_1d() {
    %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC}, @DDR>)
        outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   descriptor(<numPlanes = 0 : i32, len = 801360 : i32, srcWidth = 801360 : i32, srcStride = 801360 : i32, srcPlaneStride = 0 : i32, dstWidth = 801360 : i32, dstStride = 801360 : i32, dstPlaneStride = 0 : i32>)

    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
IE.ExecutorResource 1 of @DMA_NN
IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @nndma_1d_to_3d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_1d_to_3d() {
    %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC}, @DDR>)
        outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   descriptor(<numPlanes = 90 : i32, len = 8904 : i32, srcWidth = 8904 : i32, srcStride = 8904 : i32, srcPlaneStride = 8904 : i32, dstWidth = 168 : i32, dstStride = 360 : i32, dstPlaneStride = 115200 : i32>)

    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
IE.ExecutorResource 1 of @DMA_NN
IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @nndma_2d_to_3d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_2d_to_3d() {
    %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR>)
        outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   descriptor(<numPlanes = 90 : i32, len = 8904 : i32, srcWidth = 168 : i32, srcStride = 192 : i32, srcPlaneStride = 10176 : i32, dstWidth = 168 : i32, dstStride = 360 : i32, dstPlaneStride = 115200 : i32>)

    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
IE.ExecutorResource 1 of @DMA_NN
IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @nndma_3d_to_2d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_3d_to_2d() {
    %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkInput> [0] <20368000> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>)
        outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC, strides = [457920, 1, 5088, 96]}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   descriptor(<numPlanes = 90 : i32, len = 8904 : i32, srcWidth = 168 : i32, srcStride = 360 : i32, srcPlaneStride = 115200 : i32, dstWidth = 168 : i32, dstStride = 192 : i32, dstPlaneStride = 10176 : i32>)

    VPUMI40XX.OpRanges
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
IE.ExecutorResource 1 of @DMA_NN
IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @nndma_3d_to_1d inputsInfo : {
    DataInfo "input" : tensor<1x84x90x53xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x84x90x53xf16, {order = #NHWC}>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x84x90x53xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @nndma_3d_to_1d() {
    %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <884868> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x84x90x53xf16, {order = #NHWC}, @DDR>
    %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>)
        inputs(%1 : memref<1x84x90x53xf16, {order = #NHWC, strides = [10368000, 1, 57600, 180]}, @DDR>)
        outputs(%2 : memref<1x84x90x53xf16, {order = #NHWC}, @DDR>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK:   descriptor(<numPlanes = 90 : i32, len = 8904 : i32, srcWidth = 168 : i32, srcStride = 360 : i32, srcPlaneStride = 115200 : i32, dstWidth = 8904 : i32, dstStride = 8904 : i32, dstPlaneStride = 8904 : i32>)

    VPUMI40XX.OpRanges
  }
}
