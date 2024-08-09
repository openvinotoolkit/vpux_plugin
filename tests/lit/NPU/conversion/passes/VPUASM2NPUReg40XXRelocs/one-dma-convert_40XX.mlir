//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUASM-to-NPUReg40XX-relocs %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @OneDMAWithoutAttributes {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf32>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf32, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf32, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 96 : i32, srcWidth = 96 : i32, srcStride = 96 : i32, srcPlaneStride = 0 : i32, dstWidth = 48 : i32, dstStride = 48 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_conversion_cfg at 39 size 3 = 3
        // CHECK:  UINT dma_width_src at 0 size 32 = 0x60
        // CHECK:  UINT dma_width_dst at 32 size 32 = 0x30
        // CHECK:  UINT dma_src at 0 size 48 = 0
        // CHECK:  UINT dma_dst at 0 size 48 = 0
      }
    }
    return
  }
}

// -----

module @OneDMAWithoutAttributes {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf32>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xbf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf32, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xbf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf32, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xbf16, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 96 : i32, srcWidth = 96 : i32, srcStride = 96 : i32, srcPlaneStride = 0 : i32, dstWidth = 48 : i32, dstStride = 48 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_conversion_cfg at 39 size 3 = 4
        // CHECK:  UINT dma_width_src at 0 size 32 = 0x60
        // CHECK:  UINT dma_width_dst at 32 size 32 = 0x30
        // CHECK:  UINT dma_src at 0 size 48 = 0
        // CHECK:  UINT dma_dst at 0 size 48 = 0
      }
    }
    return
  }
}
