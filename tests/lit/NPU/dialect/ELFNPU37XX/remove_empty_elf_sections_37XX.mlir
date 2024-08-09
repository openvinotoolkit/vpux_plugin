//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --remove-empty-elf-sections  %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module @Test  {

// CHECK-LABEL: @oneDma
  func.func @oneDma(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %1 = VPUMI37XX.MappedInference dmas(%0 : !VPURegMapped.Index<0:0:0>) dmaCount([1, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
    %2 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELFNPU37XX.Section  {
      ELFNPU37XX.PutOpInSection %0 : !VPURegMapped.Index<0:0:0>
    }
    %3 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section  {
    }
    %4 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELFNPU37XX.Section  {
    }
    %5 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELFNPU37XX.Section  {
    }
    %6 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELFNPU37XX.Section  {
    }
    %7 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELFNPU37XX.Section  {
    }
    %8 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELFNPU37XX.Section  {
    }
    %9 = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section  {
      ELFNPU37XX.PutOpInSection %1 : !VPURegMapped.Index<0:0:0>
    }
    %10 = ELFNPU37XX.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELFNPU37XX.Section  {
      %49 = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>
    }
    %11 = ELFNPU37XX.Symbol %2 name("sym_dmaSection0") : !ELFNPU37XX.Section
    %12 = ELFNPU37XX.Symbol %3 name("sym_barrierSection") : !ELFNPU37XX.Section
    %13 = ELFNPU37XX.Symbol %7 name("sym_actKernelRangeSection") : !ELFNPU37XX.Section
    %14 = ELFNPU37XX.Symbol %8 name("sym_actKernelInvo") : !ELFNPU37XX.Section
    %15 = ELFNPU37XX.Symbol %4 name("sym_kernelTextSection") : !ELFNPU37XX.Section
    %16 = ELFNPU37XX.Symbol %5 name("sym_kernelDataSection") : !ELFNPU37XX.Section
    %17 = ELFNPU37XX.Symbol %6 name("sym_kernelParamsSection") : !ELFNPU37XX.Section
    %18 = ELFNPU37XX.Symbol %arg0 name("inputCNN") size(2000) : memref<1x1x1x1000xf16>
    %19 = ELFNPU37XX.Symbol %arg1 name("outputCNN") size(2000) : memref<1x1x1x1000xf16>
    %20 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section  {
      ELFNPU37XX.PutOpInSection %18 : !ELFNPU37XX.Symbol
    }
    %21 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section  {
      ELFNPU37XX.PutOpInSection %19 : !ELFNPU37XX.Symbol
    }
    %22 = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO"} -> !ELFNPU37XX.Section  {
    }
    %23 = ELFNPU37XX.Symbol %22 name("sym_bufferSection") : !ELFNPU37XX.Section
    %24 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELFNPU37XX.Section  {
      ELFNPU37XX.PutOpInSection %23 : !ELFNPU37XX.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %25 = ELFNPU37XX.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    %c1_i8 = arith.constant 1 : i8
    %26 = ELFNPU37XX.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    %c2_i8 = arith.constant 2 : i8
    %27 = ELFNPU37XX.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    %c3_i8 = arith.constant 3 : i8
    %28 = ELFNPU37XX.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    %c4_i8 = arith.constant 4 : i8
    %29 = ELFNPU37XX.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    %c5_i8 = arith.constant 5 : i8
    %30 = ELFNPU37XX.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    %c6_i8 = arith.constant 6 : i8
    %31 = ELFNPU37XX.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
    %32 = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELFNPU37XX.Section  {
      ELFNPU37XX.PutOpInSection %25 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %26 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %27 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %28 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %29 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %30 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %31 : !ELFNPU37XX.Symbol
    }
    %33 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section  {
      ELFNPU37XX.PutOpInSection %11 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %12 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %15 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %16 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %17 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %13 : !ELFNPU37XX.Symbol
      ELFNPU37XX.PutOpInSection %14 : !ELFNPU37XX.Symbol
      %49 = ELFNPU37XX.Symbol %1 name("MappedInference_entry") type(<VPU_STT_ENTRY>) : !VPURegMapped.Index<0:0:0>
    }
    %34 = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%20) targetSection(%2) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section  {
      ELFNPU37XX.Reloc baseOp(%0 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %18 0
    }
    %35 = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%21) targetSection(%2) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section  {
      ELFNPU37XX.Reloc baseOp(%0 : !VPURegMapped.Index<0:0:0>) offset(24) <R_VPU_64> %19 0
    }
    %36 = ELFNPU37XX.CreateRelocationSection secName(".rlt.dmaIO_DDR") sourceSymbolTableSection(%24) targetSection(%2) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section  {
    }
    %37 = ELFNPU37XX.CreateRelocationSection secName(".rlt.dmaIO_CMX") sourceSymbolTableSection(%32) targetSection(%2) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section  {
    }
    %38 = ELFNPU37XX.CreateRelocationSection secName(".rlt.KernelParams") sourceSymbolTableSection(%33) targetSection(%6) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section  {
    }
    %39 = ELFNPU37XX.CreateRelocationSection secName(".rlt.KernelParamsIO_DDR") sourceSymbolTableSection(%24) targetSection(%6) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section  {
    }
    %40 = ELFNPU37XX.CreateRelocationSection secName(".rlt.KernelParamsIO_CMX") sourceSymbolTableSection(%32) targetSection(%6) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section  {
    }
    %41 = ELFNPU37XX.CreateRelocationSection secName(".rlt.ActKernelRange") sourceSymbolTableSection(%33) targetSection(%7) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section  {
    }
    %42 = ELFNPU37XX.CreateRelocationSection secName(".rlt.ActKernelInvo") sourceSymbolTableSection(%33) targetSection(%8) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section  {
    }
    %43 = VPURT.DeclareBuffer <DDR> <64> -> memref<262144xi32, @DDR>
    %44 = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actKernelRtConfigSec"} -> !ELFNPU37XX.Section  {
      ELFNPU37XX.PutOpInSection %43 : memref<262144xi32, @DDR>
    }
    %45 = ELFNPU37XX.Symbol %44 name("sym_actKernelRtConfigsSec") : !ELFNPU37XX.Section
    %46 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELFNPU37XX.Section  {
      ELFNPU37XX.PutOpInSection %45 : !ELFNPU37XX.Symbol
    }
    %47 = ELFNPU37XX.CreateRelocationSection secName(".rlt.MI_AKRtConfig") sourceSymbolTableSection(%46) targetSection(%9) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section  {
      ELFNPU37XX.Reloc baseOp(%1 : !VPURegMapped.Index<0:0:0>) offset(828) <R_VPU_32> %45 0
    }
    %48 = ELFNPU37XX.CreateRelocationSection secName(".rlt.MappedInference") sourceSymbolTableSection(%33) targetSection(%9) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section  {
      ELFNPU37XX.Reloc baseOp(%1 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_64> %11 0
    }
    return %arg1 : memref<1x1x1x1000xf16>
  }

    // CHECK:      [[DMA:%.+]] =  VPUMI37XX.NNDMA
    // CHECK:      [[MPI:%.+]] =  VPUMI37XX.MappedInference
    // CHECK:      [[DMASEC:%.+]] = ELFNPU37XX.CreateSection
    // CHECK-NOT:  ELFNPU37XX.CreateSection
    // CHECK-NOT:  ELFNPU37XX.CreateSection
    // CHECK-NOT:  ELFNPU37XX.CreateSection
    // CHECK-NOT:  ELFNPU37XX.CreateSection
    // CHECK-NOT:  ELFNPU37XX.CreateSection
    // CHECK-NOT:  ELFNPU37XX.CreateSection
    // CHECK:      ELFNPU37XX.CreateSection
    // CHECK:      ELFNPU37XX.CreateMetadataSection
    // CHECK:      ELFNPU37XX.Symbol [[DMASEC]]
    // CHECK-NOT:  ELFNPU37XX.Symbol
    // CHECK-NOT:  ELFNPU37XX.Symbol
    // CHECK-NOT:  ELFNPU37XX.Symbol
    // CHECK-NOT:  ELFNPU37XX.Symbol
    // CHECK-NOT:  ELFNPU37XX.Symbol
    // CHECK-NOT:  ELFNPU37XX.Symbol
    // CHECK-NEXT:      [[SYMARG0:%.+]] = ELFNPU37XX.Symbol [[ARG0:%.+]]
    // CHECK-NEXT:      [[SYMARG1:%.+]] = ELFNPU37XX.Symbol [[ARG1:%.+]]
    // CHECK-NEXT:      ELFNPU37XX.CreateSymbolTableSection
    // CHECK-NEXT: ELFNPU37XX.PutOpInSection [[SYMARG0]] : !ELFNPU37XX.Symbol
    // CHECK:      ELFNPU37XX.CreateSymbolTableSection
    // CHECK-NEXT: ELFNPU37XX.PutOpInSection [[SYMARG1]] : !ELFNPU37XX.Symbol
    // CHECK-NOT:  ELFNPU37XX.CreateLogicalSection
    // CHECK-NOT:  ELFNPU37XX.Symbol
    // CHECK-NOT:  ELFNPU37XX.CreateSymbolTableSection
    // CHECK:      arith.constant 0 : i8
    // CHECK:      ELFNPU37XX.CreateRelocationSection
    // CHECK:      ELFNPU37XX.CreateRelocationSection
    // CHECK-NOT:  ELFNPU37XX.CreateRelocationSection
    // CHECK-NOT:  ELFNPU37XX.CreateRelocationSection
    // CHECK-NOT:  ELFNPU37XX.CreateRelocationSection
    // CHECK-NOT:  ELFNPU37XX.CreateRelocationSection
    // CHECK-NOT:  ELFNPU37XX.CreateRelocationSection
    // CHECK-NOT:  ELFNPU37XX.CreateRelocationSection
    // CHECK-NOT:  ELFNPU37XX.CreateRelocationSection
    // CHECK:      VPURT.DeclareBuffer

 }

// -----
