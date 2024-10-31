//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --create-elf-symbol-table %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

IE.CNNNetwork entryPoint : @oneDma inputsInfo : {
  DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x2x3x4xf16>
}

func.func @oneDma() {
  ELF.Main @ELFMain {
    ELF.CreateSection @dsec1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
    }
    ELF.CreateSection @dsec2 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
    }
    ELF.CreateSection @dsec3 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
    }
    ELF.CreateLogicalSection @lsec1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
    }
    ELF.CreateLogicalSection @lsec2 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
    }
    ELF.CreateLogicalSection @lsec3 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
    }
  }
  return
}

//CHECK: ELF.Main

//CHECK: ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
//CHECK-NEXT: ELF.Symbol @elfsym.dsec1 of(@dsec1) type(<STT_SECTION>) size(0) value(0)
//CHECK-NEXT: ELF.Symbol @elfsym.dsec2 of(@dsec2) type(<STT_SECTION>) size(0) value(0)
//CHECK-NEXT: ELF.Symbol @elfsym.dsec3 of(@dsec3) type(<STT_SECTION>) size(0) value(0)
//CHECK-NEXT: ELF.Symbol @elfsym.lsec1 of(@lsec1) type(<STT_SECTION>) size(0) value(0)
//CHECK-NEXT: ELF.Symbol @elfsym.lsec2 of(@lsec2) type(<STT_SECTION>) size(0) value(0)
//CHECK-NEXT: ELF.Symbol @elfsym.lsec3 of(@lsec3) type(<STT_SECTION>) size(0) value(0)
