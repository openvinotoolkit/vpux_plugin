//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-network-metadata %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

IE.CNNNetwork entryPoint : @oneDma inputsInfo : {
  DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x2x3x4xf16>
}
func.func @oneDma() {
  ELF.Main @ELFMain {
    ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
      VPUASM.MappedInference @MappedInference : dmaCount([[0, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(0)
    }
  }
  return
}

//CHECK: ELF.CreateMetadataSection @MetadataSection aligned(8) secFlags("SHF_NONE")
//CHECK-NEXT: VPUASM.NetworkMetadata @NetworkMetadata
