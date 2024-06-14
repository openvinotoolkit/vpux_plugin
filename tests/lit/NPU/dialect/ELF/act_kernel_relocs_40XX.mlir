//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUASM-to-NPUReg40XX-relocs --create-elf-relocations %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

module @Model20 {

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Sigmoid(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Input" : tensor<1x50x1x1xf16>
  } outputsInfo : {
    DataInfo "Sigmoid_225" : tensor<1x50x1x1xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @Input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x50x1x1xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @Sigmoid_225_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x50x1x1xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }

func.func @main() {
  ELF.Main @ELFMain {
    ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
      VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelRange> {elfMemOffsetAttrKey = 51200 : ui64}
      VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelInvocation> {elfMemOffsetAttrKey = 53760 : ui64}
    }
    ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") {
      VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "CMX_NN"[0] <120> : memref<1x10x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "CMX_NN"[0] <240> : memref<1x10x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> :  swizzling(0)>
    }
    VPUASM.DeclareKernelEntry @DeclareKernelEntry_0_0 : "activation_sigmoid"

    ELF.CreateSection @shave.text aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
      VPUASM.DeclareKernelText @DeclareKernelText_0_0 {elfMemOffsetAttrKey = 0 : ui64} : "activation_sigmoid"
    }
    ELF.CreateSection @shave.data aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
      VPUASM.DeclareKernelData @DeclareKernelArgs_0_0 {elfMemOffsetAttrKey = 0 : ui64} : "activation_sigmoid"
    }
    ELF.CreateSection @shave.params aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
      VPUASM.KernelParams @KernelParams_0_0 inputs([@buffer.CMX_NN.0::@DeclareBuffer5]) outputs([@buffer.CMX_NN.0::@DeclareBuffer6]) kernel_type("activation_sigmoid") kernel_params(dense<1> : vector<72xui8>) {elfMemOffsetAttrKey = 0 : ui64}
    }
    ELF.CreateSection @task.shave.range.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
      VPUASM.ActKernelRange @ActKernelRange_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelRange_0_0_0) kernelTaskType(@COMPUTE) calls @shave.text::@DeclareKernelText_0_0 : @DeclareKernelEntry_0_0 {elfMemOffsetAttrKey = 0 : ui64}
    }
    ELF.CreateSection @task.shave.invocation.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
      VPUASM.ActKernelInvocation @ActKernelInvocation_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelInvocation_0_0_0) -> @program.metadata.cmx::@DeclareTaskBuffer_ActKernelRange_0_0_0(kernel_data : @shave.data::@DeclareKernelArgs_0_0, kernel_params : @shave.params::@KernelParams_0_0) waits([0 : ui8]) updates([1 : ui8]) tile(0) start_after(2) clean_after(1) range_index(0) {elfMemOffsetAttrKey = 0 : ui64}
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.shave.text of(@shave.text) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.shave.data of(@shave.data) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.shave.params of(@shave.params) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.task.shave.range.0.0 of(@task.shave.range.0.0) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.task.shave.invocation.0.0 of(@task.shave.invocation.0.0) type(<STT_SECTION>) size(0) value(0)
    }

  }
  return
}

    // CHECK-DAG:  ELF.CreateRelocationSection @rela.shave.params.symtab
    // CHECK-DAG:  ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.buffer.CMX_NN.0) relocType(<R_VPU_32_BIT_OR_B21_B26_UNSET>) addend(120)
    // CHECK-DAG:  ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.buffer.CMX_NN.0) relocType(<R_VPU_32_BIT_OR_B21_B26_UNSET>) addend(240)
    // CHECK-DAG:  ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.shave.params) relocType(<R_VPU_32>) {{.*}}
    // CHECK-DAG:  ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.shave.params) relocType(<R_VPU_32>) {{.*}}
    // CHECK-DAG:  ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.shave.params) relocType(<R_VPU_32>) {{.*}}
    // CHECK-DAG:  ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.shave.params) relocType(<R_VPU_32>) {{.*}}

    // CHECK-DAG:  ELF.CreateRelocationSection @rela.task.shave.range.0.0.symtab
    // CHECK-DAG:  ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.shave.text) relocType(<R_VPU_64>) addend(0)

    // CHECK-DAG:  ELF.CreateRelocationSection @rela.task.shave.invocation.0.0.symtab
    // CHECK-DAG:  ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.shave.data) relocType(<R_VPU_64>) addend(0)
    // CHECK-DAG:  ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.shave.params) relocType(<R_VPU_64>) addend(0)


}
