//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUIPDPU-to-NPUReg40XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

module @TestDMLRelocMAXPOOL attributes {VPU.directML} {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input_0" : tensor<1x16x16x16xf16>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x64x64xf16>
    }
    func.func @main() {
        ELF.Main @ELFMain {
            ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
                VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <196736> : memref<1x16x16x16xf16,  [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16,  [@CMX_NN, 0]> :  swizzling(0)>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }
            ELF.CreateSection @text.invariants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
                VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0, input = @builtin.data.nncmx0::@DeclareBuffer_ActIn, output = @builtin.data.nncmx0::@DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                    DPUCfg : {
                    ^bb0(%act_in: memref<1x16x16x16xf16,  [@CMX_NN, 0]>,
                        %act_out: memref<1x16x64x64xf16,  [@CMX_NN, 0]>):
                    VPUIPDPU.IDUCfg {
                        VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16,  [@CMX_NN, 0]>)
                    }
                    VPUIPDPU.PPECfg {
                        VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
                    }
                    VPUIPDPU.ODUCfg {
                        VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                        VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16,  [@CMX_NN, 0]>)
                    }
                }
            }
            ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
                ELF.Symbol @elfsym.builtin.data.nncmx0 of(@builtin.data.nncmx0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.builtin.tasks.DPUInvariant0 of(@builtin.tasks.DPUInvariant0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.text.invariants of(@text.invariants) type(<STT_SECTION>) size(0) value(0)
            }
        }
        return
    }
}

// CHECK:       ELF.CreateRelocationSection @rela.text.invariants.symtab target(@text.invariants) symtab(@symtab) secFlags("SHF_NONE") {
// Input Relocs:
//      tensor_start:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) {{.*}}
//      act_offset[1]:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) {{.*}}
//      act_offset[2]:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) {{.*}}
//      act_offset[3]:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) {{.*}}
// Weights Relocs (relocation not generated because nce_task_type is MAXPOOL but DML getweights is NULL)
//      wt_offset:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) addend(0)
// Output Relocs:
//      tensor_output:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) {{.*}}
// relocation for preemtion workaround:
//      tensor_mode.pad_value:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.tasks.DPUInvariant0) relocType(<R_VPU_16_LSB_17_RSHIFT_5_LSHIFT_16>) {{.*}}
