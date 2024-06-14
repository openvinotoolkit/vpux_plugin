//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUIPDPU-to-NPUReg40XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

module @TestDMLRelocCONV attributes {VPU.directML} {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input_0" : tensor<1x16x12x62xf16>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x48x10x60xf16>
    }
    func.func @main() {
        ELF.Main @ELFMain {
            ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
                VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <63552> : memref<1x16x12x62xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <5952> : memref<1x48x10x60xf16, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_WeightTab !VPUASM.Buffer< "CMX_NN"[0] <576> : memref<48x1x1x4xsi32, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_Weight !VPUASM.Buffer< "CMX_NN"[0] <1344> : memref<48x16x3x3xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_WeightsSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <4416> : memref<48x1x1x256xi1, [@CMX_NN, 0]> :  swizzling(0)>
            }

            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }

            ELF.CreateSection @text.invariants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
                VPUIPDPU.DPUInvariant @DPUInvariant_0 { elfMemOffsetAttrKey = 352 : ui64,
                        task_index = !VPURegMapped.Index<0:0:0>,
                        task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0,
                        input = @builtin.data.nncmx0::@DeclareBuffer_ActIn,
                        output = @builtin.data.nncmx0::@DeclareBuffer_ActOut,
                        nce_task_type = #VPUIP.nce_task_type<CONV>,
                        weight_table = @builtin.data.nncmx0::@DeclareBuffer_WeightTab,
                        weights = @builtin.data.nncmx0::@DeclareBuffer_Weight,
                        weights_sparsity_map = @buffer.CMX_NN.0::@DeclareBuffer96
                        }
                    DPUCfg : {
                     ^bb0(
                        %arg0: memref<1x16x12x62xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>,
                        %arg1: memref<48x1x1x4xsi32, [@CMX_NN, 0]>,
                        %arg2: memref<48x16x3x3xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>,
                        order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>,
                        %arg3: memref<48x1x1x256xi1, [@CMX_NN, 0]>,
                        %arg4: memref<1x48x10x60xf16, [@CMX_NN, 0]>):
                    VPUIPDPU.IDUCfg {
                        VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x16x12x62xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>)
                        VPUIPDPU.IDUWeights wmode(f16) {wt_sparse}
                        VPUIPDPU.IDUKernel kernel_x(3) kernel_y(3)
                        VPUIPDPU.IDUStride stride_x(1) stride_y(1)
                        VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
                        }
                    VPUIPDPU.PPECfg {
                        VPUIPDPU.PPEFpBiasAdd %arg1 : memref<48x1x1x4xsi32, [@CMX_NN, 0]>
                        VPUIPDPU.PPEFpScalePreluMult %arg1 : memref<48x1x1x4xsi32, [@CMX_NN, 0]>
                        VPUIPDPU.PPEFpAddMultBypass bypass_mode(OFF)
                        VPUIPDPU.PPEFpConvert convert_mode(FP16)
                        VPUIPDPU.PPEIntBiasAdd bias_static(0)
                        VPUIPDPU.PPEIntScaleMult scale_static(1)
                        VPUIPDPU.PPEIntScaleShift shift_static(0)
                        VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
                        VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
                        VPUIPDPU.PPEIntRound round_mode(RNE)
                        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
                        VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647)
                        VPUIPDPU.PPEIntConvert convert_mode(NONE)
                    }
                    VPUIPDPU.ODUCfg {
                        VPUIPDPU.ODUOutTensorSize dim_x(60) dim_y(10) dim_z(48)
                        VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
                        VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_XYZ)
                        VPUIPDPU.ODUOutActivations out_activations(%arg4 : memref<1x48x10x60xf16, [@CMX_NN, 0]>)
                        VPUIPDPU.ODUMemoryMode mem_mode(MODE_SUPERDENSE)
                    }
                    // VPUIPDPU.BarrierCfg waits([4 : ui8]) updates([5 : ui8]) start_after(6) clean_after(5)
                    // VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:1>) variantCount(1)
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
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) addend({{.*}})
//      act_offset[1]:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) addend({{.*}})
//      act_offset[2]:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) addend({{.*}})
//      act_offset[3]:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) addend({{.*}})
// Weights Relocs (relocation generated because nce_task_type is CONV and DML getweights not NULL)
//      wt_offset:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) addend([[NOTZERO:[1-9]+]])
// CHECK-NOT:       [[NOTZERO]] 0
// Output Relocs:
//      tensor_output:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.data.nncmx0) relocType(<R_VPU_LO_21>) addend({{.*}})
// relocation for preemtion workaround:
//      tensor_mode.pad_value:
// CHECK:           ELF.Reloc offset({{.*}}) sourceSym(@symtab::@elfsym.builtin.tasks.DPUInvariant0) relocType(<R_VPU_16_LSB_17_RSHIFT_5_LSHIFT_16>) addend({{.*}})
