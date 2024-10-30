//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUIPDPU-to-NPUReg40XX="dpu-dry-run=stub" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @DPUDryRunTest {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input_0" : tensor<1x16x16x16xf16>
        DataInfo "input_1" : tensor<16x1x1x1xi64>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x64x64xf16>
    }
    func.func @main() {
        ELF.Main @ELFMain {
            ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
                VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <196736> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_WeightTable !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUVariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }
            ELF.CreateSection @text.invariants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
                VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0, input = @builtin.data.nncmx0::@DeclareBuffer_ActIn, output = @builtin.data.nncmx0::@DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
                    DPUCfg : {
                    ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                        %weight_table: memref<16x1x1x1xi64, #NHWC, [@CMX_NN, 0]>,
                        %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                        %sparse_out: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>):
                    VPUIPDPU.IDUCfg {
                        VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                    }
                    VPUIPDPU.PPECfg {
                        VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
                    }
                    VPUIPDPU.ODUCfg {
                        VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                        VPUIPDPU.ODUSparsity %sparse_out: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>
                        VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                    }
                }
            }

            ELF.CreateSection @text.variants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
                VPUIPDPU.DPUVariant @DPUVariant11 invariant(@builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @builtin.tasks.DPUVariant0::@DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, workload_id = 241 : i64}
                DPUCfg: {
                    VPUIPDPU.IDUPadding pad_count(<left = 4, right = 6, top = 3, bottom = 5>)
                    VPUIPDPU.IDUWorkloadSet start_x(3) start_y(4) start_z(5) size_x(32) size_y(33) size_z(34)
                    VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_8_8)
                    VPUIPDPU.IDUBinaryConfig
                    VPUIPDPU.IDUConvContinue
                    VPUIPDPU.IDUSEDense
                    VPUIPDPU.IDUActSwizzle swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.IDUWeightSwizzle wt_swizzle_key(SWIZZLE_KEY_2)
                    VPUIPDPU.BarrierCfg waits([1 : ui8, 2 : ui8]) updates([3 : ui8, 4 : ui8, 5 : ui8]) start_after(0) clean_after(0)

                    VPUIPDPU.ODUHaloCfg {
                        VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(15) end_coord_y(15) activations_offset(10) sparsity_offset(10) target_width(32) cast_to_tile("DPU_TILE_1|DPU_TILE_2")
                        VPUIPDPU.ODUHaloRegion begin_coord_x(4) begin_coord_y(5) end_coord_x(15) end_coord_y(16) activations_offset(9) target_width(64) cast_to_tile("DPU_TILE_1|DPU_TILE_3")
                        VPUIPDPU.ODUHaloRegion begin_coord_x(9) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(400) target_width(15424) cast_to_tile("DPU_TILE_0|DPU_TILE_1|DPU_TILE_2|DPU_TILE_3|DPU_TILE_4|DPU_TILE_5")
                        VPUIPDPU.ODUHaloRegion begin_coord_x(9) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(9) target_width(15488) cast_to_tile("DPU_TILE_0|DPU_TILE_5")
                        VPUIPDPU.ODUHaloRegion begin_coord_x(9) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(9) target_width(3328) cast_to_tile("DPU_TILE_2")
                    }                    
                    VPUIPDPU.ODUOutSubtensor begin_coord_x(1) begin_coord_y(32) begin_coord_z(64) end_coord_x(63) end_coord_y(63) end_coord_z(15)

                    VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) {isFirstVariant}
                }
            }
            ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
                ELF.Symbol @elfsym.builtin.data.nncmx0 of(@builtin.data.nncmx0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.builtin.tasks.DPUVariant0 of(@builtin.tasks.DPUVariant0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.builtin.tasks.DPUInvariant0 of(@builtin.tasks.DPUInvariant0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.text.invariants of(@text.invariants) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.text.variants of(@text.variants) type(<STT_SECTION>) size(0) value(0)
            }
        }

        return
    }
}

// CHECK-NOT:  VPUIPDPU.DPUInvariant
// CHECK:      NPUReg40XX.DPUInvariant
// CHECK-DAG:      UINT tensor_size_x at 0 size 14 = 1
// CHECK-DAG:      UINT tensor_size_y at 14 size 14 = 1
// CHECK-DAG:      UINT tensor_size_z at 0 size 14 = 0x10
// CHECK-DAG:      UINT kernel_y at 5 size 4 = 1
// CHECK-DAG:      UINT kernel_x at 9 size 4 = 1
// CHECK-DAG:      UINT zm_input at 11 size 1 = 1
// CHECK-DAG:      UINT workload_operation at 14 size 2 = 0
// CHECK-DAG:      UINT elop_wload at 0 size 1 = 1
// CHECK-DAG:      UINT elop_wload_type at 3 size 1 = 1
// CHECK-DAG:      UINT nthw at 25 size 2 = 1
// CHECK-DAG:      UINT te_dim_y at 0 size 13 = 0
// CHECK-DAG:      UINT te_dim_z at 13 size 13 = 0xF
// CHECK-DAG:      UINT te_dim_x at 0 size 13 = 0

// CHECK-NOT:  VPUIPDPU.DPUVariant
// CHECK:      NPUReg40XX.DPUVariant
// CHECK-DAG:  UINT workload_size_x at 0 size 14 = 1
// CHECK-DAG:  UINT workload_size_y at 14 size 14 = 1
// CHECK-DAG:  UINT workload_size_z at 0 size 14 = 0x10
// CHECK-DAG:  UINT pad_count_up at 14 size 3 = 0
// CHECK-DAG:  UINT pad_count_left at 17 size 3 = 0
// CHECK-DAG:  UINT pad_count_down at 20 size 3 = 0
// CHECK-DAG:  UINT pad_count_right at 23 size 3 = 0
// CHECK-DAG:  UINT workload_start_x at 0 size 14 = 0
// CHECK-DAG:  UINT workload_start_y at 14 size 14 = 0
// CHECK-DAG:  UINT workload_start_z at 0 size 14 = 0
// CHECK-DAG:  weight_size offset 180 size 32 = UINT 0x10
// CHECK-DAG:  weight_num offset 184 size 32 = UINT 0x10
// CHECK-DAG:  UINT te_beg_y at 0 size 13 = 0
// CHECK-DAG:  UINT te_beg_z at 13 size 13 = 0
// CHECK-DAG:  UINT te_beg_x at 0 size 13 = 0
// CHECK-DAG:  UINT te_end_y at 0 size 13 = 0
// CHECK-DAG:  UINT te_end_z at 13 size 13 = 0xF
// CHECK-DAG:  UINT te_end_x at 0 size 13 = 0
