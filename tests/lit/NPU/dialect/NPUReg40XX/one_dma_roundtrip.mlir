//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @OneDMAWithoutAttributes {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        "NPUReg40XX.NNDMA"() <{dma_descriptor = #NPUReg40XX.DMARegister<
          DMARegister {
            dma_watermark offset 0 size 8 {
              UINT dma_watermark at 0 size 1 = 0
            },
            dma_link_address offset 0 size 64 {
              UINT dma_link_address at 0 size 48 = 0
            },
            dma_lra offset 0 size 64 {
              UINT dma_lra at 63 size 1 = 0
            },
            dma_lba_addr offset 8 size 32 = UINT 0,
            dma_src_aub offset 12 size 8 = UINT 0,
            dma_dst_aub offset 14 size 8 = UINT 0,
            dma_cfg_fields offset 16 size 64 {
              UINT dma_cfg_fields_num_dim at 0 size 3 = 0,
              UINT dma_cfg_fields_int_en at 3 size 1 = 0,
              UINT dma_cfg_fields_int_id at 4 size 8 = 0,
              UINT dma_cfg_fields_src_burst_length at 12 size 4 = 0xF,
              UINT dma_cfg_fields_dst_burst_length at 16 size 4 = 0xF,
              UINT dma_cfg_fields_arb_qos at 20 size 8 = 0xFF,
              UINT dma_cfg_fields_ord at 28 size 1 = 1,
              UINT dma_cfg_fields_barrier_en at 29 size 1 = 1,
              UINT dma_cfg_fields_memset_en at 30 size 1 = 0,
              UINT dma_cfg_fields_atp_en at 31 size 1 = 1,
              UINT dma_cfg_fields_watermark_en at 32 size 1 = 0,
              UINT dma_cfg_fields_rwf_en at 33 size 1 = 0,
              UINT dma_cfg_fields_rws_en at 34 size 1 = 0,
              UINT dma_cfg_fields_src_list_cfg at 35 size 2 = 0,
              UINT dma_cfg_fields_dst_list_cfg at 37 size 2 = 0,
              UINT dma_cfg_fields_conversion_cfg at 39 size 3 = 0,
              UINT dma_cfg_fields_acceleration_cfg at 42 size 2 = 0,
              UINT dma_cfg_fields_tile4_cfg at 44 size 2 = 0,
              UINT dma_cfg_fields_axi_user_bits_cfg at 46 size 2 = 0,
              UINT dma_cfg_fields_hwp_id_en at 48 size 1 = 1,
              UINT dma_cfg_fields_hwp_id at 49 size 12 = 0,
              UINT dma_cfg_fields_reserved at 61 size 3 = 0
            },
            dma_remote_width_fetch offset 24 size 32 = UINT 0,
            dma_width offset 24 size 64 {
              UINT dma_width_src at 0 size 32 = 0x30,
              UINT dma_width_dst at 32 size 32 = 0x30
            },
            dma_acc_info_compress offset 32 size 64 {
              UINT dma_acc_info_compress_dtype at 0 size 2 = 0,
              UINT dma_acc_info_compress_reserved1 at 2 size 1 = 0,
              UINT dma_acc_info_compress_sparse at 3 size 1 = 0,
              UINT dma_acc_info_compress_bitc_en at 4 size 1 = 0,
              UINT dma_acc_info_compress_z at 5 size 10 = 0,
              UINT dma_acc_info_compress_bitmap_buf_sz at 15 size 19 = 0,
              UINT dma_acc_info_compress_reserved2 at 34 size 3 = 0,
              UINT dma_acc_info_compress_bitmap_base_addr at 37 size 27 = 0
            },
            dma_acc_info_decompress offset 32 size 64 {
              UINT dma_acc_info_decompress_dtype at 0 size 2 = 0,
              UINT dma_acc_info_decompress_reserved1 at 2 size 1 = 0,
              UINT dma_acc_info_decompress_sparse at 3 size 1 = 0,
              UINT dma_acc_info_decompress_bitc_en at 4 size 1 = 0,
              UINT dma_acc_info_decompress_z at 5 size 10 = 0,
              UINT dma_acc_info_decompress_reserved2 at 15 size 22 = 0,
              UINT dma_acc_info_decompress_bitmap_base_addr at 37 size 27 = 0
            },
            dma_acc_info_w_prep offset 32 size 64 {
              UINT dma_acc_info_w_prep_dtype at 0 size 2 = 0,
              UINT dma_acc_info_w_prep_reserved1 at 2 size 1 = 0,
              UINT dma_acc_info_w_prep_sparse at 3 size 2 = 0,
              UINT dma_acc_info_w_prep_zeropoint at 5 size 8 = 0,
              UINT dma_acc_info_w_prep_ic at 13 size 14 = 0,
              UINT dma_acc_info_w_prep_filtersize at 27 size 7 = 0,
              UINT dma_acc_info_w_prep_reserved2 at 34 size 3 = 0,
              UINT dma_acc_info_w_prep_bitmap_base_addr at 37 size 27 = 0
            },
            dma_mset_data offset 32 size 32 = UINT 0,
            dma_src_addr offset 40 size 64 {
              UINT dma_src at 0 size 48 = 0,
              UINT dma_sra at 63 size 1 = 0
            },
            dma_dst_addr offset 48 size 64 {
              UINT dma_dst at 0 size 48 = 0,
              UINT dma_dra at 63 size 1 = 0
            },
            dma_sba_addr offset 56 size 32 = UINT 0,
            dma_dba_addr offset 60 size 32 = UINT 0,
            dma_barrier_prod_mask_lower offset 64 size 64 = UINT 0,
            dma_barrier_cons_mask_lower offset 72 size 64 = UINT 0,
            dma_barrier_prod_mask_upper offset 80 size 64 {
              UINT dma_barrier_prod_mask_upper at 0 size 32 = 0
            },
            dma_barrier_cons_mask_upper offset 88 size 64 {
              UINT dma_barrier_cons_mask_upper at 0 size 32 = 0
            },
            dma_list_size offset 96 size 64 {
              UINT dma_list_size_src at 0 size 32 = 0,
              UINT dma_list_size_dst at 32 size 32 = 0
            },
            dma_dim_size offset 96 size 64 {
              UINT dma_dim_size_1_src at 0 size 32 = 0,
              UINT dma_dim_size_1_dst at 32 size 32 = 0
            },
            dma_stride_src_1 offset 104 size 32 = UINT 0,
            dma_stride_dst_1 offset 108 size 32 = UINT 0,
            dma_dim_size_2 offset 112 size 64 {
              UINT dma_dim_size_2_src at 0 size 32 = 0,
              UINT dma_dim_size_2_dst at 32 size 32 = 0
            },
            dma_list_addr offset 112 size 64 {
              UINT dma_list_addr_src at 0 size 32 = 0,
              UINT dma_list_addr_dst at 32 size 32 = 0
            },
            dma_stride_src_2 offset 120 size 32 = UINT 0,
            dma_stride_dst_2 offset 124 size 32 = UINT 0,
            dma_remote_width_store offset 124 size 32 = UINT 0,
            dma_dim_size_src_3 offset 128 size 16 = UINT 0,
            dma_dim_size_src_4 offset 130 size 16 = UINT 0,
            dma_dim_size_dst_3 offset 132 size 16 = UINT 0,
            dma_dim_size_dst_4 offset 134 size 16 = UINT 0,
            dma_dim_size_src_5 offset 136 size 16 = UINT 0,
            dma_dim_size_dst_5 offset 140 size 16 = UINT 0,
            dma_stride_src_3 offset 144 size 32 = UINT 0,
            dma_stride_dst_3 offset 148 size 32 = UINT 0,
            dma_stride_src_4 offset 152 size 32 = UINT 0,
            dma_stride_dst_4 offset 156 size 32 = UINT 0,
            dma_stride_src_5 offset 160 size 32 = UINT 0,
            dma_stride_dst_5 offset 164 size 32 = UINT 0,
            dma_word_21_reserved offset 168 size 64 = UINT 0,
            dma_word_22_reserved offset 176 size 64 = UINT 0,
            dma_word_23_reserved offset 184 size 64 = UINT 0,
            dma_barriers_sched offset 192 size 64 {
              UINT start_after_ at 0 size 32 = 1,
              UINT clean_after_ at 32 size 32 = 2
            },
            dma_pad_24_0 offset 200 size 64 = UINT 0,
            dma_pad_24_1 offset 208 size 64 = UINT 0,
            dma_pad_24_2 offset 216 size 64 = UINT 0
          }
        >, input = @DeclareBuffer0, output_buffs = [@DeclareBuffer1], sym_name = "NNDMA_0_0_0"}> : () -> ()
      }
    }
    return
  }
}

// CHEKC: dma_descriptor = #NPUReg40XX.DMARegister<
// CHECK: DMARegister {
// CHECK:   dma_watermark offset 0 size 8 {
// CHECK:     UINT dma_watermark at 0 size 1 = 0
// CHECK:   },
// CHECK:   dma_link_address offset 0 size 64 {
// CHECK:     UINT dma_link_address at 0 size 48 = 0
// CHECK:   },
// CHECK:   dma_lra offset 0 size 64 {
// CHECK:     UINT dma_lra at 63 size 1 = 0
// CHECK:   },
// CHECK:   dma_lba_addr offset 8 size 32 = UINT 0,
// CHECK:   dma_src_aub offset 12 size 8 = UINT 0,
// CHECK:   dma_dst_aub offset 14 size 8 = UINT 0,
// CHECK:   dma_cfg_fields offset 16 size 64 {
// CHECK:     UINT dma_cfg_fields_num_dim at 0 size 3 = 0,
// CHECK:     UINT dma_cfg_fields_int_en at 3 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_int_id at 4 size 8 = 0,
// CHECK:     UINT dma_cfg_fields_src_burst_length at 12 size 4 = 0xF,
// CHECK:     UINT dma_cfg_fields_dst_burst_length at 16 size 4 = 0xF,
// CHECK:     UINT dma_cfg_fields_arb_qos at 20 size 8 = 0xFF,
// CHECK:     UINT dma_cfg_fields_ord at 28 size 1 = 1,
// CHECK:     UINT dma_cfg_fields_barrier_en at 29 size 1 = 1,
// CHECK:     UINT dma_cfg_fields_memset_en at 30 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_atp_en at 31 size 1 = 1,
// CHECK:     UINT dma_cfg_fields_watermark_en at 32 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_rwf_en at 33 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_rws_en at 34 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_src_list_cfg at 35 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_dst_list_cfg at 37 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_conversion_cfg at 39 size 3 = 0,
// CHECK:     UINT dma_cfg_fields_acceleration_cfg at 42 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_tile4_cfg at 44 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_axi_user_bits_cfg at 46 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_hwp_id_en at 48 size 1 = 1,
// CHECK:     UINT dma_cfg_fields_hwp_id at 49 size 12 = 0,
// CHECK:     UINT dma_cfg_fields_reserved at 61 size 3 = 0
// CHECK:   },
// CHECK:   dma_remote_width_fetch offset 24 size 32 = UINT 0x30,
// CHECK:   dma_width offset 24 size 64 {
// CHECK:     UINT dma_width_src at 0 size 32 = 0x30,
// CHECK:     UINT dma_width_dst at 32 size 32 = 0x30
// CHECK:   },
// CHECK:   dma_acc_info_compress offset 32 size 64 {
// CHECK:     UINT dma_acc_info_compress_dtype at 0 size 2 = 0,
// CHECK:     UINT dma_acc_info_compress_reserved1 at 2 size 1 = 0,
// CHECK:     UINT dma_acc_info_compress_sparse at 3 size 1 = 0,
// CHECK:     UINT dma_acc_info_compress_bitc_en at 4 size 1 = 0,
// CHECK:     UINT dma_acc_info_compress_z at 5 size 10 = 0,
// CHECK:     UINT dma_acc_info_compress_bitmap_buf_sz at 15 size 19 = 0,
// CHECK:     UINT dma_acc_info_compress_reserved2 at 34 size 3 = 0,
// CHECK:     UINT dma_acc_info_compress_bitmap_base_addr at 37 size 27 = 0
// CHECK:   },
// CHECK:   dma_acc_info_decompress offset 32 size 64 {
// CHECK:     UINT dma_acc_info_decompress_dtype at 0 size 2 = 0,
// CHECK:     UINT dma_acc_info_decompress_reserved1 at 2 size 1 = 0,
// CHECK:     UINT dma_acc_info_decompress_sparse at 3 size 1 = 0,
// CHECK:     UINT dma_acc_info_decompress_bitc_en at 4 size 1 = 0,
// CHECK:     UINT dma_acc_info_decompress_z at 5 size 10 = 0,
// CHECK:     UINT dma_acc_info_decompress_reserved2 at 15 size 22 = 0,
// CHECK:     UINT dma_acc_info_decompress_bitmap_base_addr at 37 size 27 = 0
// CHECK:   },
// CHECK:   dma_acc_info_w_prep offset 32 size 64 {
// CHECK:     UINT dma_acc_info_w_prep_dtype at 0 size 2 = 0,
// CHECK:     UINT dma_acc_info_w_prep_reserved1 at 2 size 1 = 0,
// CHECK:     UINT dma_acc_info_w_prep_sparse at 3 size 2 = 0,
// CHECK:     UINT dma_acc_info_w_prep_zeropoint at 5 size 8 = 0,
// CHECK:     UINT dma_acc_info_w_prep_ic at 13 size 14 = 0,
// CHECK:     UINT dma_acc_info_w_prep_filtersize at 27 size 7 = 0,
// CHECK:     UINT dma_acc_info_w_prep_reserved2 at 34 size 3 = 0,
// CHECK:     UINT dma_acc_info_w_prep_bitmap_base_addr at 37 size 27 = 0
// CHECK:   },
// CHECK:   dma_mset_data offset 32 size 32 = UINT 0,
// CHECK:   dma_src_addr offset 40 size 64 {
// CHECK:     UINT dma_src at 0 size 48 = 0,
// CHECK:     UINT dma_sra at 63 size 1 = 0
// CHECK:   },
// CHECK:   dma_dst_addr offset 48 size 64 {
// CHECK:     UINT dma_dst at 0 size 48 = 0,
// CHECK:     UINT dma_dra at 63 size 1 = 0
// CHECK:   },
// CHECK:   dma_sba_addr offset 56 size 32 = UINT 0,
// CHECK:   dma_dba_addr offset 60 size 32 = UINT 0,
// CHECK:   dma_barrier_prod_mask_lower offset 64 size 64 = UINT 0,
// CHECK:   dma_barrier_cons_mask_lower offset 72 size 64 = UINT 0,
// CHECK:   dma_barrier_prod_mask_upper offset 80 size 64 {
// CHECK:     UINT dma_barrier_prod_mask_upper at 0 size 32 = 0
// CHECK:   },
// CHECK:   dma_barrier_cons_mask_upper offset 88 size 64 {
// CHECK:     UINT dma_barrier_cons_mask_upper at 0 size 32 = 0
// CHECK:   },
// CHECK:   dma_list_size offset 96 size 64 {
// CHECK:     UINT dma_list_size_src at 0 size 32 = 0,
// CHECK:     UINT dma_list_size_dst at 32 size 32 = 0
// CHECK:   },
// CHECK:   dma_dim_size offset 96 size 64 {
// CHECK:     UINT dma_dim_size_1_src at 0 size 32 = 0,
// CHECK:     UINT dma_dim_size_1_dst at 32 size 32 = 0
// CHECK:   },
// CHECK:   dma_stride_src_1 offset 104 size 32 = UINT 0,
// CHECK:   dma_stride_dst_1 offset 108 size 32 = UINT 0,
// CHECK:   dma_dim_size_2 offset 112 size 64 {
// CHECK:     UINT dma_dim_size_2_src at 0 size 32 = 0,
// CHECK:     UINT dma_dim_size_2_dst at 32 size 32 = 0
// CHECK:   },
// CHECK:   dma_list_addr offset 112 size 64 {
// CHECK:     UINT dma_list_addr_src at 0 size 32 = 0,
// CHECK:     UINT dma_list_addr_dst at 32 size 32 = 0
// CHECK:   },
// CHECK:   dma_stride_src_2 offset 120 size 32 = UINT 0,
// CHECK:   dma_stride_dst_2 offset 124 size 32 = UINT 0,
// CHECK:   dma_remote_width_store offset 124 size 32 = UINT 0,
// CHECK:   dma_dim_size_src_3 offset 128 size 16 = UINT 0,
// CHECK:   dma_dim_size_src_4 offset 130 size 16 = UINT 0,
// CHECK:   dma_dim_size_dst_3 offset 132 size 16 = UINT 0,
// CHECK:   dma_dim_size_dst_4 offset 134 size 16 = UINT 0,
// CHECK:   dma_dim_size_src_5 offset 136 size 16 = UINT 0,
// CHECK:   dma_dim_size_dst_5 offset 140 size 16 = UINT 0,
// CHECK:   dma_stride_src_3 offset 144 size 32 = UINT 0,
// CHECK:   dma_stride_dst_3 offset 148 size 32 = UINT 0,
// CHECK:   dma_stride_src_4 offset 152 size 32 = UINT 0,
// CHECK:   dma_stride_dst_4 offset 156 size 32 = UINT 0,
// CHECK:   dma_stride_src_5 offset 160 size 32 = UINT 0,
// CHECK:   dma_stride_dst_5 offset 164 size 32 = UINT 0,
// CHECK:   dma_word_21_reserved offset 168 size 64 = UINT 0,
// CHECK:   dma_word_22_reserved offset 176 size 64 = UINT 0,
// CHECK:   dma_word_23_reserved offset 184 size 64 = UINT 0,
// CHECK:   dma_barriers_sched offset 192 size 64 {
// CHECK:     UINT start_after_ at 0 size 32 = 1,
// CHECK:     UINT clean_after_ at 32 size 32 = 2
// CHECK:   },
// CHECK:   dma_pad_24_0 offset 200 size 64 = UINT 0,
// CHECK:   dma_pad_24_1 offset 208 size 64 = UINT 0,
// CHECK:   dma_pad_24_2 offset 216 size 64 = UINT 0
// CHECK: }
// CHECK: >

// -----

module @OneDMAWithCustomVersions {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        "NPUReg40XX.NNDMA"() <{dma_descriptor = #NPUReg40XX.DMARegister<
          DMARegister {
            dma_watermark offset 0 size 8 {
              UINT dma_watermark at 0 size 1 = 0 requires 1:1:1
            },
            dma_link_address offset 0 size 64 {
              UINT dma_link_address at 0 size 48 = 0
            },
            dma_lra offset 0 size 64 {
              UINT dma_lra at 63 size 1 = 0
            },
            dma_lba_addr offset 8 size 32 = UINT 0,
            dma_src_aub offset 12 size 8 = UINT 0,
            dma_dst_aub offset 14 size 8 = UINT 0,
            dma_cfg_fields offset 16 size 64 {
              UINT dma_cfg_fields_num_dim at 0 size 3 = 0 requires 1:2:3,
              UINT dma_cfg_fields_int_en at 3 size 1 = 0,
              UINT dma_cfg_fields_int_id at 4 size 8 = 0,
              UINT dma_cfg_fields_src_burst_length at 12 size 4 = 0xF,
              UINT dma_cfg_fields_dst_burst_length at 16 size 4 = 0xF,
              UINT dma_cfg_fields_arb_qos at 20 size 8 = 0xFF,
              UINT dma_cfg_fields_ord at 28 size 1 = 1,
              UINT dma_cfg_fields_barrier_en at 29 size 1 = 1,
              UINT dma_cfg_fields_memset_en at 30 size 1 = 0,
              UINT dma_cfg_fields_atp_en at 31 size 1 = 1,
              UINT dma_cfg_fields_watermark_en at 32 size 1 = 0,
              UINT dma_cfg_fields_rwf_en at 33 size 1 = 0,
              UINT dma_cfg_fields_rws_en at 34 size 1 = 0,
              UINT dma_cfg_fields_src_list_cfg at 35 size 2 = 0,
              UINT dma_cfg_fields_dst_list_cfg at 37 size 2 = 0,
              UINT dma_cfg_fields_conversion_cfg at 39 size 3 = 0,
              UINT dma_cfg_fields_acceleration_cfg at 42 size 2 = 0,
              UINT dma_cfg_fields_tile4_cfg at 44 size 2 = 0,
              UINT dma_cfg_fields_axi_user_bits_cfg at 46 size 2 = 0,
              UINT dma_cfg_fields_hwp_id_en at 48 size 1 = 1,
              UINT dma_cfg_fields_hwp_id at 49 size 12 = 0,
              UINT dma_cfg_fields_reserved at 61 size 3 = 0
            },
            dma_remote_width_fetch offset 24 size 32 = UINT 0,
            dma_width offset 24 size 64 {
              UINT dma_width_src at 0 size 32 = 0x30,
              UINT dma_width_dst at 32 size 32 = 0x30
            },
            dma_acc_info_compress offset 32 size 64 {
              UINT dma_acc_info_compress_dtype at 0 size 2 = 0,
              UINT dma_acc_info_compress_reserved1 at 2 size 1 = 0,
              UINT dma_acc_info_compress_sparse at 3 size 1 = 0,
              UINT dma_acc_info_compress_bitc_en at 4 size 1 = 0,
              UINT dma_acc_info_compress_z at 5 size 10 = 0,
              UINT dma_acc_info_compress_bitmap_buf_sz at 15 size 19 = 0,
              UINT dma_acc_info_compress_reserved2 at 34 size 3 = 0,
              UINT dma_acc_info_compress_bitmap_base_addr at 37 size 27 = 0
            },
            dma_acc_info_decompress offset 32 size 64 {
              UINT dma_acc_info_decompress_dtype at 0 size 2 = 0,
              UINT dma_acc_info_decompress_reserved1 at 2 size 1 = 0,
              UINT dma_acc_info_decompress_sparse at 3 size 1 = 0,
              UINT dma_acc_info_decompress_bitc_en at 4 size 1 = 0,
              UINT dma_acc_info_decompress_z at 5 size 10 = 0,
              UINT dma_acc_info_decompress_reserved2 at 15 size 22 = 0,
              UINT dma_acc_info_decompress_bitmap_base_addr at 37 size 27 = 0
            },
            dma_acc_info_w_prep offset 32 size 64 {
              UINT dma_acc_info_w_prep_dtype at 0 size 2 = 0,
              UINT dma_acc_info_w_prep_reserved1 at 2 size 1 = 0,
              UINT dma_acc_info_w_prep_sparse at 3 size 2 = 0,
              UINT dma_acc_info_w_prep_zeropoint at 5 size 8 = 0,
              UINT dma_acc_info_w_prep_ic at 13 size 14 = 0,
              UINT dma_acc_info_w_prep_filtersize at 27 size 7 = 0,
              UINT dma_acc_info_w_prep_reserved2 at 34 size 3 = 0,
              UINT dma_acc_info_w_prep_bitmap_base_addr at 37 size 27 = 0
            },
            dma_mset_data offset 32 size 32 = UINT 0,
            dma_src_addr offset 40 size 64 {
              UINT dma_src at 0 size 48 = 0,
              UINT dma_sra at 63 size 1 = 0
            },
            dma_dst_addr offset 48 size 64 {
              UINT dma_dst at 0 size 48 = 0,
              UINT dma_dra at 63 size 1 = 0
            },
            dma_sba_addr offset 56 size 32 = UINT 0,
            dma_dba_addr offset 60 size 32 = UINT 0,
            dma_barrier_prod_mask_lower offset 64 size 64 = UINT 0,
            dma_barrier_cons_mask_lower offset 72 size 64 = UINT 0,
            dma_barrier_prod_mask_upper offset 80 size 64 {
              UINT dma_barrier_prod_mask_upper at 0 size 32 = 0
            },
            dma_barrier_cons_mask_upper offset 88 size 64 {
              UINT dma_barrier_cons_mask_upper at 0 size 32 = 0
            },
            dma_list_size offset 96 size 64 {
              UINT dma_list_size_src at 0 size 32 = 0,
              UINT dma_list_size_dst at 32 size 32 = 0
            },
            dma_dim_size offset 96 size 64 {
              UINT dma_dim_size_1_src at 0 size 32 = 0,
              UINT dma_dim_size_1_dst at 32 size 32 = 0
            },
            dma_stride_src_1 offset 104 size 32 = UINT 0,
            dma_stride_dst_1 offset 108 size 32 = UINT 0,
            dma_dim_size_2 offset 112 size 64 {
              UINT dma_dim_size_2_src at 0 size 32 = 0,
              UINT dma_dim_size_2_dst at 32 size 32 = 0
            },
            dma_list_addr offset 112 size 64 {
              UINT dma_list_addr_src at 0 size 32 = 0,
              UINT dma_list_addr_dst at 32 size 32 = 0
            },
            dma_stride_src_2 offset 120 size 32 = UINT 0,
            dma_stride_dst_2 offset 124 size 32 = UINT 0,
            dma_remote_width_store offset 124 size 32 = UINT 0,
            dma_dim_size_src_3 offset 128 size 16 = UINT 0,
            dma_dim_size_src_4 offset 130 size 16 = UINT 0,
            dma_dim_size_dst_3 offset 132 size 16 = UINT 0,
            dma_dim_size_dst_4 offset 134 size 16 = UINT 0,
            dma_dim_size_src_5 offset 136 size 16 = UINT 0,
            dma_dim_size_dst_5 offset 140 size 16 = UINT 0,
            dma_stride_src_3 offset 144 size 32 = UINT 0,
            dma_stride_dst_3 offset 148 size 32 = UINT 0,
            dma_stride_src_4 offset 152 size 32 = UINT 0,
            dma_stride_dst_4 offset 156 size 32 = UINT 0,
            dma_stride_src_5 offset 160 size 32 = UINT 0,
            dma_stride_dst_5 offset 164 size 32 = UINT 0,
            dma_word_21_reserved offset 168 size 64 = UINT 0,
            dma_word_22_reserved offset 176 size 64 = UINT 0,
            dma_word_23_reserved offset 184 size 64 = UINT 0,
            dma_barriers_sched offset 192 size 64 {
              UINT start_after_ at 0 size 32 = 1,
              UINT clean_after_ at 32 size 32 = 2
            },
            dma_pad_24_0 offset 200 size 64 = UINT 0,
            dma_pad_24_1 offset 208 size 64 = UINT 0,
            dma_pad_24_2 offset 216 size 64 = UINT 0
          }
        >, input = @DeclareBuffer0, output_buffs = [@DeclareBuffer1], sym_name = "NNDMA_0_0_0"}> : () -> ()
      }
    }
    return
  }
}

// CHECK: dma_descriptor = #NPUReg40XX.DMARegister<
// CHECK: DMARegister {
// CHECK:   dma_watermark offset 0 size 8 {
// CHECK:     UINT dma_watermark at 0 size 1 = 0 requires 1:1:1
// CHECK:   },
// CHECK:   dma_link_address offset 0 size 64 {
// CHECK:     UINT dma_link_address at 0 size 48 = 0
// CHECK:   },
// CHECK:   dma_lra offset 0 size 64 {
// CHECK:     UINT dma_lra at 63 size 1 = 0
// CHECK:   },
// CHECK:   dma_lba_addr offset 8 size 32 = UINT 0,
// CHECK:   dma_src_aub offset 12 size 8 = UINT 0,
// CHECK:   dma_dst_aub offset 14 size 8 = UINT 0,
// CHECK:   dma_cfg_fields offset 16 size 64 {
// CHECK:     UINT dma_cfg_fields_num_dim at 0 size 3 = 0 requires 1:2:3,
// CHECK:     UINT dma_cfg_fields_int_en at 3 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_int_id at 4 size 8 = 0,
// CHECK:     UINT dma_cfg_fields_src_burst_length at 12 size 4 = 0xF,
// CHECK:     UINT dma_cfg_fields_dst_burst_length at 16 size 4 = 0xF,
// CHECK:     UINT dma_cfg_fields_arb_qos at 20 size 8 = 0xFF,
// CHECK:     UINT dma_cfg_fields_ord at 28 size 1 = 1,
// CHECK:     UINT dma_cfg_fields_barrier_en at 29 size 1 = 1,
// CHECK:     UINT dma_cfg_fields_memset_en at 30 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_atp_en at 31 size 1 = 1,
// CHECK:     UINT dma_cfg_fields_watermark_en at 32 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_rwf_en at 33 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_rws_en at 34 size 1 = 0,
// CHECK:     UINT dma_cfg_fields_src_list_cfg at 35 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_dst_list_cfg at 37 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_conversion_cfg at 39 size 3 = 0,
// CHECK:     UINT dma_cfg_fields_acceleration_cfg at 42 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_tile4_cfg at 44 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_axi_user_bits_cfg at 46 size 2 = 0,
// CHECK:     UINT dma_cfg_fields_hwp_id_en at 48 size 1 = 1,
// CHECK:     UINT dma_cfg_fields_hwp_id at 49 size 12 = 0,
// CHECK:     UINT dma_cfg_fields_reserved at 61 size 3 = 0
// CHECK:   },
// CHECK:   dma_remote_width_fetch offset 24 size 32 = UINT 0x30,
// CHECK:   dma_width offset 24 size 64 {
// CHECK:     UINT dma_width_src at 0 size 32 = 0x30,
// CHECK:     UINT dma_width_dst at 32 size 32 = 0x30
// CHECK:   },
// CHECK:   dma_acc_info_compress offset 32 size 64 {
// CHECK:     UINT dma_acc_info_compress_dtype at 0 size 2 = 0,
// CHECK:     UINT dma_acc_info_compress_reserved1 at 2 size 1 = 0,
// CHECK:     UINT dma_acc_info_compress_sparse at 3 size 1 = 0,
// CHECK:     UINT dma_acc_info_compress_bitc_en at 4 size 1 = 0,
// CHECK:     UINT dma_acc_info_compress_z at 5 size 10 = 0,
// CHECK:     UINT dma_acc_info_compress_bitmap_buf_sz at 15 size 19 = 0,
// CHECK:     UINT dma_acc_info_compress_reserved2 at 34 size 3 = 0,
// CHECK:     UINT dma_acc_info_compress_bitmap_base_addr at 37 size 27 = 0
// CHECK:   },
// CHECK:   dma_acc_info_decompress offset 32 size 64 {
// CHECK:     UINT dma_acc_info_decompress_dtype at 0 size 2 = 0,
// CHECK:     UINT dma_acc_info_decompress_reserved1 at 2 size 1 = 0,
// CHECK:     UINT dma_acc_info_decompress_sparse at 3 size 1 = 0,
// CHECK:     UINT dma_acc_info_decompress_bitc_en at 4 size 1 = 0,
// CHECK:     UINT dma_acc_info_decompress_z at 5 size 10 = 0,
// CHECK:     UINT dma_acc_info_decompress_reserved2 at 15 size 22 = 0,
// CHECK:     UINT dma_acc_info_decompress_bitmap_base_addr at 37 size 27 = 0
// CHECK:   },
// CHECK:   dma_acc_info_w_prep offset 32 size 64 {
// CHECK:     UINT dma_acc_info_w_prep_dtype at 0 size 2 = 0,
// CHECK:     UINT dma_acc_info_w_prep_reserved1 at 2 size 1 = 0,
// CHECK:     UINT dma_acc_info_w_prep_sparse at 3 size 2 = 0,
// CHECK:     UINT dma_acc_info_w_prep_zeropoint at 5 size 8 = 0,
// CHECK:     UINT dma_acc_info_w_prep_ic at 13 size 14 = 0,
// CHECK:     UINT dma_acc_info_w_prep_filtersize at 27 size 7 = 0,
// CHECK:     UINT dma_acc_info_w_prep_reserved2 at 34 size 3 = 0,
// CHECK:     UINT dma_acc_info_w_prep_bitmap_base_addr at 37 size 27 = 0
// CHECK:   },
// CHECK:   dma_mset_data offset 32 size 32 = UINT 0,
// CHECK:   dma_src_addr offset 40 size 64 {
// CHECK:     UINT dma_src at 0 size 48 = 0,
// CHECK:     UINT dma_sra at 63 size 1 = 0
// CHECK:   },
// CHECK:   dma_dst_addr offset 48 size 64 {
// CHECK:     UINT dma_dst at 0 size 48 = 0,
// CHECK:     UINT dma_dra at 63 size 1 = 0
// CHECK:   },
// CHECK:   dma_sba_addr offset 56 size 32 = UINT 0,
// CHECK:   dma_dba_addr offset 60 size 32 = UINT 0,
// CHECK:   dma_barrier_prod_mask_lower offset 64 size 64 = UINT 0,
// CHECK:   dma_barrier_cons_mask_lower offset 72 size 64 = UINT 0,
// CHECK:   dma_barrier_prod_mask_upper offset 80 size 64 {
// CHECK:     UINT dma_barrier_prod_mask_upper at 0 size 32 = 0
// CHECK:   },
// CHECK:   dma_barrier_cons_mask_upper offset 88 size 64 {
// CHECK:     UINT dma_barrier_cons_mask_upper at 0 size 32 = 0
// CHECK:   },
// CHECK:   dma_list_size offset 96 size 64 {
// CHECK:     UINT dma_list_size_src at 0 size 32 = 0,
// CHECK:     UINT dma_list_size_dst at 32 size 32 = 0
// CHECK:   },
// CHECK:   dma_dim_size offset 96 size 64 {
// CHECK:     UINT dma_dim_size_1_src at 0 size 32 = 0,
// CHECK:     UINT dma_dim_size_1_dst at 32 size 32 = 0
// CHECK:   },
// CHECK:   dma_stride_src_1 offset 104 size 32 = UINT 0,
// CHECK:   dma_stride_dst_1 offset 108 size 32 = UINT 0,
// CHECK:   dma_dim_size_2 offset 112 size 64 {
// CHECK:     UINT dma_dim_size_2_src at 0 size 32 = 0,
// CHECK:     UINT dma_dim_size_2_dst at 32 size 32 = 0
// CHECK:   },
// CHECK:   dma_list_addr offset 112 size 64 {
// CHECK:     UINT dma_list_addr_src at 0 size 32 = 0,
// CHECK:     UINT dma_list_addr_dst at 32 size 32 = 0
// CHECK:   },
// CHECK:   dma_stride_src_2 offset 120 size 32 = UINT 0,
// CHECK:   dma_stride_dst_2 offset 124 size 32 = UINT 0,
// CHECK:   dma_remote_width_store offset 124 size 32 = UINT 0,
// CHECK:   dma_dim_size_src_3 offset 128 size 16 = UINT 0,
// CHECK:   dma_dim_size_src_4 offset 130 size 16 = UINT 0,
// CHECK:   dma_dim_size_dst_3 offset 132 size 16 = UINT 0,
// CHECK:   dma_dim_size_dst_4 offset 134 size 16 = UINT 0,
// CHECK:   dma_dim_size_src_5 offset 136 size 16 = UINT 0,
// CHECK:   dma_dim_size_dst_5 offset 140 size 16 = UINT 0,
// CHECK:   dma_stride_src_3 offset 144 size 32 = UINT 0,
// CHECK:   dma_stride_dst_3 offset 148 size 32 = UINT 0,
// CHECK:   dma_stride_src_4 offset 152 size 32 = UINT 0,
// CHECK:   dma_stride_dst_4 offset 156 size 32 = UINT 0,
// CHECK:   dma_stride_src_5 offset 160 size 32 = UINT 0,
// CHECK:   dma_stride_dst_5 offset 164 size 32 = UINT 0,
// CHECK:   dma_word_21_reserved offset 168 size 64 = UINT 0,
// CHECK:   dma_word_22_reserved offset 176 size 64 = UINT 0,
// CHECK:   dma_word_23_reserved offset 184 size 64 = UINT 0,
// CHECK:   dma_barriers_sched offset 192 size 64 {
// CHECK:     UINT start_after_ at 0 size 32 = 1,
// CHECK:     UINT clean_after_ at 32 size 32 = 2
// CHECK:   },
// CHECK:   dma_pad_24_0 offset 200 size 64 = UINT 0,
// CHECK:   dma_pad_24_1 offset 208 size 64 = UINT 0,
// CHECK:   dma_pad_24_2 offset 216 size 64 = UINT 0
// CHECK: }
// CHECK: >
