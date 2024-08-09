//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_40xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

using namespace npu40xx;

#define CREATE_HW_DMA_DESC(field, value)                                   \
    [] {                                                                   \
        nn_public::VpuDMATask hwDMADesc;                                   \
        memset(reinterpret_cast<void*>(&hwDMADesc), 0, sizeof(hwDMADesc)); \
        hwDMADesc.field = value;                                           \
        return hwDMADesc;                                                  \
    }()

class NPUReg40XX_DMARegisterTest :
        public MLIR_RegMappedNPUReg40XXUnitBase<nn_public::VpuDMATask, vpux::NPUReg40XX::RegMapped_DMARegisterType> {};

TEST_P(NPUReg40XX_DMARegisterTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, nn_public::VpuDMATask>> valuesSet = {
        // word 0
        {{
                 {"dma_watermark", {{"dma_watermark", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.watermark, 1)},
        {{
                 {"dma_link_address", {{"dma_link_address", 0x123}}},
         },
         CREATE_HW_DMA_DESC(transaction_.link_address, 0x123)},
        {{
                 {"dma_lra", {{"dma_lra", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.lra, 1)},
        // word 1
        {{
                 {"dma_lba_addr", {{"dma_lba_addr", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.lba_addr, 1)},
        {{
                 {"dma_src_aub", {{"dma_src_aub", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.src_aub, 1)},
        {{
                 {"dma_dst_aub", {{"dma_dst_aub", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dst_aub, 1)},
        // word 2
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_num_dim", 7}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.num_dim, 7)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_int_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.int_en, 1)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_int_id", 255}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.int_id, 255)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_src_burst_length", 15}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.src_burst_length, 15)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_dst_burst_length", 15}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.dst_burst_length, 15)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_arb_qos", 255}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.arb_qos, 255)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_ord", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.ord, 1)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_barrier_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.barrier_en, 1)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_memset_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.memset_en, 1)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_atp_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.atp_en, 1)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_watermark_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.watermark_en, 1)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_rwf_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.rwf_en, 1)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_rws_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.rws_en, 1)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_src_list_cfg", 3}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.src_list_cfg, 3)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_dst_list_cfg", 3}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.dst_list_cfg, 3)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_conversion_cfg", 7}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.conversion_cfg, 7)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_acceleration_cfg", 3}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.acceleration_cfg, 3)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_tile4_cfg", 3}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.tile4_cfg, 3)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_axi_user_bits_cfg", 3}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.axi_user_bits_cfg, 3)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_hwp_id_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.hwp_id_en, 1)},
        {{
                 {"dma_cfg_fields", {{"dma_cfg_fields_hwp_id", 0xFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.cfg.fields.hwp_id, 0xFFF)},
        // word 3
        {{
                 {"dma_remote_width_fetch", {{"dma_remote_width_fetch", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.remote_width_fetch, 0xFFFFFFFF)},
        {{
                 {"dma_width", {{"dma_width_src", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.width.src, 0xFFFFFFFF)},
        {{
                 {"dma_width", {{"dma_width_dst", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.width.dst, 0xFFFFFFFF)},
        // word 4
        // compress
        {{
                 {"dma_acc_info_compress", {{"dma_acc_info_compress_dtype", 3}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.compress.dtype, 3)},
        {{
                 {"dma_acc_info_compress", {{"dma_acc_info_compress_sparse", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.compress.sparse, 1)},
        {{
                 {"dma_acc_info_compress", {{"dma_acc_info_compress_bitc_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.compress.bitc_en, 1)},
        {{
                 {"dma_acc_info_compress", {{"dma_acc_info_compress_z", 0x3FF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.compress.z, 0x3FF)},
        {{
                 {"dma_acc_info_compress", {{"dma_acc_info_compress_bitmap_buf_sz", 0x7FFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.compress.bitmap_buf_sz, 0x7FFFF)},
        {{
                 {"dma_acc_info_compress", {{"dma_acc_info_compress_bitmap_base_addr", 0x7FFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.compress.bitmap_base_addr, 0x7FFFFFF)},
        // decompress
        {{
                 {"dma_acc_info_decompress", {{"dma_acc_info_decompress_dtype", 3}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.decompress.dtype, 3)},
        {{
                 {"dma_acc_info_decompress", {{"dma_acc_info_decompress_sparse", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.decompress.sparse, 1)},
        {{
                 {"dma_acc_info_decompress", {{"dma_acc_info_decompress_bitc_en", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.decompress.bitc_en, 1)},
        {{
                 {"dma_acc_info_decompress", {{"dma_acc_info_decompress_z", 0x3FF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.decompress.z, 0x3FF)},
        {{
                 {"dma_acc_info_decompress", {{"dma_acc_info_decompress_bitmap_base_addr", 0x7FFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.decompress.bitmap_base_addr, 0x7FFFFFF)},
        // w_prep
        {{
                 {"dma_acc_info_w_prep", {{"dma_acc_info_w_prep_dtype", 3}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.w_prep.dtype, 3)},
        {{
                 {"dma_acc_info_w_prep", {{"dma_acc_info_w_prep_sparse", 3}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.w_prep.sparse, 3)},
        {{
                 {"dma_acc_info_w_prep", {{"dma_acc_info_w_prep_zeropoint", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.w_prep.zeropoint, 0xFF)},
        {{
                 {"dma_acc_info_w_prep", {{"dma_acc_info_w_prep_ic", 0x3FFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.w_prep.ic, 0x3FFF)},
        {{
                 {"dma_acc_info_w_prep", {{"dma_acc_info_w_prep_filtersize", 0x7F}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.w_prep.filtersize, 0x7F)},
        {{
                 {"dma_acc_info_w_prep", {{"dma_acc_info_w_prep_bitmap_base_addr", 0x7FFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.acc_info.w_prep.bitmap_base_addr, 0x7FFFFFF)},
        // mset
        {{
                 {"dma_mset_data", {{"dma_mset_data", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.mset_data, 0xFFFFFFFF)},
        // word 5
        {{
                 {"dma_src_addr", {{"dma_src", 0xFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.src, 0xFFFFFFFFFFFF)},
        {{
                 {"dma_src_addr", {{"dma_sra", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.sra, 1)},
        // word 6
        {{
                 {"dma_dst_addr", {{"dma_dst", 0xFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dst, 0xFFFFFFFFFFFF)},
        {{
                 {"dma_dst_addr", {{"dma_dra", 1}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dra, 1)},
        // word 7
        {{
                 {"dma_sba_addr", {{"dma_sba_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.sba_addr, 0xFFFFFFFF)},
        {{
                 {"dma_dba_addr", {{"dma_dba_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dba_addr, 0xFFFFFFFF)},
        // word 8
        {{
                 {"dma_barrier_prod_mask_lower", {{"dma_barrier_prod_mask_lower", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.barrier.prod_mask_lower, 0xFFFFFFFF)},
        // word 9
        {{
                 {"dma_barrier_cons_mask_lower", {{"dma_barrier_cons_mask_lower", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.barrier.cons_mask_lower, 0xFFFFFFFF)},
        // word 10
        {{
                 {"dma_barrier_prod_mask_upper", {{"dma_barrier_prod_mask_upper", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.barrier.prod_mask_upper, 0xFFFFFFFF)},
        // word 11
        {{
                 {"dma_barrier_cons_mask_upper", {{"dma_barrier_cons_mask_upper", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.barrier.cons_mask_upper, 0xFFFFFFFF)},
        // word 12
        {{
                 {"dma_list_size", {{"dma_list_size_src", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.list_size.src, 0xFFFFFFFF)},
        {{
                 {"dma_list_size", {{"dma_list_size_dst", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.list_size.dst, 0xFFFFFFFF)},
        {{
                 {"dma_dim_size", {{"dma_dim_size_1_src", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_1.src, 0xFFFFFFFF)},
        {{
                 {"dma_dim_size", {{"dma_dim_size_1_dst", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_1.dst, 0xFFFFFFFF)},
        // word 13
        {{
                 {"dma_stride_src_1", {{"dma_stride_src_1", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_src_1, 0xFFFFFFFF)},
        {{
                 {"dma_stride_dst_1", {{"dma_stride_dst_1", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_dst_1, 0xFFFFFFFF)},
        // word 14
        {{
                 {"dma_dim_size_2", {{"dma_dim_size_2_src", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_2.src, 0xFFFFFFFF)},
        {{
                 {"dma_dim_size_2", {{"dma_dim_size_2_dst", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_2.dst, 0xFFFFFFFF)},
        {{
                 {"dma_list_addr", {{"dma_list_addr_src", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.list_addr.src, 0xFFFFFFFF)},
        {{
                 {"dma_list_addr", {{"dma_list_addr_dst", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.list_addr.dst, 0xFFFFFFFF)},
        // word 15
        {{
                 {"dma_stride_src_2", {{"dma_stride_src_2", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_src_2, 0xFFFFFFFF)},
        {{
                 {"dma_stride_dst_2", {{"dma_stride_dst_2", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_dst_2, 0xFFFFFFFF)},
        {{
                 {"dma_remote_width_store", {{"dma_remote_width_store", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.remote_width_store, 0xFFFFFFFF)},
        // word 16
        {{
                 {"dma_dim_size_src_3", {{"dma_dim_size_src_3", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_src_3, 0xFFFF)},
        {{
                 {"dma_dim_size_src_4", {{"dma_dim_size_src_4", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_src_4, 0xFFFF)},
        {{
                 {"dma_dim_size_dst_3", {{"dma_dim_size_dst_3", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_dst_3, 0xFFFF)},
        {{
                 {"dma_dim_size_dst_4", {{"dma_dim_size_dst_4", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_dst_4, 0xFFFF)},
        // word 17
        {{
                 {"dma_dim_size_src_5", {{"dma_dim_size_src_5", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_src_5, 0xFFFF)},
        {{
                 {"dma_dim_size_dst_5", {{"dma_dim_size_dst_5", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.dim_size_dst_5, 0xFFFF)},
        // word 18
        {{
                 {"dma_stride_src_3", {{"dma_stride_src_3", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_src_3, 0xFFFFFFFF)},
        {{
                 {"dma_stride_dst_3", {{"dma_stride_dst_3", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_dst_3, 0xFFFFFFFF)},
        // word 19
        {{
                 {"dma_stride_src_4", {{"dma_stride_src_4", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_src_4, 0xFFFFFFFF)},
        {{
                 {"dma_stride_dst_4", {{"dma_stride_dst_4", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_dst_4, 0xFFFFFFFF)},
        {{
                 {"dma_stride_src_5", {{"dma_stride_src_5", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_src_5, 0xFFFFFFFF)},
        {{
                 {"dma_stride_dst_5", {{"dma_stride_dst_5", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(transaction_.stride_dst_5, 0xFFFFFFFF)},
        {{
                 {"dma_barriers_sched", {{"start_after_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(barriers_sched_.start_after_, 0xFFFFFFFF)},
        {{
                 {"dma_barriers_sched", {{"clean_after_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(barriers_sched_.clean_after_, 0xFFFFFFFF)},
};

INSTANTIATE_TEST_CASE_P(NPUReg40XX_MappedRegs, NPUReg40XX_DMARegisterTest, testing::ValuesIn(valuesSet));
