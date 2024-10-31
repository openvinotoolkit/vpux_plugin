//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_40xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

using namespace npu40xx;

#define CREATE_HW_DMA_DESC(field, value)                                                 \
    [] {                                                                                 \
        nn_public::VpuDPUVariant hwDPUVariantDesc;                                       \
        memset(reinterpret_cast<void*>(&hwDPUVariantDesc), 0, sizeof(hwDPUVariantDesc)); \
        hwDPUVariantDesc.field = value;                                                  \
        return hwDPUVariantDesc;                                                         \
    }()

class NPUReg40XX_NpuDPUVariantTest :
        public MLIR_RegMappedNPUReg40XXUnitBase<nn_public::VpuDPUVariant,
                                                vpux::NPUReg40XX::RegMapped_DpuVariantRegisterType> {};

TEST_P(NPUReg40XX_NpuDPUVariantTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, nn_public::VpuDPUVariant>> dpuVariantFieldSet = {
        // invar_ptr ---------------------------------------------------------------------
        {{
                 {"invar_ptr", {{"invar_ptr", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.invar_ptr.invar_ptr_bf.invar_ptr, 0xFFFF)},
        {{
                 {"invar_ptr", {{"var_tag", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.invar_ptr.invar_ptr_bf.var_tag, 0xFFFF)},
        // workload_size0 ---------------------------------------------------------------------
        {{
                 {"workload_size0", {{"workload_size_x", 0x3FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_size0.workload_size0_bf.workload_size_x, 0x3FFF)},
        {{
                 {"workload_size0", {{"workload_size_y", 0x3FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_size0.workload_size0_bf.workload_size_y, 0x3FFF)},
        // workload_size1 ---------------------------------------------------------------------
        {{
                 {"workload_size1", {{"workload_size_z", 0x3FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_size1.workload_size1_bf.workload_size_z, 0x3FFF)},
        {{
                 {"workload_size1", {{"pad_count_up", 7}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_size1.workload_size1_bf.pad_count_up, 7)},
        {{
                 {"workload_size1", {{"pad_count_left", 7}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_size1.workload_size1_bf.pad_count_left, 7)},
        {{
                 {"workload_size1", {{"pad_count_down", 7}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_size1.workload_size1_bf.pad_count_down, 7)},
        {{
                 {"workload_size1", {{"pad_count_right", 7}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_size1.workload_size1_bf.pad_count_right, 7)},
        // workload_start0 ---------------------------------------------------------------------
        {{
                 {"workload_start0", {{"workload_start_x", 0x3FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_start0.workload_start0_bf.workload_start_x, 0x3FFF)},
        {{
                 {"workload_start0", {{"workload_start_y", 0x3FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_start0.workload_start0_bf.workload_start_y, 0x3FFF)},
        // workload_start1 ---------------------------------------------------------------------
        {{
                 {"workload_start1", {{"workload_start_z", 0x3FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.workload_start1.workload_start1_bf.workload_start_z, 0x3FFF)},
        // offset_addr ---------------------------------------------------------------------
        {{
                 {"offset_addr", {{"nthw_ntk", 3}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.nthw_ntk, 3)},
        {{
                 {"offset_addr", {{"bin_cfg", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.bin_cfg, 1)},
        {{
                 {"offset_addr", {{"conv_cond", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.conv_cond, 1)},
        {{
                 {"offset_addr", {{"dense_se", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.dense_se, 1)},
        {{
                 {"offset_addr", {{"idx_quad", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.idx_quad, 1)},
        {{
                 {"offset_addr", {{"swizzle_key_offset", 7}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.swizzle_key, 7)},
        {{
                 {"offset_addr", {{"idu_mrm_clk_en", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.idu_mrm_clk_en, 1)},
        {{
                 {"offset_addr", {{"odu_clk_en", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.odu_clk_en, 1)},
        {{
                 {"offset_addr", {{"mpe_clk_en", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.mpe_clk_en, 1)},
        {{
                 {"offset_addr", {{"ppe_clk_en", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.ppe_clk_en, 1)},
        {{
                 {"offset_addr", {{"odu_stat_en", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.odu_stat_en, 1)},
        {{
                 {"offset_addr", {{"idu_stat_en", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.idu_stat_en, 1)},
        {{
                 {"offset_addr", {{"odu_stat_clr_mode", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.odu_stat_clr_mode, 1)},
        {{
                 {"offset_addr", {{"idu_stat_clr_mode", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.idu_stat_clr_mode, 1)},
        {{
                 {"offset_addr", {{"shave_l2_cache_en", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.shave_l2_cache_en, 1)},
        {{
                 {"offset_addr", {{"idu_dbg_en", 3}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.idu_dbg_en, 3)},
        {{
                 {"offset_addr", {{"wt_swizzle_key", 7}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.wt_swizzle_key, 7)},
        {{
                 {"offset_addr", {{"wt_swizzle_sel", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.offset_addr.offset_addr_bf.wt_swizzle_sel, 1)},
        // hwp_wload_id ---------------------------------------------------------------------
        {{
                 {"hwp_wload_id", {{"hwp_wload_id", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.hwp_wload_id.hwp_wload_id_bf.wload_id, 0xFFFF)},
        // var_cfg
        {{
                 {"var_cfg", {{"invar_line_cnt_en", 0b1}}},
         },
         CREATE_HW_DMA_DESC(registers_.var_cfg.var_cfg_bf.invar_line_cnt_en, 0b1)},
        {{
                 {"var_cfg", {{"invar_line_cnt", 0xC}}},
         },
         CREATE_HW_DMA_DESC(registers_.var_cfg.var_cfg_bf.invar_line_cnt_cnt, 0xC)},
        {{
                 {"var_cfg", {{"invar_lptr_force", 0b1}}},
         },
         CREATE_HW_DMA_DESC(registers_.var_cfg.var_cfg_bf.invar_lptr_force, 0b1)},
        {{
                 {"var_cfg", {{"next_sram_job_valid", 0b1}}},
         },
         CREATE_HW_DMA_DESC(registers_.var_cfg.var_cfg_bf.next_sram_job_valid, 0b1)},
        {{
                 {"var_cfg", {{"next_sram_job_addr", 0x1234}}},
         },
         CREATE_HW_DMA_DESC(registers_.var_cfg.var_cfg_bf.next_sram_job_addr, 0x1234)},
        // cbarrier_lo
        {{
                 {"cbarrier_lo", {{"cbarrier_lo", 0xDEADC0DE}}},
         },
         CREATE_HW_DMA_DESC(registers_.cbarrier_lo, 0xDEADC0DE)},
        // cbarrier_hi
        {{
                 {"cbarrier_hi", {{"cbarrier_hi", 0xBAADC0DE}}},
         },
         CREATE_HW_DMA_DESC(registers_.cbarrier_hi, 0xBAADC0DE)},
        // pbarrier_lo
        {{
                 {"pbarrier_lo", {{"pbarrier_lo", 0xDEADC0DE}}},
         },
         CREATE_HW_DMA_DESC(registers_.pbarrier_lo, 0xDEADC0DE)},
        // pbarrier_hi
        {{
                 {"pbarrier_hi", {{"pbarrier_hi", 0xBAADC0DE}}},
         },
         CREATE_HW_DMA_DESC(registers_.pbarrier_hi, 0xBAADC0DE)},

        // halo_region 0 ---------------------------------------------------------------------
        {{
                 {"halo_region0A", {{"sp_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_a.halo_region_a_bf.sp_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region0A", {{"tile_select", 0x7F}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_a.halo_region_a_bf.tile_select, 0x7F)},
        {{
                 {"halo_region0A", {{"rsvdA", 3}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_a.halo_region_a_bf.rsvd, 3)},
        {{
                 {"halo_region0A", {{"enable", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_a.halo_region_a_bf.enable, 1)},
        {{
                 {"halo_region0B", {{"ac_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_b.halo_region_b_bf.ac_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region0B", {{"target_width_lsb", 0x3FF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_b.halo_region_b_bf.target_width_lsb, 0x3FF)},
        {{
                 {"halo_region0C", {{"begin_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_c.halo_region_c_bf.begin_x, 0x1FFF)},
        {{
                 {"halo_region0C", {{"begin_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_c.halo_region_c_bf.begin_y, 0x1FFF)},
        {{
                 {"halo_region0C", {{"target_width_msb", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_c.halo_region_c_bf.target_width_msb, 0xF)},
        {{
                 {"halo_region0D", {{"end_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_d.halo_region_d_bf.end_x, 0x1FFF)},
        {{
                 {"halo_region0D", {{"end_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_d.halo_region_d_bf.end_y, 0x1FFF)},
        {{
                 {"halo_region0D", {{"rsvdD", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[0].halo_region_d.halo_region_d_bf.rsvd, 0xF)},
        // halo_region 1 ---------------------------------------------------------------------
        {{
                 {"halo_region1A", {{"sp_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_a.halo_region_a_bf.sp_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region1A", {{"tile_select", 0x7F}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_a.halo_region_a_bf.tile_select, 0x7F)},
        {{
                 {"halo_region1A", {{"rsvdA", 3}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_a.halo_region_a_bf.rsvd, 3)},
        {{
                 {"halo_region1A", {{"enable", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_a.halo_region_a_bf.enable, 1)},
        {{
                 {"halo_region1B", {{"ac_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_b.halo_region_b_bf.ac_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region1B", {{"target_width_lsb", 0x3FF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_b.halo_region_b_bf.target_width_lsb, 0x3FF)},
        {{
                 {"halo_region1C", {{"begin_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_c.halo_region_c_bf.begin_x, 0x1FFF)},
        {{
                 {"halo_region1C", {{"begin_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_c.halo_region_c_bf.begin_y, 0x1FFF)},
        {{
                 {"halo_region1C", {{"target_width_msb", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_c.halo_region_c_bf.target_width_msb, 0xF)},
        {{
                 {"halo_region1D", {{"end_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_d.halo_region_d_bf.end_x, 0x1FFF)},
        {{
                 {"halo_region1D", {{"end_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_d.halo_region_d_bf.end_y, 0x1FFF)},
        {{
                 {"halo_region1D", {{"rsvdD", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[1].halo_region_d.halo_region_d_bf.rsvd, 0xF)},
        // halo_region 2 ---------------------------------------------------------------------
        {{
                 {"halo_region2A", {{"sp_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_a.halo_region_a_bf.sp_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region2A", {{"tile_select", 0x7F}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_a.halo_region_a_bf.tile_select, 0x7F)},
        {{
                 {"halo_region2A", {{"rsvdA", 3}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_a.halo_region_a_bf.rsvd, 3)},
        {{
                 {"halo_region2A", {{"enable", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_a.halo_region_a_bf.enable, 1)},
        {{
                 {"halo_region2B", {{"ac_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_b.halo_region_b_bf.ac_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region2B", {{"target_width_lsb", 0x3FF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_b.halo_region_b_bf.target_width_lsb, 0x3FF)},
        {{
                 {"halo_region2C", {{"begin_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_c.halo_region_c_bf.begin_x, 0x1FFF)},
        {{
                 {"halo_region2C", {{"begin_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_c.halo_region_c_bf.begin_y, 0x1FFF)},
        {{
                 {"halo_region2C", {{"target_width_msb", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_c.halo_region_c_bf.target_width_msb, 0xF)},
        {{
                 {"halo_region2D", {{"end_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_d.halo_region_d_bf.end_x, 0x1FFF)},
        {{
                 {"halo_region2D", {{"end_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_d.halo_region_d_bf.end_y, 0x1FFF)},
        {{
                 {"halo_region2D", {{"rsvdD", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[2].halo_region_d.halo_region_d_bf.rsvd, 0xF)},
        // halo_region 3 ---------------------------------------------------------------------
        {{
                 {"halo_region3A", {{"sp_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_a.halo_region_a_bf.sp_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region3A", {{"tile_select", 0x7F}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_a.halo_region_a_bf.tile_select, 0x7F)},
        {{
                 {"halo_region3A", {{"rsvdA", 3}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_a.halo_region_a_bf.rsvd, 3)},
        {{
                 {"halo_region3A", {{"enable", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_a.halo_region_a_bf.enable, 1)},
        {{
                 {"halo_region3B", {{"ac_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_b.halo_region_b_bf.ac_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region3B", {{"target_width_lsb", 0x3FF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_b.halo_region_b_bf.target_width_lsb, 0x3FF)},
        {{
                 {"halo_region3C", {{"begin_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_c.halo_region_c_bf.begin_x, 0x1FFF)},
        {{
                 {"halo_region3C", {{"begin_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_c.halo_region_c_bf.begin_y, 0x1FFF)},
        {{
                 {"halo_region3C", {{"target_width_msb", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_c.halo_region_c_bf.target_width_msb, 0xF)},
        {{
                 {"halo_region3D", {{"end_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_d.halo_region_d_bf.end_x, 0x1FFF)},
        {{
                 {"halo_region3D", {{"end_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_d.halo_region_d_bf.end_y, 0x1FFF)},
        {{
                 {"halo_region3D", {{"rsvdD", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[3].halo_region_d.halo_region_d_bf.rsvd, 0xF)},
        // halo_region 4 ---------------------------------------------------------------------
        {{
                 {"halo_region4A", {{"sp_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_a.halo_region_a_bf.sp_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region4A", {{"tile_select", 0x7F}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_a.halo_region_a_bf.tile_select, 0x7F)},
        {{
                 {"halo_region4A", {{"rsvdA", 3}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_a.halo_region_a_bf.rsvd, 3)},
        {{
                 {"halo_region4A", {{"enable", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_a.halo_region_a_bf.enable, 1)},
        {{
                 {"halo_region4B", {{"ac_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_b.halo_region_b_bf.ac_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region4B", {{"target_width_lsb", 0x3FF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_b.halo_region_b_bf.target_width_lsb, 0x3FF)},
        {{
                 {"halo_region4C", {{"begin_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_c.halo_region_c_bf.begin_x, 0x1FFF)},
        {{
                 {"halo_region4C", {{"begin_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_c.halo_region_c_bf.begin_y, 0x1FFF)},
        {{
                 {"halo_region4C", {{"target_width_msb", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_c.halo_region_c_bf.target_width_msb, 0xF)},
        {{
                 {"halo_region4D", {{"end_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_d.halo_region_d_bf.end_x, 0x1FFF)},
        {{
                 {"halo_region4D", {{"end_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_d.halo_region_d_bf.end_y, 0x1FFF)},
        {{
                 {"halo_region4D", {{"rsvdD", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[4].halo_region_d.halo_region_d_bf.rsvd, 0xF)},
        // halo_region 5 ---------------------------------------------------------------------
        {{
                 {"halo_region5A", {{"sp_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_a.halo_region_a_bf.sp_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region5A", {{"tile_select", 0x7F}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_a.halo_region_a_bf.tile_select, 0x7F)},
        {{
                 {"halo_region5A", {{"rsvdA", 3}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_a.halo_region_a_bf.rsvd, 3)},
        {{
                 {"halo_region5A", {{"enable", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_a.halo_region_a_bf.enable, 1)},
        {{
                 {"halo_region5B", {{"ac_adr_offset", 0x3FFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_b.halo_region_b_bf.ac_adr_offset, 0x3FFFFF)},
        {{
                 {"halo_region5B", {{"target_width_lsb", 0x3FF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_b.halo_region_b_bf.target_width_lsb, 0x3FF)},
        {{
                 {"halo_region5C", {{"begin_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_c.halo_region_c_bf.begin_x, 0x1FFF)},
        {{
                 {"halo_region5C", {{"begin_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_c.halo_region_c_bf.begin_y, 0x1FFF)},
        {{
                 {"halo_region5C", {{"target_width_msb", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_c.halo_region_c_bf.target_width_msb, 0xF)},
        {{
                 {"halo_region5D", {{"end_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_d.halo_region_d_bf.end_x, 0x1FFF)},
        {{
                 {"halo_region5D", {{"end_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_d.halo_region_d_bf.end_y, 0x1FFF)},
        {{
                 {"halo_region5D", {{"rsvdD", 0xF}}},
         },
         CREATE_HW_DMA_DESC(registers_.halo_region[5].halo_region_d.halo_region_d_bf.rsvd, 0xF)},
        // dpu_cfg ---------------------------------------------------------------------
        {{
                 {"dpu_cfg", {{"workload_start_odu", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.workload_start_odu, 1)},
        {{
                 {"dpu_cfg", {{"workload_start_idu", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.workload_start_idu, 1)},
        {{
                 {"dpu_cfg", {{"workload_prm_sel", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.workload_prm_sel, 1)},
        {{
                 {"dpu_cfg", {{"workload_valid", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.workload_valid, 1)},
        {{
                 {"dpu_cfg", {{"workload_shad_odu", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.workload_shad_odu, 1)},
        {{
                 {"dpu_cfg", {{"workload_shad_idu", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.workload_shad_idu, 1)},
        {{
                 {"dpu_cfg", {{"workload_idu_auto_upd_0", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.workload_idu_auto_upd_0, 1)},
        {{
                 {"dpu_cfg", {{"workload_idu_auto_upd_1", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.workload_idu_auto_upd_1, 1)},
        {{
                 {"dpu_cfg", {{"workload_odu_auto_upd", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.workload_odu_auto_upd, 1)},
        {{
                 {"dpu_cfg", {{"cfg_Reserved_0", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_0, 1)},
        {{
                 {"dpu_cfg", {{"cfg_Reserved_1", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_1, 1)},
        {{
                 {"dpu_cfg", {{"cfg_Reserved_2", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_2, 1)},
        {{
                 {"dpu_cfg", {{"rst_ctxt_new", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.rst_ctxt_new, 1)},
        {{
                 {"dpu_cfg", {{"cfg_Reserved_3", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_3, 1)},
        {{
                 {"dpu_cfg", {{"cfg_Reserved_4", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_4, 1)},
        {{
                 {"dpu_cfg", {{"odu_stat_clr", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.odu_stat_clr, 1)},
        {{
                 {"dpu_cfg", {{"idu_stat_clr", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.idu_stat_clr, 1)},
        {{
                 {"dpu_cfg", {{"cfg_Reserved_5", 1}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_5, 1)},
        {{
                 {"dpu_cfg", {{"cfg_Reserved_6", 0x3FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_6, 0x3FFF)},
        // te_beg0 ---------------------------------------------------------------------
        {{
                 {"te_beg0", {{"te_beg_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.te_beg0.te_beg0_bf.te_beg_y, 0x1FFF)},
        {{
                 {"te_beg0", {{"te_beg_z", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.te_beg0.te_beg0_bf.te_beg_z, 0x1FFF)},
        // te_beg1 ---------------------------------------------------------------------
        {{
                 {"te_beg1", {{"te_beg_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.te_beg1.te_beg1_bf.te_beg_x, 0x1FFF)},
        // te_end0 ---------------------------------------------------------------------
        {{
                 {"te_end0", {{"te_end_y", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.te_end0.te_end0_bf.te_end_y, 0x1FFF)},
        {{
                 {"te_end0", {{"te_end_z", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.te_end0.te_end0_bf.te_end_z, 0x1FFF)},
        // te_end1 ---------------------------------------------------------------------
        {{
                 {"te_end1", {{"te_end_x", 0x1FFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.te_end1.te_end1_bf.te_end_x, 0x1FFF)},
        // weight_size ---------------------------------------------------------------------
        {{
                 {"weight_size", {{"weight_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.weight_size, 0xFFFFFFFF)},
        // weight_num ---------------------------------------------------------------------
        {{
                 {"weight_num", {{"weight_num", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.weight_num, 0xFFFFFFFF)},
        // weight_start ---------------------------------------------------------------------
        {{
                 {"weight_start", {{"weight_start", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(registers_.weight_start, 0xFFFFFFFF)},
        // invariant_ ---------------------------------------------------------------------
        {{
                 {"invariant_", {{"invariant_", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(invariant_, 0xFFFFFFFFFFFFFFFF)},
        // invariant_index_ ---------------------------------------------------------------------
        {{
                 {"invariant_index_", {{"invariant_index_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(invariant_index_, 0xFFFFFFFF)}};

INSTANTIATE_TEST_SUITE_P(NPUReg40XX_MappedRegs, NPUReg40XX_NpuDPUVariantTest, testing::ValuesIn(dpuVariantFieldSet));
