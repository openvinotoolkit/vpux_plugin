//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//
#ifndef VPUX_COMPILER_DIALECT_NPUReg40XX_TYPES
#define VPUX_COMPILER_DIALECT_NPUReg40XX_TYPES

include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.td"

//
// RegField Specialization
//

class NPUReg40XX_Version<int vMajor, int vMinor, int vPatch> : VPURegMapped_Version {
    let major = vMajor;
    let minor = vMinor;
    let patch = vPatch;
}

// MI Version that acts as baseline for NPU4 NNRT API
// All registers default to this baseline unless specified otherwise
def NPUReg40XX_BaselineVersion : NPUReg40XX_Version<11, 4, 0>;

class NPUReg40XX_RegFieldWrapper<string name = "", int width = 0, int pos = 0, string dataType = "UINT", VPURegMapped_Version version = NPUReg40XX_BaselineVersion> :
    VPURegMapped_RegFieldWrapper<NPUReg40XX_Dialect, name, width, pos, dataType, version> {

    }

//
// Register Specialization
//

class NPUReg40XX_RegisterWrapper<string name = "", int size = 0, int address = 0x00, list<string> regFieldTypes = [], int resetValue = 0x0, bit allowOverlap = 0> :
    VPURegMapped_RegisterWrapper<NPUReg40XX_Dialect, "vpux::NPUReg40XX::", name, size, address, regFieldTypes, resetValue, allowOverlap> {

}

//
// DPUInvariantRegisters
//

// cmx_slice0_low_addr
def bf_CmxSlice0LowAddr : NPUReg40XX_RegFieldWrapper<"cmx_slice0_low_addr", 32, 0>;
def reg_CmxSlice0LowAddr : NPUReg40XX_RegisterWrapper<"cmx_slice0_low_addr", 4, 0x0, ["cmx_slice0_low_addr"], 0x4000000>;

// cmx_slice1_low_addr
def bf_CmxSlice1LowAddr : NPUReg40XX_RegFieldWrapper<"cmx_slice1_low_addr", 32, 0>;
def reg_CmxSlice1LowAddr : NPUReg40XX_RegisterWrapper<"cmx_slice1_low_addr", 4, 0x4, ["cmx_slice1_low_addr"], 0x4000000>;

// cmx_slice2_low_addr
def bf_CmxSlice2LowAddr : NPUReg40XX_RegFieldWrapper<"cmx_slice2_low_addr", 32, 0>;
def reg_CmxSlice2LowAddr : NPUReg40XX_RegisterWrapper<"cmx_slice2_low_addr", 4, 0x8, ["cmx_slice2_low_addr"], 0x4000000>;

// cmx_slice3_low_addr
def bf_CmxSlice3LowAddr : NPUReg40XX_RegFieldWrapper<"cmx_slice3_low_addr", 32, 0>;
def reg_CmxSlice3LowAddr : NPUReg40XX_RegisterWrapper<"cmx_slice3_low_addr", 4, 0xC, ["cmx_slice3_low_addr"], 0x4000000>;

// cmx_slice_size
def bf_CmxSliceSize : NPUReg40XX_RegFieldWrapper<"cmx_slice_size", 32, 0>;
def reg_CmxSliceSize : NPUReg40XX_RegisterWrapper<"cmx_slice_size", 4, 0x10, ["cmx_slice_size"], 0x00018000>;

// se_addr
def bf_SeAddr : NPUReg40XX_RegFieldWrapper<"se_addr", 32, 0>;
def reg_SeAddr : NPUReg40XX_RegisterWrapper<"se_addr", 4, 0x14, ["se_addr"]>;

// sparsity_addr
def bf_SparsityAddr : NPUReg40XX_RegFieldWrapper<"sparsity_addr", 32, 0>;
def reg_SparsityAddr : NPUReg40XX_RegisterWrapper<"sparsity_addr", 4, 0x18, ["sparsity_addr"]>;

// se_sise
def bf_SeSize : NPUReg40XX_RegFieldWrapper<"se_size", 32, 0>;
def reg_SeSize : NPUReg40XX_RegisterWrapper<"se_size", 4, 0x1C, ["se_size"]>;

// z_config
def bf_SeZSplit : NPUReg40XX_RegFieldWrapper<"se_z_split", 4, 0>;
def bf_NumSesInZDir : NPUReg40XX_RegFieldWrapper<"num_ses_in_z_dir", 9, 4>;
def bf_CmSpPattern : NPUReg40XX_RegFieldWrapper<"cm_sp_pattern", 16, 13>;
def bf_Npo2SeZSplitEnable : NPUReg40XX_RegFieldWrapper<"npo2_se_z_split_en", 1, 29>;
def bf_Reserved : NPUReg40XX_RegFieldWrapper<"reserved", 1, 30>;
def bf_AddrFormatSel : NPUReg40XX_RegFieldWrapper<"addr_format_sel", 1, 31>;
def reg_ZConfig : NPUReg40XX_RegisterWrapper<"z_config", 4, 0x20,
    ["se_z_split", "num_ses_in_z_dir", "cm_sp_pattern", "npo2_se_z_split_en", "reserved", "addr_format_sel"], 0x0>;

// kernel_pad_cfg
def bf_MpeAssign : NPUReg40XX_RegFieldWrapper<"mpe_assign", 1, 0>;
def bf_PadRightEn : NPUReg40XX_RegFieldWrapper<"pad_right_en", 1, 1>;
def bf_PadLeftEn : NPUReg40XX_RegFieldWrapper<"pad_left_en", 1, 2>;
def bf_PadBottomEn : NPUReg40XX_RegFieldWrapper<"pad_bottom_en", 1, 3>;
def bf_PadTopEn : NPUReg40XX_RegFieldWrapper<"pad_top_en", 1, 4>;
def bf_KernelY : NPUReg40XX_RegFieldWrapper<"kernel_y", 4, 5>;
def bf_KernelX : NPUReg40XX_RegFieldWrapper<"kernel_x", 4, 9>;
def bf_WtPltCfg : NPUReg40XX_RegFieldWrapper<"wt_plt_cfg", 2, 13>;
def bf_ActDense : NPUReg40XX_RegFieldWrapper<"act_dense", 1, 15>;
def bf_WtDense : NPUReg40XX_RegFieldWrapper<"wt_dense", 1, 16>;
def bf_StrideYEn : NPUReg40XX_RegFieldWrapper<"stride_y_en", 1, 17>;
def bf_StrideY : NPUReg40XX_RegFieldWrapper<"stride_y", 3, 18>;
def bf_DynamicBwEn : NPUReg40XX_RegFieldWrapper<"dynamic_bw_en", 1, 21>;
def bf_DwWtSpIns : NPUReg40XX_RegFieldWrapper<"dw_wt_sp_ins", 1, 22>;
def bf_Layer1WtSpIns : NPUReg40XX_RegFieldWrapper<"layer1_wt_sp_ins", 1, 23>;
def bf_Layer1CmpEn : NPUReg40XX_RegFieldWrapper<"layer1_cmp_en", 1, 24>;
def bf_PoolOptEn : NPUReg40XX_RegFieldWrapper<"pool_opt_en", 1, 25>;
def bf_SpSeTblSegment : NPUReg40XX_RegFieldWrapper<"sp_se_tbl_segment", 1, 29>;
def bf_RstCtxt : NPUReg40XX_RegFieldWrapper<"rst_ctxt", 1, 30>;
def reg_KernelPadCfg : NPUReg40XX_RegisterWrapper<"kernel_pad_cfg", 4, 0x24,
    ["mpe_assign", "pad_right_en", "pad_left_en", "pad_bottom_en", "pad_top_en", "kernel_y", "kernel_x", "wt_plt_cfg",
    "act_dense", "wt_dense", "stride_y_en", "stride_y", "dynamic_bw_en", "dw_wt_sp_ins", "layer1_wt_sp_ins",
    "layer1_cmp_en", "pool_opt_en", "sp_se_tbl_segment", "rst_ctxt"]>;

// tensor_size0
def bf_TensorSizeX : NPUReg40XX_RegFieldWrapper<"tensor_size_x", 14, 0>;
def bf_TensorSizeY : NPUReg40XX_RegFieldWrapper<"tensor_size_y", 14, 14>;
def reg_TensorSize0 : NPUReg40XX_RegisterWrapper<"tensor_size0", 4, 0x28, ["tensor_size_x", "tensor_size_y"]>;

// tensor_size1
def bf_TensorSizeZ : NPUReg40XX_RegFieldWrapper<"tensor_size_z", 14, 0>;
def bf_Npo2_se_size : NPUReg40XX_RegFieldWrapper<"npo2_se_size",  9, 14>;
def reg_TensorSize1 : NPUReg40XX_RegisterWrapper<"tensor_size1", 4, 0x2c, ["tensor_size_z", "npo2_se_size"]>;

// tensor_start
def bf_TensorStart : NPUReg40XX_RegFieldWrapper<"tensor_start", 32, 0>;
def reg_TensorStart : NPUReg40XX_RegisterWrapper<"tensor_start", 4, 0x30, ["tensor_start"]>;

// tensor_mode
def bf_Wmode : NPUReg40XX_RegFieldWrapper<"wmode", 4, 0>;
def bf_Amode : NPUReg40XX_RegFieldWrapper<"amode", 4, 4>;
def bf_Stride : NPUReg40XX_RegFieldWrapper<"stride", 3, 8>;
def bf_ZmInput : NPUReg40XX_RegFieldWrapper<"zm_input", 1, 11>;
def bf_DwInput : NPUReg40XX_RegFieldWrapper<"dw_input", 1, 12>;
def bf_CmInput : NPUReg40XX_RegFieldWrapper<"cm_input", 1, 13>;
def bf_WorkloadOperation : NPUReg40XX_RegFieldWrapper<"workload_operation", 2, 14>;
def bf_PadValue : NPUReg40XX_RegFieldWrapper<"pad_value", 16, 16>;
def reg_TensorMode : NPUReg40XX_RegisterWrapper<"tensor_mode", 4, 0x34,
    ["wmode", "amode", "stride", "zm_input", "dw_input", "cm_input", "workload_operation", "pad_value"]>;

// elops_sparsity_addr
def bf_ElopsSparityAddr : NPUReg40XX_RegFieldWrapper<"elops_sparsity_addr", 32, 0>;
def reg_ElopsSparityAddr : NPUReg40XX_RegisterWrapper<"elops_sparsity_addr", 4, 0x38, ["elops_sparsity_addr"]>;

// elops_se_addr
def bf_ElopsSeAddr : NPUReg40XX_RegFieldWrapper<"elops_se_addr", 32, 0>;
def reg_ElopsSeAddr : NPUReg40XX_RegisterWrapper<"elops_se_addr", 4, 0x3C, ["elops_se_addr"]>;

// elops_wload
def bf_ElopWload : NPUReg40XX_RegFieldWrapper<"elop_wload", 1, 0>;
def bf_SeedWload : NPUReg40XX_RegFieldWrapper<"seed_wload", 1, 1>;
def bf_FifoWrWload : NPUReg40XX_RegFieldWrapper<"fifo_wr_wload", 1, 2>;
def bf_ElopWloadType : NPUReg40XX_RegFieldWrapper<"elop_wload_type", 1, 3>;
def bf_PoolWtData : NPUReg40XX_RegFieldWrapper<"pool_wt_data", 16, 4>;
def bf_PoolWtRdDis : NPUReg40XX_RegFieldWrapper<"pool_wt_rd_dis", 1, 26>;
def reg_ElopsWload : NPUReg40XX_RegisterWrapper<"elops_wload", 4, 0x40,
    ["elop_wload", "seed_wload", "fifo_wr_wload", "elop_wload_type", "pool_wt_data", "pool_wt_rd_dis"], 0x0>;

// act0_offset
def bf_Adr0Offset : NPUReg40XX_RegFieldWrapper<"act_offset0", 32, 0>;
def reg_Act0Offset : NPUReg40XX_RegisterWrapper<"act_offset0", 4, 0x44, ["act_offset0"]>;

// act1_offset
def bf_Adr1Offset : NPUReg40XX_RegFieldWrapper<"act_offset1", 32, 0>;
def reg_Act1Offset : NPUReg40XX_RegisterWrapper<"act_offset1", 4, 0x48, ["act_offset1"]>;

// act2_offset
def bf_Adr2Offset : NPUReg40XX_RegFieldWrapper<"act_offset2", 32, 0>;
def reg_Act2Offset : NPUReg40XX_RegisterWrapper<"act_offset2", 4, 0x4C, ["act_offset2"]>;

// act3_offset
def bf_Adr3Offset : NPUReg40XX_RegFieldWrapper<"act_offset3", 32, 0>;
def reg_Act3Offset : NPUReg40XX_RegisterWrapper<"act_offset3", 4, 0x50, ["act_offset3"]>;

// base_offset_a
def bf_BaseOffset0 : NPUReg40XX_RegFieldWrapper<"base_offset_a", 32, 0>;
def reg_BaseOffsetA : NPUReg40XX_RegisterWrapper<"base_offset_a", 4, 0x54, ["base_offset_a"]>;

// base_offset_b
def bf_BaseOffset2 : NPUReg40XX_RegFieldWrapper<"base_offset_2", 9, 0>;
def bf_BaseOffset3 : NPUReg40XX_RegFieldWrapper<"base_offset_3", 9, 9>;
def bf_DwOptOffset : NPUReg40XX_RegFieldWrapper<"dw_opt_offset", 6, 18>;
def bf_DwOptEn : NPUReg40XX_RegFieldWrapper<"dw_opt_en", 1, 24>;
def bf_Dw3x3s1OptDis : NPUReg40XX_RegFieldWrapper<"dw_3x3s1_opt_dis", 1, 25>;
def reg_BaseOffsetB : NPUReg40XX_RegisterWrapper<"base_offset_b", 4, 0x58,
    ["base_offset_2", "base_offset_3", "dw_opt_offset", "dw_opt_en", "dw_3x3s1_opt_dis"]>;

// wt_offset
def bf_WtOffset : NPUReg40XX_RegFieldWrapper<"wt_offset", 32, 0>;
def reg_WtOffset : NPUReg40XX_RegisterWrapper<"wt_offset", 4, 0x5C, ["wt_offset"]>;

// odu_cfg
def bf_Dtype : NPUReg40XX_RegFieldWrapper<"dtype", 3, 0>;
def bf_WcbAcMode : NPUReg40XX_RegFieldWrapper<"wcb_ac_mode", 1, 3>;
def bf_WcbSpMode : NPUReg40XX_RegFieldWrapper<"wcb_sp_mode", 1, 4>;
def bf_SpValue : NPUReg40XX_RegFieldWrapper<"sp_value", 8, 5>;
def bf_SpOutEn : NPUReg40XX_RegFieldWrapper<"sp_out_en", 1, 13>;
def bf_CmxPortMuxingDisable : NPUReg40XX_RegFieldWrapper<"cmx_port_muxing_disable", 1, 14>;
def bf_WriteSp : NPUReg40XX_RegFieldWrapper<"write_sp", 1, 15>;
def bf_WritePt : NPUReg40XX_RegFieldWrapper<"write_pt", 1, 16>;
def bf_WriteAc : NPUReg40XX_RegFieldWrapper<"write_ac", 1, 17>;
def bf_Mode : NPUReg40XX_RegFieldWrapper<"mode", 2, 18>;
def bf_Grid : NPUReg40XX_RegFieldWrapper<"grid", 1, 20>;
def bf_SwizzleKey : NPUReg40XX_RegFieldWrapper<"swizzle_key", 3, 21>;
def bf_WlBpOnStartEn : NPUReg40XX_RegFieldWrapper<"wl_bp_on_start_en", 1, 24>;
def bf_Nthw : NPUReg40XX_RegFieldWrapper<"nthw", 2, 25>;
def bf_Permutation : NPUReg40XX_RegFieldWrapper<"permutation", 3, 27>;
def bf_WcbStallAvoidance : NPUReg40XX_RegFieldWrapper<"wcb_stall_avoidance", 1, 30>;
def bf_WcbBypass : NPUReg40XX_RegFieldWrapper<"wcb_bypass", 1, 31>;
def reg_OduCfg : NPUReg40XX_RegisterWrapper<"odu_cfg", 4, 0x60,
    ["dtype", "wcb_ac_mode", "wcb_sp_mode", "sp_value", "sp_out_en", "cmx_port_muxing_disable", "write_sp", "write_pt", "write_ac", "mode", "grid",
    "swizzle_key", "wl_bp_on_start_en", "nthw", "permutation", "wcb_stall_avoidance", "wcb_bypass"]>;

// odu_be_size
def bf_OduBeSize : NPUReg40XX_RegFieldWrapper<"odu_be_size", 32, 0>;
def reg_OduBeSize : NPUReg40XX_RegisterWrapper<"odu_be_size", 4, 0x64, ["odu_be_size"]>;

// odu_be_cnt
def bf_OduBeCnt : NPUReg40XX_RegFieldWrapper<"odu_be_cnt", 32, 0>;
def reg_OduBeCnt : NPUReg40XX_RegisterWrapper<"odu_be_cnt", 4, 0x68, ["odu_be_cnt"]>;

// odu_se_size
def bf_OduSeSize : NPUReg40XX_RegFieldWrapper<"odu_se_size", 32, 0>;
def reg_OduSeSize : NPUReg40XX_RegisterWrapper<"odu_se_size", 4, 0x6C, ["odu_se_size"]>;

// te_dim0
def bf_TeDimY : NPUReg40XX_RegFieldWrapper<"te_dim_y", 13, 0>;
def bf_TeDimZ : NPUReg40XX_RegFieldWrapper<"te_dim_z", 13, 13>;
def reg_TeDim0 : NPUReg40XX_RegisterWrapper<"te_dim0", 4, 0x70, ["te_dim_y", "te_dim_z"]>;

// te_dim1
def bf_TeDimX : NPUReg40XX_RegFieldWrapper<"te_dim_x", 13, 0>;
def reg_TeDim1 : NPUReg40XX_RegisterWrapper<"te_dim1", 4, 0x74, ["te_dim_x"]>;

// pt_base
def bf_PtBase : NPUReg40XX_RegFieldWrapper<"pt_base", 32, 0>;
def reg_PtBase : NPUReg40XX_RegisterWrapper<"pt_base", 4, 0x78, ["pt_base"]>;

// sp_base
def bf_SpBase : NPUReg40XX_RegFieldWrapper<"sp_base", 32, 0>;
def reg_SpBase : NPUReg40XX_RegisterWrapper<"sp_base", 4, 0x7C, ["sp_base"]>;

// mpe_cfg
def bf_MpeWtbias : NPUReg40XX_RegFieldWrapper<"mpe_wtbias", 8, 0, "UINT">;   // TODO E#80099
def bf_MpeActbias : NPUReg40XX_RegFieldWrapper<"mpe_actbias", 8, 8, "UINT">; // TODO E#80099
def bf_MpeMode : NPUReg40XX_RegFieldWrapper<"mpe_mode", 3, 16>;
def bf_MpeDense : NPUReg40XX_RegFieldWrapper<"mpe_dense", 1, 19>;
def bf_MrmWeightDense : NPUReg40XX_RegFieldWrapper<"mrm_weight_dense", 1, 20>;
def bf_MrmActDense : NPUReg40XX_RegFieldWrapper<"mrm_act_dense", 1, 21>;
def bf_MpeDaz : NPUReg40XX_RegFieldWrapper<"mpe_daz", 1, 22>;
def bf_MpeFtz : NPUReg40XX_RegFieldWrapper<"mpe_ftz", 1, 23>;
def reg_MpeCfg : NPUReg40XX_RegisterWrapper<"mpe_cfg", 4, 0x80,
    ["mpe_wtbias", "mpe_actbias", "mpe_mode", "mpe_dense", "mrm_weight_dense", "mrm_act_dense", "mpe_daz", "mpe_ftz"]>;

// mpe_bus_data_sel
def bf_MpeBusDataSel : NPUReg40XX_RegFieldWrapper<"mpe_bus_data_sel", 32, 0>;
def reg_MpeBusDataSel : NPUReg40XX_RegisterWrapper<"mpe_bus_data_sel", 4, 0x84, ["mpe_bus_data_sel"]>;

// elop_scale
def bf_ElopScaleB : NPUReg40XX_RegFieldWrapper<"elop_scale_b", 16, 0>;
def bf_ElopScaleA : NPUReg40XX_RegFieldWrapper<"elop_scale_a", 16, 16>;
def reg_ElopScale : NPUReg40XX_RegisterWrapper<"elop_scale", 4, 0x88, ["elop_scale_b", "elop_scale_a"]>;

// ppe_cfg
def bf_PpeG8BiasC : NPUReg40XX_RegFieldWrapper<"ppe_g8_bias_c", 9, 0, "SINT">;
def bf_PpeG8BiasB : NPUReg40XX_RegFieldWrapper<"ppe_g8_bias_b", 9, 9>;
def bf_PpeG8BiasA : NPUReg40XX_RegFieldWrapper<"ppe_g8_bias_a", 9, 18>;
def reg_PpeCfg : NPUReg40XX_RegisterWrapper<"ppe_cfg", 4, 0x8C, ["ppe_g8_bias_c", "ppe_g8_bias_b", "ppe_g8_bias_a"]>;

// ppe_bias
def bf_PpeBias : NPUReg40XX_RegFieldWrapper<"ppe_bias", 32, 0, "SINT">;
def reg_PpeBias : NPUReg40XX_RegisterWrapper<"ppe_bias", 4, 0x90, ["ppe_bias"]>;

// ppe_scale
def bf_PpeScaleShift : NPUReg40XX_RegFieldWrapper<"ppe_scale_shift", 6, 2>;
def bf_PpeScaleRound : NPUReg40XX_RegFieldWrapper<"ppe_scale_round", 2, 10>;
def bf_PpeScaleMult : NPUReg40XX_RegFieldWrapper<"ppe_scale_mult", 16, 16, "SINT">;
def reg_PpeScale : NPUReg40XX_RegisterWrapper<"ppe_scale", 4, 0x94, ["ppe_scale_shift", "ppe_scale_round", "ppe_scale_mult"], 0x00000c00>;

// ppe_scale_ctrl
def bf_PpeScaleOverride : NPUReg40XX_RegFieldWrapper<"ppe_scale_override", 1, 0>;
def bf_PpeFpScaleOverride : NPUReg40XX_RegFieldWrapper<"ppe_fp_scale_override", 1, 1>;
def reg_PpeScaleCtrl : NPUReg40XX_RegisterWrapper<"ppe_scale_ctrl", 4, 0x98, ["ppe_scale_override", "ppe_fp_scale_override"], 0x0>;

// ppe_prelu
def bf_PpePreluShift : NPUReg40XX_RegFieldWrapper<"ppe_prelu_shift", 5, 8>;
def bf_PpePreluMult : NPUReg40XX_RegFieldWrapper<"ppe_prelu_mult", 11, 16>;
def reg_PpePrelu : NPUReg40XX_RegisterWrapper<"ppe_prelu", 4, 0x9C, ["ppe_prelu_shift", "ppe_prelu_mult"], 0x00010000>;

// ppe_scale_hclamp
def bf_PpeScaleHclamp : NPUReg40XX_RegFieldWrapper<"ppe_scale_hclamp", 32, 0, "SINT">;
def reg_PpeScaleHclamp : NPUReg40XX_RegisterWrapper<"ppe_scale_hclamp", 4, 0xA0, ["ppe_scale_hclamp"], 0x7fffffff>;

// ppe_scale_lclamp
def bf_PpeScaleLclamp : NPUReg40XX_RegFieldWrapper<"ppe_scale_lclamp", 32, 0, "SINT">;
def reg_PpeScaleLclamp : NPUReg40XX_RegisterWrapper<"ppe_scale_lclamp", 4, 0xA4, ["ppe_scale_lclamp"], 0x80000000>;

// ppe_misc
def bf_PpeFp16Ftz : NPUReg40XX_RegFieldWrapper<"ppe_fp16_ftz", 1, 6>;
def bf_PpeFp16Clamp : NPUReg40XX_RegFieldWrapper<"ppe_fp16_clamp", 1, 7>;
def bf_PpeI32Convert : NPUReg40XX_RegFieldWrapper<"ppe_i32_convert", 2, 8>;
def reg_PpeMisc : NPUReg40XX_RegisterWrapper<"ppe_misc", 4, 0xA8, ["ppe_fp16_ftz", "ppe_fp16_clamp", "ppe_i32_convert"]>;

// ppe_fp_bias
def bf_PpeFpBias : NPUReg40XX_RegFieldWrapper<"ppe_fp_bias", 32, 0, "FP">;
def reg_PpeFpBias : NPUReg40XX_RegisterWrapper<"ppe_fp_bias", 4, 0xAC, ["ppe_fp_bias"]>;

// ppe_fp_scale
def bf_PpeFpScale : NPUReg40XX_RegFieldWrapper<"ppe_fp_scale", 32, 0, "FP">;
def reg_PpeFpScale : NPUReg40XX_RegisterWrapper<"ppe_fp_scale", 4, 0xB0, ["ppe_fp_scale"]>;

// ppe_fp_prelu
def bf_PpeFpPrelu : NPUReg40XX_RegFieldWrapper<"ppe_fp_prelu", 32, 0, "FP">;
def reg_PpeFpPrelu : NPUReg40XX_RegisterWrapper<"ppe_fp_prelu", 4, 0xB4, ["ppe_fp_prelu"]>;

// ppe_fp_cfg
def bf_PpeFpConvert : NPUReg40XX_RegFieldWrapper<"ppe_fp_convert", 3, 0>;
def bf_PpeFpBypass : NPUReg40XX_RegFieldWrapper<"ppe_fp_bypass", 1, 3>;
def bf_PpeBf16Round : NPUReg40XX_RegFieldWrapper<"ppe_bf16_round", 1, 4>;
def bf_PpeFpPreluEn : NPUReg40XX_RegFieldWrapper<"ppe_fp_prelu_en", 1, 5>;
//  Note: ppe prelu needs to be off by default, only turned on explicitly
def reg_PpeFpCfg : NPUReg40XX_RegisterWrapper<"ppe_fp_cfg", 4, 0xB8, ["ppe_fp_convert", "ppe_fp_bypass", "ppe_bf16_round", "ppe_fp_prelu_en"], 0x00000008>;

// odu_ac_base
def bf_AcBase : NPUReg40XX_RegFieldWrapper<"ac_base", 28, 4>;
def reg_OduAcBase : NPUReg40XX_RegisterWrapper<"odu_ac_base", 4, 0xBC, ["ac_base"]>;

// hwp_ctrl
def bf_HwpEn : NPUReg40XX_RegFieldWrapper<"hwp_en", 1, 0>;
def bf_HwpStatMode : NPUReg40XX_RegFieldWrapper<"hwp_stat_mode", 3, 1>;
def bf_HwpTimerEn : NPUReg40XX_RegFieldWrapper<"local_timer_en", 1, 4>;
def bf_HwpTimerRst : NPUReg40XX_RegFieldWrapper<"local_timer_rst", 1, 5>;
// rsvd : 9
def bf_HwpUniqueID : NPUReg40XX_RegFieldWrapper<"unique_ID", 16, 16>;

def reg_HwpCtrl : NPUReg40XX_RegisterWrapper<"hwp_ctrl", 4, 0xC0, ["hwp_en", "hwp_stat_mode", "local_timer_en", "local_timer_rst", "unique_ID"]>;

// hwp_cmx_mem_addr
def bf_HwpCmxMemAddr : NPUReg40XX_RegFieldWrapper<"hwp_cmx_mem_addr", 32, 0>;
def reg_HwpCmxMemAddr : NPUReg40XX_RegisterWrapper<"hwp_cmx_mem_addr", 4, 0xC4, ["hwp_cmx_mem_addr"]>;

// odu_cast_0
def bf_CastEnable0 : NPUReg40XX_RegFieldWrapper<"cast_enable0", 1, 0>;
def bf_CastOffset0 : NPUReg40XX_RegFieldWrapper<"cast_offset0", 28, 4>;
def reg_OduCast0 : NPUReg40XX_RegisterWrapper<"odu_cast0", 4, 0xC8, ["cast_enable0", "cast_offset0"]>;

// odu_cast_1
def bf_CastEnable1 : NPUReg40XX_RegFieldWrapper<"cast_enable1", 1, 0>;
def bf_CastOffset1 : NPUReg40XX_RegFieldWrapper<"cast_offset1", 28, 4>;
def reg_OduCast1 : NPUReg40XX_RegisterWrapper<"odu_cast1", 4, 0xCC, ["cast_enable1", "cast_offset1"]>;

// odu_cast_2
def bf_CastEnable2 : NPUReg40XX_RegFieldWrapper<"cast_enable2", 1, 0>;
def bf_CastOffset2 : NPUReg40XX_RegFieldWrapper<"cast_offset2", 28, 4>;
def reg_OduCast2 : NPUReg40XX_RegisterWrapper<"odu_cast2", 4, 0xD0, ["cast_enable2", "cast_offset2"]>;

// wl_tag
def bf_NvarTag : NPUReg40XX_RegFieldWrapper<"nvar_tag", 32, 0>;
def reg_NvarTag : NPUReg40XX_RegisterWrapper<"nvar_tag", 4, 0xDC, ["nvar_tag"]>;

// pallet

// pallet tb 0
def bf_PalletIndx0 : NPUReg40XX_RegFieldWrapper<"plt_idx_0", 16, 0>;
def bf_PalletIndx1 : NPUReg40XX_RegFieldWrapper<"plt_idx_1", 16, 16>;
def reg_PalletTbl0 : NPUReg40XX_RegisterWrapper<"pallet0", 4, 0xE0, ["plt_idx_0", "plt_idx_1"]>;

// pallet tb 1
def bf_PalletIndx2 : NPUReg40XX_RegFieldWrapper<"plt_idx_2", 16, 0>;
def bf_PalletIndx3 : NPUReg40XX_RegFieldWrapper<"plt_idx_3", 16, 16>;
def reg_PalletTbl1 : NPUReg40XX_RegisterWrapper<"pallet1", 4, 0xE4, ["plt_idx_2", "plt_idx_3"]>;

// pallet tb 2
def bf_PalletIndx4 : NPUReg40XX_RegFieldWrapper<"plt_idx_4", 16, 0>;
def bf_PalletIndx5 : NPUReg40XX_RegFieldWrapper<"plt_idx_5", 16, 16>;
def reg_PalletTbl2 : NPUReg40XX_RegisterWrapper<"pallet2", 4, 0xE8, ["plt_idx_4", "plt_idx_5"]>;

// pallet tb 3
def bf_PalletIndx6 : NPUReg40XX_RegFieldWrapper<"plt_idx_6", 16, 0>;
def bf_PalletIndx7 : NPUReg40XX_RegFieldWrapper<"plt_idx_7", 16, 16>;
def reg_PalletTbl3 : NPUReg40XX_RegisterWrapper<"pallet3", 4, 0xEC, ["plt_idx_6", "plt_idx_7"]>;

// pallet tb 4
def bf_PalletIndx8 : NPUReg40XX_RegFieldWrapper<"plt_idx_8", 16, 0>;
def bf_PalletIndx9 : NPUReg40XX_RegFieldWrapper<"plt_idx_9", 16, 16>;
def reg_PalletTbl4 : NPUReg40XX_RegisterWrapper<"pallet4", 4, 0xF0, ["plt_idx_8", "plt_idx_9"]>;

// pallet tb 5
def bf_PalletIndx10 : NPUReg40XX_RegFieldWrapper<"plt_idx_10", 16, 0>;
def bf_PalletIndx11 : NPUReg40XX_RegFieldWrapper<"plt_idx_11", 16, 16>;
def reg_PalletTbl5 : NPUReg40XX_RegisterWrapper<"pallet5", 4, 0xF4, ["plt_idx_10", "plt_idx_11"]>;

// pallet tb 6
def bf_PalletIndx12 : NPUReg40XX_RegFieldWrapper<"plt_idx_12", 16, 0>;
def bf_PalletIndx13 : NPUReg40XX_RegFieldWrapper<"plt_idx_13", 16, 16>;
def reg_PalletTbl6 : NPUReg40XX_RegisterWrapper<"pallet6", 4, 0xF8, ["plt_idx_12", "plt_idx_13"]>;

// pallet tb 7
def bf_PalletIndx14 : NPUReg40XX_RegFieldWrapper<"plt_idx_14", 16, 0>;
def bf_PalletIndx15 : NPUReg40XX_RegFieldWrapper<"plt_idx_15", 16, 16>;
def reg_PalletTbl7 : NPUReg40XX_RegisterWrapper<"pallet7", 4, 0xFC, ["plt_idx_14", "plt_idx_15"]>;

// se_addr1
def bf_SeAddr1 : NPUReg40XX_RegFieldWrapper<"se_addr1", 32, 0>;
def reg_SeAddr1 : NPUReg40XX_RegisterWrapper<"se_addr1", 4, 0x100, ["se_addr1"]>;

// sparsity_addr1
def bf_SparsityAddr1 : NPUReg40XX_RegFieldWrapper<"sparsity_addr1", 32, 0>;
def reg_SparsityAddr1 : NPUReg40XX_RegisterWrapper<"sparsity_addr1", 4, 0x104, ["sparsity_addr1"]>;

// se_addr2
def bf_SeAddr2 : NPUReg40XX_RegFieldWrapper<"se_addr2", 32, 0>;
def reg_SeAddr2 : NPUReg40XX_RegisterWrapper<"se_addr2", 4, 0x108, ["se_addr2"]>;

// sparsity_addr2
def bf_SparsityAddr2 : NPUReg40XX_RegFieldWrapper<"sparsity_addr2", 32, 0>;
def reg_SparsityAddr2 : NPUReg40XX_RegisterWrapper<"sparsity_addr2", 4, 0x10C, ["sparsity_addr2"]>;

// se_addr3
def bf_SeAddr3 : NPUReg40XX_RegFieldWrapper<"se_addr3", 32, 0>;
def reg_SeAddr3 : NPUReg40XX_RegisterWrapper<"se_addr3", 4, 0x110, ["se_addr3"]>;

// sparsity_addr3
def bf_SparsityAddr3 : NPUReg40XX_RegFieldWrapper<"sparsity_addr3", 32, 0>;
def reg_SparsityAddr3 : NPUReg40XX_RegisterWrapper<"sparsity_addr3", 4, 0x114, ["sparsity_addr3"]>;

// se_sp_size1
def bf_SpSegSize1 : NPUReg40XX_RegFieldWrapper<"se_sp_size1", 32, 0>;
def reg_SeSpSize1 : NPUReg40XX_RegisterWrapper<"se_sp_size1", 4, 0x118, ["se_sp_size1"]>;

// se_sp_size2
def bf_SpSegSize2 : NPUReg40XX_RegFieldWrapper<"se_sp_size2", 32, 0>;
def reg_SeSpSize2 : NPUReg40XX_RegisterWrapper<"se_sp_size2", 4, 0x11C, ["se_sp_size2"]>;

def bf_BarrierWaitMaskHi : NPUReg40XX_RegFieldWrapper<"barriers_wait_mask_hi_", 32, 0>;
def reg_BarrierWaitMaskHi : NPUReg40XX_RegisterWrapper<"barriers_wait_mask_hi_", 8, 0x120, ["barriers_wait_mask_hi_"]>;

def bf_BarrierWaitMaskLo : NPUReg40XX_RegFieldWrapper<"barriers_wait_mask_lo_", 64, 0>;
def reg_BarrierWaitMaskLo : NPUReg40XX_RegisterWrapper<"barriers_wait_mask_lo_", 8, 0x128, ["barriers_wait_mask_lo_"]>;

def bf_BarrierWaitPostHi : NPUReg40XX_RegFieldWrapper<"barriers_post_mask_hi_", 32, 0>;
def reg_BarrierWaitPostHi : NPUReg40XX_RegisterWrapper<"barriers_post_mask_hi_", 8, 0x130, ["barriers_post_mask_hi_"]>;

def bf_BarrierWaitPostLo : NPUReg40XX_RegFieldWrapper<"barriers_post_mask_lo_", 64, 0>;
def reg_BarrierWaitPostLo : NPUReg40XX_RegisterWrapper<"barriers_post_mask_lo_", 8, 0x138, ["barriers_post_mask_lo_"]>;

def bf_BarrierGroup : NPUReg40XX_RegFieldWrapper<"group_", 8, 0>;
def bf_BarrierMask : NPUReg40XX_RegFieldWrapper<"mask_", 8, 8>;
def reg_BarrierGroupMask : NPUReg40XX_RegisterWrapper<"barriers_group_mask_", 8, 0x140, ["group_", "mask_"]>;

def bf_StartAfter : NPUReg40XX_RegFieldWrapper<"start_after_", 32, 0>;
def bf_CleanAfter : NPUReg40XX_RegFieldWrapper<"clean_after_", 32, 32>;
def reg_BarriersSchedulingCfg : NPUReg40XX_RegisterWrapper<"barriers_sched_", 8, 0x148, ["start_after_", "clean_after_"]>;

def bf_ReservedInvariant : NPUReg40XX_RegFieldWrapper<"reserved_inv", 64, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_ReservedInvariant : NPUReg40XX_RegisterWrapper<"reserved_inv", 8, 0x150, ["reserved_inv"]>;

def bf_VariantCnt : NPUReg40XX_RegFieldWrapper<"variant_count_", 16, 0>;
def reg_VariantCnt : NPUReg40XX_RegisterWrapper<"variant_count_", 2, 0x158, ["variant_count_"]>;

def bf_ClusterInvariant : NPUReg40XX_RegFieldWrapper<"cluster_invariant_", 8, 0>;
def reg_ClusterInvariant : NPUReg40XX_RegisterWrapper<"cluster_invariant_", 1, 0x15A, ["cluster_invariant_"]>;

def bf_Pad3 : NPUReg40XX_RegFieldWrapper<"pad_3", 40, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_Pad3 : NPUReg40XX_RegisterWrapper<"pad_3", 5, 0x15B, ["pad_3"]>;

//
// DPUVariantRegisters
//

// invar_ptr
def bf_InvarPtr : NPUReg40XX_RegFieldWrapper<"invar_ptr", 16, 0>;
def bf_VarTag : NPUReg40XX_RegFieldWrapper<"var_tag", 16, 16>;
def reg_InvarPtr : NPUReg40XX_RegisterWrapper<"invar_ptr", 4, 0x0, ["invar_ptr", "var_tag"]>;

// workload_size0
def bf_WorkloadSizeX : NPUReg40XX_RegFieldWrapper<"workload_size_x", 14, 0>;
def bf_WorkloadSizeY : NPUReg40XX_RegFieldWrapper<"workload_size_y", 14, 14>;
def reg_WorkloadSize0 : NPUReg40XX_RegisterWrapper<"workload_size0", 4, 0x4, ["workload_size_x", "workload_size_y"]>;

// workload_size1
def bf_WorkloadSizeZ : NPUReg40XX_RegFieldWrapper<"workload_size_z", 14, 0>;
def bf_PadCountUp : NPUReg40XX_RegFieldWrapper<"pad_count_up", 3, 14>;
def bf_PadCountLeft : NPUReg40XX_RegFieldWrapper<"pad_count_left", 3, 17>;
def bf_PadCountDown : NPUReg40XX_RegFieldWrapper<"pad_count_down", 3, 20>;
def bf_PadCountRight : NPUReg40XX_RegFieldWrapper<"pad_count_right", 3, 23>;
def reg_WorkloadSize1 : NPUReg40XX_RegisterWrapper<"workload_size1", 4, 0x8,
    ["workload_size_z", "pad_count_up", "pad_count_left", "pad_count_down", "pad_count_right"]>;

// workload_start0
def bf_WorkloadStartX : NPUReg40XX_RegFieldWrapper<"workload_start_x", 14, 0>;
def bf_WorkloadStartY : NPUReg40XX_RegFieldWrapper<"workload_start_y", 14, 14>;
def reg_WorkloadStart0 : NPUReg40XX_RegisterWrapper<"workload_start0", 4, 0xC, ["workload_start_x", "workload_start_y"]>;

// workload_start1
def bf_WorkloadStartZ : NPUReg40XX_RegFieldWrapper<"workload_start_z", 14, 0>;
def reg_WorkloadStart1 : NPUReg40XX_RegisterWrapper<"workload_start1", 4, 0x10, ["workload_start_z"]>;

// offset_addr
def bf_NthwNtk : NPUReg40XX_RegFieldWrapper<"nthw_ntk", 2, 0>;
def bf_BinCfg : NPUReg40XX_RegFieldWrapper<"bin_cfg", 1, 2>;
def bf_ConvCond : NPUReg40XX_RegFieldWrapper<"conv_cond", 1, 3>;
def bf_DenseSe : NPUReg40XX_RegFieldWrapper<"dense_se", 1, 4>;
def bf_IdxQuad : NPUReg40XX_RegFieldWrapper<"idx_quad", 1, 5>;
def bf_SwizzleKeyOffsetAddr : NPUReg40XX_RegFieldWrapper<"swizzle_key_offset", 3, 6>;
def bf_IduMrmClkEn : NPUReg40XX_RegFieldWrapper<"idu_mrm_clk_en", 1, 9>;
def bf_OduClkEn : NPUReg40XX_RegFieldWrapper<"odu_clk_en", 1, 10>;
def bf_MpeClkEn : NPUReg40XX_RegFieldWrapper<"mpe_clk_en", 1, 11>;
def bf_PpeClkEn : NPUReg40XX_RegFieldWrapper<"ppe_clk_en", 1, 12>;
def bf_OduStatEn : NPUReg40XX_RegFieldWrapper<"odu_stat_en", 1, 13>;
def bf_IduStatEn : NPUReg40XX_RegFieldWrapper<"idu_stat_en", 1, 14>;
def bf_OduStatClrMode : NPUReg40XX_RegFieldWrapper<"odu_stat_clr_mode", 1, 16>;
def bf_IduStatClrMode : NPUReg40XX_RegFieldWrapper<"idu_stat_clr_mode", 1, 17>;
def bf_ShaveL2CacheEn : NPUReg40XX_RegFieldWrapper<"shave_l2_cache_en", 1, 19>;
def bf_IduDbgEn : NPUReg40XX_RegFieldWrapper<"idu_dbg_en", 2, 20>;
def bf_WtSwizzleKey : NPUReg40XX_RegFieldWrapper<"wt_swizzle_key", 3, 27>;
def bf_WtSwizzleSel : NPUReg40XX_RegFieldWrapper<"wt_swizzle_sel", 1, 30>;
//  Note: wt_swizzle_sel needs to be on by default to match NNRT GF behaviour
def reg_OffsetAddr : NPUReg40XX_RegisterWrapper<"offset_addr", 4, 0x14,
    ["nthw_ntk", "bin_cfg", "conv_cond", "dense_se", "idx_quad", "swizzle_key_offset", "idu_mrm_clk_en", "odu_clk_en",
    "mpe_clk_en", "ppe_clk_en", "odu_stat_en", "idu_stat_en", "odu_stat_clr_mode", "idu_stat_clr_mode", "shave_l2_cache_en",
    "idu_dbg_en", "wt_swizzle_key", "wt_swizzle_sel"], 0x40000000>;

// hwp_wload_id
def bf_HwpWLoadId : NPUReg40XX_RegFieldWrapper<"hwp_wload_id", 16, 0>;
def reg_HwpWloadId : NPUReg40XX_RegisterWrapper<"hwp_wload_id", 4, 0x18, ["hwp_wload_id"]>;

// var_cfg
def bf_InvarLineCntEn : NPUReg40XX_RegFieldWrapper<"invar_line_cnt_en", 1, 9>;
def bf_InvarLineCnt : NPUReg40XX_RegFieldWrapper<"invar_line_cnt", 4, 10>;
def bf_InvarPtrForce : NPUReg40XX_RegFieldWrapper<"invar_lptr_force", 1, 14>;
def bf_NextSramJobValid : NPUReg40XX_RegFieldWrapper<"next_sram_job_valid", 1, 15>;
def bf_NextSramJobAddr : NPUReg40XX_RegFieldWrapper<"next_sram_job_addr", 16, 16>;
def reg_VarCfg : NPUReg40XX_RegisterWrapper<"var_cfg", 4, 0x1C,
        ["invar_line_cnt_en", "invar_line_cnt", "invar_lptr_force", "next_sram_job_valid", "next_sram_job_addr"]>;

// cbarrier_lo
def bf_CBarrierLo : NPUReg40XX_RegFieldWrapper<"cbarrier_lo", 64, 0>;
def reg_CBarrierLo : NPUReg40XX_RegisterWrapper<"cbarrier_lo", 8, 0x20, ["cbarrier_lo"]>;

// cbarrier_hi
def bf_CBarrierHi : NPUReg40XX_RegFieldWrapper<"cbarrier_hi", 32, 0>;
def reg_CBarrierHi : NPUReg40XX_RegisterWrapper<"cbarrier_hi", 8, 0x28, ["cbarrier_hi"]>;

// pbarrier_lo
def bf_PBarrierLo : NPUReg40XX_RegFieldWrapper<"pbarrier_lo", 64, 0>;
def reg_PBarrierLo : NPUReg40XX_RegisterWrapper<"pbarrier_lo", 8, 0x30, ["pbarrier_lo"]>;

// pbarrier_hi
def bf_PBarrierHi : NPUReg40XX_RegFieldWrapper<"pbarrier_hi", 32, 0>;
def reg_PBarrierHi : NPUReg40XX_RegisterWrapper<"pbarrier_hi", 8, 0x38, ["pbarrier_hi"]>;

// halo_region0..5A
def bf_SpAdrOffset : NPUReg40XX_RegFieldWrapper<"sp_adr_offset", 22, 0, "SINT">;
def bf_TileSelected : NPUReg40XX_RegFieldWrapper<"tile_select", 7, 22>;
def bf_RsvdA : NPUReg40XX_RegFieldWrapper<"rsvdA", 2, 29>;
def bf_Enable : NPUReg40XX_RegFieldWrapper<"enable", 1, 31>;
def reg_HaloRegion0A : NPUReg40XX_RegisterWrapper<"halo_region0A", 4, 0x40, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion1A : NPUReg40XX_RegisterWrapper<"halo_region1A", 4, 0x50, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion2A : NPUReg40XX_RegisterWrapper<"halo_region2A", 4, 0x60, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion3A : NPUReg40XX_RegisterWrapper<"halo_region3A", 4, 0x70, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion4A : NPUReg40XX_RegisterWrapper<"halo_region4A", 4, 0x80, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion5A : NPUReg40XX_RegisterWrapper<"halo_region5A", 4, 0x90, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;

// halo_region0..5B
def bf_AcAdrOffset : NPUReg40XX_RegFieldWrapper<"ac_adr_offset", 22, 0, "SINT">;
def bf_TargetWidthLsb : NPUReg40XX_RegFieldWrapper<"target_width_lsb", 10, 22>;
def reg_HaloRegion0B : NPUReg40XX_RegisterWrapper<"halo_region0B", 4, 0x44, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion1B : NPUReg40XX_RegisterWrapper<"halo_region1B", 4, 0x54, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion2B : NPUReg40XX_RegisterWrapper<"halo_region2B", 4, 0x64, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion3B : NPUReg40XX_RegisterWrapper<"halo_region3B", 4, 0x74, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion4B : NPUReg40XX_RegisterWrapper<"halo_region4B", 4, 0x84, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion5B : NPUReg40XX_RegisterWrapper<"halo_region5B", 4, 0x94, ["ac_adr_offset", "target_width_lsb"]>;

// halo_region0..5C
def bf_BeginX : NPUReg40XX_RegFieldWrapper<"begin_x", 13, 0>;
def bf_BeginY : NPUReg40XX_RegFieldWrapper<"begin_y", 13, 13>;
def bf_TargetWidthMsb : NPUReg40XX_RegFieldWrapper<"target_width_msb", 4, 26>;
def reg_HaloRegion0C : NPUReg40XX_RegisterWrapper<"halo_region0C", 4, 0x48, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion1C : NPUReg40XX_RegisterWrapper<"halo_region1C", 4, 0x58, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion2C : NPUReg40XX_RegisterWrapper<"halo_region2C", 4, 0x68, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion3C : NPUReg40XX_RegisterWrapper<"halo_region3C", 4, 0x78, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion4C : NPUReg40XX_RegisterWrapper<"halo_region4C", 4, 0x88, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion5C : NPUReg40XX_RegisterWrapper<"halo_region5C", 4, 0x98, ["begin_x", "begin_y", "target_width_msb"]>;

// halo_region0..5D
def bf_EndX : NPUReg40XX_RegFieldWrapper<"end_x", 13, 0>;
def bf_EndY : NPUReg40XX_RegFieldWrapper<"end_y", 13, 13>;
def bf_RsvdD : NPUReg40XX_RegFieldWrapper<"rsvdD", 4, 26>;
def reg_HaloRegion0D : NPUReg40XX_RegisterWrapper<"halo_region0D", 4, 0x4C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion1D : NPUReg40XX_RegisterWrapper<"halo_region1D", 4, 0x5C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion2D : NPUReg40XX_RegisterWrapper<"halo_region2D", 4, 0x6C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion3D : NPUReg40XX_RegisterWrapper<"halo_region3D", 4, 0x7C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion4D : NPUReg40XX_RegisterWrapper<"halo_region4D", 4, 0x8C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion5D : NPUReg40XX_RegisterWrapper<"halo_region5D", 4, 0x9C, ["end_x", "end_y", "rsvdD"]>;

// dpu_cfg
def bf_WorkloadStartOdu : NPUReg40XX_RegFieldWrapper<"workload_start_odu", 1, 0>;
def bf_WorkloadStartIdu : NPUReg40XX_RegFieldWrapper<"workload_start_idu", 1, 1>;
def bf_WorkloadPrmSel : NPUReg40XX_RegFieldWrapper<"workload_prm_sel", 1, 2>;
def bf_WorkloadValid : NPUReg40XX_RegFieldWrapper<"workload_valid", 1, 3>;
def bf_WorkloadShadOdu : NPUReg40XX_RegFieldWrapper<"workload_shad_odu", 1, 4>;
def bf_WorkloadShadIdu : NPUReg40XX_RegFieldWrapper<"workload_shad_idu", 1, 5>;
def bf_WorkloadIduAutoUpd0 : NPUReg40XX_RegFieldWrapper<"workload_idu_auto_upd_0", 1, 6>;
def bf_WorkloadIduAutoUpd1 : NPUReg40XX_RegFieldWrapper<"workload_idu_auto_upd_1", 1, 7>;
def bf_WorkloadOduAutoUpd : NPUReg40XX_RegFieldWrapper<"workload_odu_auto_upd", 1, 8>;
def bf_CfgReserved0 : NPUReg40XX_RegFieldWrapper<"cfg_Reserved_0", 1, 9>;
def bf_CfgReserved1 : NPUReg40XX_RegFieldWrapper<"cfg_Reserved_1", 1, 10>;
def bf_CfgReserved2 : NPUReg40XX_RegFieldWrapper<"cfg_Reserved_2", 1, 11>;
def bf_RstCtxtNew : NPUReg40XX_RegFieldWrapper<"rst_ctxt_new", 1, 12>;
def bf_CfgReserved3 : NPUReg40XX_RegFieldWrapper<"cfg_Reserved_3", 1, 13>;
def bf_CfgReserved4 : NPUReg40XX_RegFieldWrapper<"cfg_Reserved_4", 1, 14>;
def bf_OduStatClr : NPUReg40XX_RegFieldWrapper<"odu_stat_clr", 1, 15>;
def bf_IduStatClr : NPUReg40XX_RegFieldWrapper<"idu_stat_clr", 1, 16>;
def bf_CfgReserved5 : NPUReg40XX_RegFieldWrapper<"cfg_Reserved_5", 1, 17>;
def bf_CfgReserved6 : NPUReg40XX_RegFieldWrapper<"cfg_Reserved_6", 14, 18>;
def reg_DpuCfg : NPUReg40XX_RegisterWrapper<"dpu_cfg", 4, 0xA0,
    ["workload_start_odu", "workload_start_idu", "workload_prm_sel", "workload_valid", "workload_shad_odu",
    "workload_shad_idu", "workload_idu_auto_upd_0", "workload_idu_auto_upd_1", "workload_odu_auto_upd",
    "cfg_Reserved_0", "cfg_Reserved_1", "cfg_Reserved_2", "rst_ctxt_new", "cfg_Reserved_3", "cfg_Reserved_4",
    "odu_stat_clr", "idu_stat_clr", "cfg_Reserved_5", "cfg_Reserved_6"], 0x00000004>;

// te_beg0
def bf_TeBegY : NPUReg40XX_RegFieldWrapper<"te_beg_y", 13, 0>;
def bf_TeBegZ : NPUReg40XX_RegFieldWrapper<"te_beg_z", 13, 13>;
def reg_TeBeg0 : NPUReg40XX_RegisterWrapper<"te_beg0", 4, 0xA4, ["te_beg_y", "te_beg_z"]>;

// te_beg1
def bf_TeBegX : NPUReg40XX_RegFieldWrapper<"te_beg_x", 13, 0>;
def reg_TeBeg1 : NPUReg40XX_RegisterWrapper<"te_beg1", 4, 0xA8, ["te_beg_x"]>;

// te_end0
def bf_TeEndY : NPUReg40XX_RegFieldWrapper<"te_end_y", 13, 0>;
def bf_TeEndZ : NPUReg40XX_RegFieldWrapper<"te_end_z", 13, 13>;
def reg_TeEnd0 : NPUReg40XX_RegisterWrapper<"te_end0", 4, 0xAC, ["te_end_y", "te_end_z"]>;

// te_end1
def bf_TeEndX : NPUReg40XX_RegFieldWrapper<"te_end_x", 13, 0>;
def reg_TeEnd1 : NPUReg40XX_RegisterWrapper<"te_end1", 4, 0xB0, ["te_end_x"]>;

// weight_size
def bf_WeightSize : NPUReg40XX_RegFieldWrapper<"weight_size", 32, 0>;
def reg_WeightSize : NPUReg40XX_RegisterWrapper<"weight_size", 4, 0xB4, ["weight_size"]>;

// weight_num
def bf_WeighNum : NPUReg40XX_RegFieldWrapper<"weight_num", 32, 0>;
def reg_WeightNum : NPUReg40XX_RegisterWrapper<"weight_num", 4, 0xB8, ["weight_num"]>;

// weight_start
def bf_WeightStart : NPUReg40XX_RegFieldWrapper<"weight_start", 32, 0>;
def reg_WeightStart : NPUReg40XX_RegisterWrapper<"weight_start", 4, 0xBC, ["weight_start"]>;

def bf_Invariant : NPUReg40XX_RegFieldWrapper<"invariant_", 64, 0>;
def reg_Invariant : NPUReg40XX_RegisterWrapper<"invariant_", 8, 0xC0, ["invariant_"]>;

def bf_InvariantIndex : NPUReg40XX_RegFieldWrapper<"invariant_index_", 32, 0>;
def reg_InvariantIndex : NPUReg40XX_RegisterWrapper<"invariant_index_", 4, 0xC8, ["invariant_index_"]>;

// uint8_t pad_[20]
def bf_Pad_7_1 : NPUReg40XX_RegFieldWrapper<"pad_7_1", 64, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_Pad_7_1 : NPUReg40XX_RegisterWrapper<"pad_7_1", 8, 0xCC, ["pad_7_1"]>;

def bf_Pad_2_2 : NPUReg40XX_RegFieldWrapper<"pad_7_2", 64, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_Pad_7_2 : NPUReg40XX_RegisterWrapper<"pad_7_2", 8, 0xD4, ["pad_7_2"]>;

def bf_Pad_2_3 : NPUReg40XX_RegFieldWrapper<"pad_7_3", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_Pad_7_3 : NPUReg40XX_RegisterWrapper<"pad_7_3", 4, 0xDC, ["pad_7_3"]>;

//
// RegMapped Specialization
//

class NPUReg40XX_RegMappedWrapper<string name = "", list<string> regsTypes = []> :
    VPURegMapped_RegMappedWrapper<NPUReg40XX_Dialect, "vpux::NPUReg40XX::", name, regsTypes> {

}

// DPUInvariantRegister
def reg_mapped_DpuInvariantRegister : NPUReg40XX_RegMappedWrapper<"DpuInvariantRegister",
    ["cmx_slice0_low_addr", "cmx_slice1_low_addr", "cmx_slice2_low_addr", "cmx_slice3_low_addr", "cmx_slice_size", "se_addr", "sparsity_addr", "se_size",
    "z_config", "kernel_pad_cfg", "tensor_size0", "tensor_size1", "tensor_start", "tensor_mode", "elops_sparsity_addr", "elops_se_addr",
    "elops_wload", "act_offset0", "act_offset1", "act_offset2", "act_offset3", "base_offset_a", "base_offset_b", "wt_offset",
    "odu_cfg", "odu_be_size", "odu_be_cnt", "odu_se_size",
    "te_dim0", "te_dim1", "pt_base", "sp_base",
    "mpe_cfg", "mpe_bus_data_sel", "elop_scale", "ppe_cfg", "ppe_bias", "ppe_scale", "ppe_scale_ctrl",
    "ppe_prelu", "ppe_scale_hclamp", "ppe_scale_lclamp", "ppe_misc", "ppe_fp_bias", "ppe_fp_scale", "ppe_fp_prelu",
    "ppe_fp_cfg", "odu_ac_base", "hwp_ctrl", "hwp_cmx_mem_addr", "odu_cast0", "odu_cast1", "odu_cast2", "nvar_tag",
    "pallet0", "pallet1", "pallet2", "pallet3", "pallet4", "pallet5", "pallet6", "pallet7",
    "se_addr1", "sparsity_addr1", "se_addr2", "sparsity_addr2", "se_addr3", "sparsity_addr3", "se_sp_size1", "se_sp_size2",
    "barriers_wait_mask_hi_", "barriers_wait_mask_lo_", "barriers_post_mask_hi_", "barriers_post_mask_lo_", "barriers_group_mask_",
    "barriers_sched_", "reserved_inv", "variant_count_", "cluster_invariant_", "pad_3"
    ]>;

// DPUVariantRegister
def reg_mapped_DpuVariantRegister : NPUReg40XX_RegMappedWrapper<"DpuVariantRegister",
    ["invar_ptr", "workload_size0", "workload_size1", "workload_start0", "workload_start1", "offset_addr", "hwp_wload_id",
    "var_cfg", "cbarrier_lo", "cbarrier_hi", "pbarrier_lo", "pbarrier_hi",
    "halo_region0A", "halo_region0B", "halo_region0C", "halo_region0D",
    "halo_region1A", "halo_region1B", "halo_region1C", "halo_region1D",
    "halo_region2A", "halo_region2B", "halo_region2C", "halo_region2D",
    "halo_region3A", "halo_region3B", "halo_region3C", "halo_region3D",
    "halo_region4A", "halo_region4B", "halo_region4C", "halo_region4D",
    "halo_region5A", "halo_region5B", "halo_region5C", "halo_region5D",
    "dpu_cfg", "te_beg0", "te_beg1", "te_end0", "te_end1",
    "weight_size", "weight_num", "weight_start",
    "invariant_", "invariant_index_", "pad_7_1", "pad_7_2", "pad_7_3"
    ]>;

// DMADescriptor

//
//  dma_64_bit_word_0
//
def bf_dma_watermark : NPUReg40XX_RegFieldWrapper<"dma_watermark", 1, 0>;                                  // Watermark to indicate that the job has completed
def reg_dma_watermark : NPUReg40XX_RegisterWrapper<"dma_watermark", 1, 0x0, ["dma_watermark"]>;

def bf_dma_link_address : NPUReg40XX_RegFieldWrapper<"dma_link_address", 48, 0>;                           // Pointer to the next element in linked list
def reg_dma_link_address : NPUReg40XX_RegisterWrapper<"dma_link_address", 8, 0x0, ["dma_link_address"]>;

def bf_dma_lra : NPUReg40XX_RegFieldWrapper<"dma_lra", 1, 63>;                                             // Link Relative Address. Base address fetched from LBA_ADDR
def reg_dma_lra : NPUReg40XX_RegisterWrapper<"dma_lra", 8, 0x0, ["dma_lra"]>;

//
//  dma_64_bit_word_1
//

def bf_dma_lba_addr : NPUReg40XX_RegFieldWrapper<"dma_lba_addr", 32, 0>;       // CMX address to the location of the Base Address for Relative Addressing options
def reg_dma_lba_addr : NPUReg40XX_RegisterWrapper<"dma_lba_addr", 4, 0x8,     // 8 dec
    ["dma_lba_addr"]>;

def bf_dma_src_aub : NPUReg40XX_RegFieldWrapper<"dma_src_aub", 8, 0>;          // Source AXI user bits
def reg_dma_src_aub : NPUReg40XX_RegisterWrapper<"dma_src_aub", 1, 0xC,        // 12 dec
    ["dma_src_aub"]>;

def bf_dma_dst_aub : NPUReg40XX_RegFieldWrapper<"dma_dst_aub", 8, 0>;          // Destination AXI user bits
def reg_dma_dst_aub : NPUReg40XX_RegisterWrapper<"dma_dst_aub", 1, 0xE,        // 14 dec
    ["dma_dst_aub"]>;

//
//  dma_64_bit_word_2
//

def bf_dma_cfg_fields_num_dim : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_num_dim", 3, 0>;
def bf_dma_cfg_fields_int_en : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_int_en", 1, 3>;
def bf_dma_cfg_fields_int_id : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_int_id", 8, 4>;
def bf_dma_cfg_fields_src_burst_length : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_src_burst_length", 4, 12>;
def bf_dma_cfg_fields_dst_burst_length : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_dst_burst_length", 4, 16>;
def bf_dma_cfg_fields_arb_qos : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_arb_qos", 8, 20>;
def bf_dma_cfg_fields_ord : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_ord", 1, 28>;
def bf_dma_cfg_fields_barrier_en : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_barrier_en", 1, 29>;
def bf_dma_cfg_fields_memset_en : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_memset_en", 1, 30>;
def bf_dma_cfg_fields_atp_en : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_atp_en", 1, 31>;
def bf_dma_cfg_fields_watermark_en : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_watermark_en", 1, 32>;
def bf_dma_cfg_fields_rwf_en : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_rwf_en", 1, 33>;
def bf_dma_cfg_fields_rws_en : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_rws_en", 1, 34>;
def bf_dma_cfg_fields_src_list_cfg : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_src_list_cfg", 2, 35>;
def bf_dma_cfg_fields_dst_list_cfg : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_dst_list_cfg", 2, 37>;
def bf_dma_cfg_fields_conversion_cfg : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_conversion_cfg", 3, 39>;
def bf_dma_cfg_fields_acceleration_cfg : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_acceleration_cfg", 2, 42>;
def bf_dma_cfg_fields_tile4_cfg : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_tile4_cfg", 2, 44>;
def bf_dma_cfg_fields_axi_user_bits_cfg : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_axi_user_bits_cfg", 2, 46>;
def bf_dma_cfg_fields_hwp_id_en : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_hwp_id_en", 1, 48>;
def bf_dma_cfg_fields_hwp_id : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_hwp_id", 12, 49>;
def bf_dma_cfg_fields_reserved : NPUReg40XX_RegFieldWrapper<"dma_cfg_fields_reserved", 3, 61>;

def reg_dma_cfg_fields : NPUReg40XX_RegisterWrapper<"dma_cfg_fields", 8, 0x10,  // 16 dec
    ["dma_cfg_fields_num_dim", "dma_cfg_fields_int_en", "dma_cfg_fields_int_id", "dma_cfg_fields_src_burst_length", "dma_cfg_fields_dst_burst_length",
    "dma_cfg_fields_arb_qos", "dma_cfg_fields_ord", "dma_cfg_fields_barrier_en", "dma_cfg_fields_memset_en", "dma_cfg_fields_atp_en",
    "dma_cfg_fields_watermark_en", "dma_cfg_fields_rwf_en", "dma_cfg_fields_rws_en", "dma_cfg_fields_src_list_cfg", "dma_cfg_fields_dst_list_cfg",
    "dma_cfg_fields_conversion_cfg", "dma_cfg_fields_acceleration_cfg", "dma_cfg_fields_tile4_cfg", "dma_cfg_fields_axi_user_bits_cfg",
    "dma_cfg_fields_hwp_id_en", "dma_cfg_fields_hwp_id", "dma_cfg_fields_reserved"]>;

//
//  dma_64_bit_word_3
//

def bf_dma_remote_width_fetch : NPUReg40XX_RegFieldWrapper<"dma_remote_width_fetch", 32, 0>;
def reg_dma_remote_width_fetch : NPUReg40XX_RegisterWrapper<"dma_remote_width_fetch", 4, 0x18, // 24 dec
    ["dma_remote_width_fetch"]>;

def bf_dma_width_src : NPUReg40XX_RegFieldWrapper<"dma_width_src", 32, 0>;
def bf_dma_width_dst : NPUReg40XX_RegFieldWrapper<"dma_width_dst", 32, 32>;

// overlap because fields dma_src_width and dma_rwf_addr are placed at the same memory address
def reg_dma_width : NPUReg40XX_RegisterWrapper<"dma_width", 8, 0x18, // 24 dec
    ["dma_width_src", "dma_width_dst"]>;

//
//  dma_64_bit_word_4: compress, decompress and WPREP regs are placed at the same memory address
//

//
//  dma_64_bit_word_4_compress
//

def bf_dma_acc_info_compress_dtype : NPUReg40XX_RegFieldWrapper<"dma_acc_info_compress_dtype", 2, 0>;
def bf_dma_acc_info_compress_reserved1 : NPUReg40XX_RegFieldWrapper<"dma_acc_info_compress_reserved1", 1, 2>;
def bf_dma_acc_info_compress_sparse : NPUReg40XX_RegFieldWrapper<"dma_acc_info_compress_sparse", 1, 3>;
def bf_dma_acc_info_compress_bitc_en : NPUReg40XX_RegFieldWrapper<"dma_acc_info_compress_bitc_en", 1, 4>;
def bf_dma_acc_info_compress_z : NPUReg40XX_RegFieldWrapper<"dma_acc_info_compress_z", 10, 5>;
def bf_dma_acc_info_compress_bitmap_buf_sz : NPUReg40XX_RegFieldWrapper<"dma_acc_info_compress_bitmap_buf_sz", 19, 15>;
def bf_dma_acc_info_compress_reserved2 : NPUReg40XX_RegFieldWrapper<"dma_acc_info_compress_reserved2", 3, 34>;
def bf_dma_acc_info_compress_bitmap_base_addr : NPUReg40XX_RegFieldWrapper<"dma_acc_info_compress_bitmap_base_addr", 27, 37>;

def reg_dma_acc_info_compress : NPUReg40XX_RegisterWrapper<"dma_acc_info_compress", 8, 0x20, // 32 dec
    ["dma_acc_info_compress_dtype", "dma_acc_info_compress_reserved1", "dma_acc_info_compress_sparse",
    "dma_acc_info_compress_bitc_en", "dma_acc_info_compress_z", "dma_acc_info_compress_bitmap_buf_sz",
    "dma_acc_info_compress_reserved2", "dma_acc_info_compress_bitmap_base_addr"]>;

//
//  dma_64_bit_word_4_decompress
//

def bf_dma_acc_info_decompress_dtype : NPUReg40XX_RegFieldWrapper<"dma_acc_info_decompress_dtype", 2, 0>;
def bf_dma_acc_info_decompress_reserved1 : NPUReg40XX_RegFieldWrapper<"dma_acc_info_decompress_reserved1", 1, 2>;
def bf_dma_acc_info_decompress_sparse : NPUReg40XX_RegFieldWrapper<"dma_acc_info_decompress_sparse", 1, 3>;
def bf_dma_acc_info_decompress_bitc_en : NPUReg40XX_RegFieldWrapper<"dma_acc_info_decompress_bitc_en", 1, 4>;
def bf_dma_acc_info_decompress_z : NPUReg40XX_RegFieldWrapper<"dma_acc_info_decompress_z", 10, 5>;
def bf_dma_acc_info_decompress_reserved2 : NPUReg40XX_RegFieldWrapper<"dma_acc_info_decompress_reserved2", 22, 15>;
def bf_dma_acc_info_decompress_bitmap_base_addr : NPUReg40XX_RegFieldWrapper<"dma_acc_info_decompress_bitmap_base_addr", 27, 37>;

def reg_dma_acc_info_decompress_reg : NPUReg40XX_RegisterWrapper<"dma_acc_info_decompress", 8, 0x20,  // 32 dec
    ["dma_acc_info_decompress_dtype", "dma_acc_info_decompress_reserved1",
    "dma_acc_info_decompress_sparse", "dma_acc_info_decompress_bitc_en", "dma_acc_info_decompress_z",
    "dma_acc_info_decompress_reserved2", "dma_acc_info_decompress_bitmap_base_addr"]>;

//
//  dma_64_bit_word_4_w_prep
//

def bf_dma_acc_info_w_prep_dtype : NPUReg40XX_RegFieldWrapper<"dma_acc_info_w_prep_dtype", 2, 0>;
def bf_dma_acc_info_w_prep_reserved1 : NPUReg40XX_RegFieldWrapper<"dma_acc_info_w_prep_reserved1", 1, 2>;
def bf_dma_acc_info_w_prep_sparse : NPUReg40XX_RegFieldWrapper<"dma_acc_info_w_prep_sparse", 2, 3>;
def bf_dma_acc_info_w_prep_zeropoint : NPUReg40XX_RegFieldWrapper<"dma_acc_info_w_prep_zeropoint", 8, 5>;
def bf_dma_acc_info_w_prep_ic : NPUReg40XX_RegFieldWrapper<"dma_acc_info_w_prep_ic", 14, 13>;
def bf_dma_acc_info_w_prep_filtersize : NPUReg40XX_RegFieldWrapper<"dma_acc_info_w_prep_filtersize", 7, 27>;
def bf_dma_acc_info_w_prep_reserved2 : NPUReg40XX_RegFieldWrapper<"dma_acc_info_w_prep_reserved2", 3, 34>;
def bf_dma_acc_info_w_prep_bitmap_base_addr : NPUReg40XX_RegFieldWrapper<"dma_acc_info_w_prep_bitmap_base_addr", 27, 37>;

def reg_dma_acc_info_w_prep_reg : NPUReg40XX_RegisterWrapper<"dma_acc_info_w_prep", 8, 0x20,  // 32 dec
    ["dma_acc_info_w_prep_dtype", "dma_acc_info_w_prep_reserved1", "dma_acc_info_w_prep_sparse",
    "dma_acc_info_w_prep_zeropoint", "dma_acc_info_w_prep_ic", "dma_acc_info_w_prep_filtersize",
    "dma_acc_info_w_prep_reserved2", "dma_acc_info_w_prep_bitmap_base_addr"]>;

//
//  dma_64_bit_word_4_acc_info_mem_set
//

def bf_dma_mset_data : NPUReg40XX_RegFieldWrapper<"dma_mset_data", 32, 0>;

def reg_dma_mset_data : NPUReg40XX_RegisterWrapper<"dma_mset_data", 4, 0x20,  // 32 dec
    ["dma_mset_data"]>;

//
//  dma_64_bit_word_5
//

def bf_dma_src : NPUReg40XX_RegFieldWrapper<"dma_src", 48, 0>;
def bf_dma_sra : NPUReg40XX_RegFieldWrapper<"dma_sra", 1, 63>;

def reg_dma_src_addr : NPUReg40XX_RegisterWrapper<"dma_src_addr", 8, 0x28,  // 40 dec
    ["dma_src", "dma_sra"]>;

//
//  dma_64_bit_word_6
//

def bf_dma_dst : NPUReg40XX_RegFieldWrapper<"dma_dst", 48, 0>;
def bf_dma_dra : NPUReg40XX_RegFieldWrapper<"dma_dra", 1, 63>;

def reg_dma_dst_addr : NPUReg40XX_RegisterWrapper<"dma_dst_addr", 8, 0x30,  // 48 dec
    ["dma_dst", "dma_dra"]>;

//
//  dma_64_bit_word_7
//

def bf_dma_sba_addr : NPUReg40XX_RegFieldWrapper<"dma_sba_addr", 32, 0>;
def reg_dma_sba_addr : NPUReg40XX_RegisterWrapper<"dma_sba_addr", 4, 0x38,  // 56 dec
    ["dma_sba_addr"]>;

def bf_dma_dba_addr : NPUReg40XX_RegFieldWrapper<"dma_dba_addr", 32, 0>;
def reg_dma_dba_addr : NPUReg40XX_RegisterWrapper<"dma_dba_addr", 4, 0x3C,  // 56 dec
    ["dma_dba_addr"]>;

//
//  dma_64_bit_word_8
//

def bf_dma_barrier_prod_mask_lower : NPUReg40XX_RegFieldWrapper<"dma_barrier_prod_mask_lower", 64, 0>;
def reg_dma_barrier_prod_mask_lower : NPUReg40XX_RegisterWrapper<"dma_barrier_prod_mask_lower", 8, 0x40,  // 64 dec
    ["dma_barrier_prod_mask_lower"]>;

//
//  dma_64_bit_word_9
//

def bf_dma_barrier_cons_mask_lower : NPUReg40XX_RegFieldWrapper<"dma_barrier_cons_mask_lower", 64, 0>;
def reg_dma_barrier_cons_mask_lower : NPUReg40XX_RegisterWrapper<"dma_barrier_cons_mask_lower", 8, 0x48,  // 72 dec
    ["dma_barrier_cons_mask_lower"]>;

//
//  dma_64_bit_word_10
//

def bf_dma_barrier_prod_mask_upper : NPUReg40XX_RegFieldWrapper<"dma_barrier_prod_mask_upper", 32, 0>;
def reg_dma_barrier_prod_mask_upper : NPUReg40XX_RegisterWrapper<"dma_barrier_prod_mask_upper", 8, 0x50,  // 80 dec
    ["dma_barrier_prod_mask_upper"]>;

//
//  dma_64_bit_word_11
//

def bf_dma_barrier_cons_mask_upper : NPUReg40XX_RegFieldWrapper<"dma_barrier_cons_mask_upper", 32, 0>;
def reg_dma_barrier_cons_mask_upper : NPUReg40XX_RegisterWrapper<"dma_barrier_cons_mask_upper", 8, 0x58,  // 88 dec
    ["dma_barrier_cons_mask_upper"]>;

//
//  dma_64_bit_word_12
//

def bf_dma_list_size_src : NPUReg40XX_RegFieldWrapper<"dma_list_size_src", 32, 0>;
def bf_dma_list_size_dst : NPUReg40XX_RegFieldWrapper<"dma_list_size_dst", 32, 32>;

def reg_dma_list_size : NPUReg40XX_RegisterWrapper<"dma_list_size", 8, 0x60,  // 96 dec
    ["dma_list_size_src", "dma_list_size_dst"]>;

def bf_dma_dim_size_src_1 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_src_1", 32, 0>;
def bf_dma_dim_size_dst_1 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_dst_1", 32, 32>;

def reg_dma_dim_size : NPUReg40XX_RegisterWrapper<"dma_dim_size", 8, 0x60,  // 96 dec
    ["dma_dim_size_src_1", "dma_dim_size_dst_1"]>;

//
//  dma_64_bit_word_13
//

def bf_dma_stride_src_1 : NPUReg40XX_RegFieldWrapper<"dma_stride_src_1", 32, 0>;
def reg_dma_stride_src_1 : NPUReg40XX_RegisterWrapper<"dma_stride_src_1", 4, 0x68,  // 104 dec
    ["dma_stride_src_1"]>;

def bf_dma_stride_dst_1 : NPUReg40XX_RegFieldWrapper<"dma_stride_dst_1", 32, 0>;
def reg_dma_stride_dst_1 : NPUReg40XX_RegisterWrapper<"dma_stride_dst_1", 4, 0x6C,  // 108 dec
    ["dma_stride_dst_1"]>;

//
//  dma_64_bit_word_14
//

def bf_dma_dim_size_src_2 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_src_2", 32, 0>;
def bf_dma_dim_size_dst_2 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_dst_2", 32, 32>;

def reg_dma_dim_size_2 : NPUReg40XX_RegisterWrapper<"dma_dim_size_2", 8, 0x70,  // 112 dec
    ["dma_dim_size_src_2", "dma_dim_size_dst_2"]>;

def bf_dma_list_addr_src : NPUReg40XX_RegFieldWrapper<"dma_list_addr_src", 32, 0>;
def bf_dma_list_addr_dst : NPUReg40XX_RegFieldWrapper<"dma_list_addr_dst", 32, 32>;

def reg_dma_list_addr : NPUReg40XX_RegisterWrapper<"dma_list_addr", 8, 0x70,  // 112 dec
    ["dma_list_addr_src", "dma_list_addr_dst"]>;

//
//  dma_64_bit_word_15
//

def bf_dma_stride_src_2 : NPUReg40XX_RegFieldWrapper<"dma_stride_src_2", 32, 0>;
def reg_dma_stride_src_2 : NPUReg40XX_RegisterWrapper<"dma_stride_src_2", 4, 0x78,  // 120 dec
    ["dma_stride_src_2"]>;

def bf_dma_stride_dst_2 : NPUReg40XX_RegFieldWrapper<"dma_stride_dst_2", 32, 0>;
def reg_dma_stride_dst_2 : NPUReg40XX_RegisterWrapper<"dma_stride_dst_2", 4, 0x7C,  // 124 dec
    ["dma_stride_dst_2"]>;

def bf_dma_remote_width_store : NPUReg40XX_RegFieldWrapper<"dma_remote_width_store", 32, 0>;
def reg_dma_remote_width_store : NPUReg40XX_RegisterWrapper<"dma_remote_width_store", 4, 0x7C,  // 124 dec
    ["dma_remote_width_store"]>;

//
//  dma_64_bit_word_16
//

def bf_dma_dim_size_src_3 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_src_3", 16, 0>;
def reg_dma_dim_size_src_3 : NPUReg40XX_RegisterWrapper<"dma_dim_size_src_3", 2, 0x80,  // 128 dec
    ["dma_dim_size_src_3"]>;

def bf_dma_dim_size_src_4 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_src_4", 16, 0>;
def reg_dma_dim_size_src_4 : NPUReg40XX_RegisterWrapper<"dma_dim_size_src_4", 2, 0x82,  // 130 dec
    ["dma_dim_size_src_4"]>;

def bf_dma_dim_size_dst_3 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_dst_3", 16, 0>;
def reg_dma_dim_size_dst_3 : NPUReg40XX_RegisterWrapper<"dma_dim_size_dst_3", 2, 0x84,  // 132 dec
    ["dma_dim_size_dst_3"]>;

def bf_dma_dim_size_dst_4 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_dst_4", 16, 0>;
def reg_dma_dim_size_dst_4 : NPUReg40XX_RegisterWrapper<"dma_dim_size_dst_4", 2, 0x86,  // 134 dec
    ["dma_dim_size_dst_4"]>;

//
//  dma_64_bit_word_17
//

def bf_dma_dim_size_src_5 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_src_5", 16, 0>;
def reg_dma_dim_size_src_5 : NPUReg40XX_RegisterWrapper<"dma_dim_size_src_5", 2, 0x88,  // 136 dec
    ["dma_dim_size_src_5"]>;

def bf_dma_dim_size_dst_5 : NPUReg40XX_RegFieldWrapper<"dma_dim_size_dst_5", 16, 0>;
def reg_dma_dst_dim_size_5 : NPUReg40XX_RegisterWrapper<"dma_dim_size_dst_5", 2, 0x8C,  // 140 dec
    ["dma_dim_size_dst_5"]>;

//
//  dma_64_bit_word_18
//

def bf_dma_stride_src_3 : NPUReg40XX_RegFieldWrapper<"dma_stride_src_3", 32, 0>;
def reg_dma_stride_src_3 : NPUReg40XX_RegisterWrapper<"dma_stride_src_3", 4, 0x90,  // 144 dec
    ["dma_stride_src_3"]>;

def bf_dma_stride_dst_3 : NPUReg40XX_RegFieldWrapper<"dma_stride_dst_3", 32, 0>;
def reg_dma_stride_dst_3 : NPUReg40XX_RegisterWrapper<"dma_stride_dst_3", 4, 0x94,  // 148 dec
    ["dma_stride_dst_3"]>;

//
//  dma_64_bit_word_19
//

def bf_dma_stride_src_4 : NPUReg40XX_RegFieldWrapper<"dma_stride_src_4", 32, 0>;
def reg_dma_stride_src_4 : NPUReg40XX_RegisterWrapper<"dma_stride_src_4", 4, 0x98,  // 152 dec
    ["dma_stride_src_4"]>;

def bf_dma_stride_dst_4 : NPUReg40XX_RegFieldWrapper<"dma_stride_dst_4", 32, 0>;
def reg_dma_stride_dst_4 : NPUReg40XX_RegisterWrapper<"dma_stride_dst_4", 4, 0x9C,  // 156 dec
    ["dma_stride_dst_4"]>;

//
//  dma_64_bit_word_20
//

def bf_dma_stride_src_5 : NPUReg40XX_RegFieldWrapper<"dma_stride_src_5", 32, 0>;
def reg_dma_stride_src_5 : NPUReg40XX_RegisterWrapper<"dma_stride_src_5", 4, 0xA0,  // 160 dec
    ["dma_stride_src_5"]>;

def bf_dma_stride_dst_5 : NPUReg40XX_RegFieldWrapper<"dma_stride_dst_5", 32, 0>;
def reg_dma_stride_dst_5 : NPUReg40XX_RegisterWrapper<"dma_stride_dst_5", 4, 0xA4,  // 164 dec
    ["dma_stride_dst_5"]>;

//
//  dma_64_bit_word_21
//

def bf_dma_word_21_reserved : NPUReg40XX_RegFieldWrapper<"dma_word_21_reserved", 64, 0>;
def reg_dma_word_21_reserved : NPUReg40XX_RegisterWrapper<"dma_word_21_reserved", 8, 0xA8,  // 168 dec
    ["dma_word_21_reserved"]>;

//
//  dma_64_bit_word_22
//

def bf_dma_word_22_reserved : NPUReg40XX_RegFieldWrapper<"dma_word_22_reserved", 64, 0>;
def reg_dma_word_22_reserved : NPUReg40XX_RegisterWrapper<"dma_word_22_reserved", 8, 0xB0,  // 176 dec
    ["dma_word_22_reserved"]>;

//
//  dma_64_bit_word_21
//

def bf_dma_word_23_reserved : NPUReg40XX_RegFieldWrapper<"dma_word_23_reserved", 64, 0>;
def reg_dma_word_23_reserved : NPUReg40XX_RegisterWrapper<"dma_word_23_reserved", 8, 0xB8,  // 184 dec
    ["dma_word_23_reserved"]>;

//
//  dma_barriers_sched
//

def reg_VpuTaskSchedulingBarrierConfig : NPUReg40XX_RegisterWrapper<"dma_barriers_sched", 8, 0xC0,
    ["start_after_", "clean_after_"]>;

def bf_dma_pad_24_0 : NPUReg40XX_RegFieldWrapper<"dma_pad_24_0", 64, 0>;
def reg_dma_pad_24_0 : NPUReg40XX_RegisterWrapper<"dma_pad_24_0", 8, 0xC8, ["dma_pad_24_0"]>;

def bf_dma_pad_24_1 : NPUReg40XX_RegFieldWrapper<"dma_pad_24_1", 64, 0>;
def reg_dma_pad_24_1 : NPUReg40XX_RegisterWrapper<"dma_pad_24_1", 8, 0xD0, ["dma_pad_24_1"]>;

def bf_dma_pad_24_2 : NPUReg40XX_RegFieldWrapper<"dma_pad_24_2", 64, 0>;
def reg_dma_pad_24_2 : NPUReg40XX_RegisterWrapper<"dma_pad_24_2", 8, 0xD8, ["dma_pad_24_2"]>;

// DMARegister descriptor
def reg_mapped_DMARegister : NPUReg40XX_RegMappedWrapper<"DMARegister",
    ["dma_watermark", "dma_link_address", "dma_lra",
    "dma_lba_addr", "dma_src_aub", "dma_dst_aub",
    "dma_cfg_fields",
    "dma_remote_width_fetch", "dma_width",
    "dma_acc_info_compress", "dma_acc_info_decompress", "dma_acc_info_w_prep", "dma_mset_data",
    "dma_src_addr",
    "dma_dst_addr",
    "dma_sba_addr", "dma_dba_addr",
    "dma_barrier_prod_mask_lower",
    "dma_barrier_cons_mask_lower",
    "dma_barrier_prod_mask_upper",
    "dma_barrier_cons_mask_upper",
    "dma_list_size", "dma_dim_size",
    "dma_stride_src_1", "dma_stride_dst_1",
    "dma_dim_size_2", "dma_list_addr",
    "dma_stride_src_2", "dma_stride_dst_2", "dma_remote_width_store",
    "dma_dim_size_src_3", "dma_dim_size_src_4", "dma_dim_size_dst_3", "dma_dim_size_dst_4",
    "dma_dim_size_src_5", "dma_dim_size_dst_5",
    "dma_stride_src_3", "dma_stride_dst_3",
    "dma_stride_src_4", "dma_stride_dst_4",
    "dma_stride_src_5", "dma_stride_dst_5",
    "dma_word_21_reserved",
    "dma_word_22_reserved",
    "dma_word_23_reserved",
    "dma_barriers_sched",
    "dma_pad_24_0", "dma_pad_24_1", "dma_pad_24_2"
    ]>;

//
//  BarrierCfg registers
//

def bf_NextSameId : NPUReg40XX_RegFieldWrapper<"next_same_id_", 32, 0, "UINT">;
def reg_NextSameId : NPUReg40XX_RegisterWrapper<"next_same_id_", 4, 0x0, ["next_same_id_"]>;

def bf_ProducerCount: NPUReg40XX_RegFieldWrapper<"producer_count_", 16, 0>;
def reg_ProducerCount : NPUReg40XX_RegisterWrapper<"producer_count_", 2, 0x4, ["producer_count_"]>;

def bf_ConsumerCount : NPUReg40XX_RegFieldWrapper<"consumer_count_", 16, 0>;
def reg_ConsumerCount : NPUReg40XX_RegisterWrapper<"consumer_count_", 2, 0x6, ["consumer_count_"]>;

def bf_BarrierId : NPUReg40XX_RegFieldWrapper<"real_id_", 8, 0>;
def reg_BarrierId : NPUReg40XX_RegisterWrapper<"real_id_", 1, 0x8, ["real_id_"]>;

def bf_BarCfg_Pad3 : NPUReg40XX_RegFieldWrapper<"barcfg_pad_3_", 24, 0>;
def reg_BarCfg_Pad3 : NPUReg40XX_RegisterWrapper<"barcfg_pad_3_", 3, 0x9, ["barcfg_pad_3_"]>;

// BarrierCfg descriptor
def reg_mapped_BarrierDescriptor : NPUReg40XX_RegMappedWrapper<"VpuBarrierCountConfig",
    ["next_same_id_", "producer_count_", "consumer_count_", "real_id_", "barcfg_pad_3_"]>;

//
// ActKernelRange registers
//

def bf_ActWLType : NPUReg40XX_RegFieldWrapper<"type", 8, 0>;
def reg_ActWLType : NPUReg40XX_RegisterWrapper<"type", 1, 0x0, ["type"]>; // WL_KERNEL = 0x00, WL_DEBUG = 0x04, WL_UNKNOWN

def bf_KernelEntry : NPUReg40XX_RegFieldWrapper<"kernel_entry", 64, 0>;
def reg_KernelEntry : NPUReg40XX_RegisterWrapper<"kernel_entry", 8, 0x8, ["kernel_entry"]>;

def bf_TextWindowBase: NPUReg40XX_RegFieldWrapper<"text_window_base", 64, 0>;
def reg_TextWindowBase : NPUReg40XX_RegisterWrapper<"text_window_base", 8, 0x10, ["text_window_base"]>;

def bf_CodeSize : NPUReg40XX_RegFieldWrapper<"code_size", 32, 0>;
def reg_CodeSize : NPUReg40XX_RegisterWrapper<"code_size", 4, 0x18, ["code_size"]>;

def bf_ReservedAkr : NPUReg40XX_RegFieldWrapper<"reserved_akr", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_ReservedAkr : NPUReg40XX_RegisterWrapper<"reserved_akr", 4, 0x1C, ["reserved_akr"]>;

def bf_KernelInvoCount : NPUReg40XX_RegFieldWrapper<"kernel_invo_count", 32, 0>;
def reg_KernelInvoCount : NPUReg40XX_RegisterWrapper<"kernel_invo_count", 4, 0x20, ["kernel_invo_count"]>;

def bf_Pad1_4 : NPUReg40XX_RegFieldWrapper<"pad1_4", 32, 0>;
def reg_Pad1_4 : NPUReg40XX_RegisterWrapper<"pad1_4", 4, 0x24, ["pad1_4"]>;

// ActKernelRange descriptor
def reg_mapped_ActKernelRangeDescriptor : NPUReg40XX_RegMappedWrapper<"VpuActKernelRange",
    ["type", "kernel_entry", "text_window_base", "code_size", "reserved_akr", "kernel_invo_count", "pad1_4"]>;

//
// ActKernelInvocation registers
//

def bf_Range : NPUReg40XX_RegFieldWrapper<"range", 64, 0>;
def reg_Range : NPUReg40XX_RegisterWrapper<"range", 8, 0x0, ["range"]>;

def bf_KernelArgs : NPUReg40XX_RegFieldWrapper<"kernel_args", 64, 0>;
def reg_KernelArgs : NPUReg40XX_RegisterWrapper<"kernel_args", 8, 0x8, ["kernel_args"]>;

def bf_DataWindowBase: NPUReg40XX_RegFieldWrapper<"data_window_base", 64, 0>;
def reg_DataWindowBase : NPUReg40XX_RegisterWrapper<"data_window_base", 8, 0x10, ["data_window_base"]>;

def bf_PerfPacketOut : NPUReg40XX_RegFieldWrapper<"perf_packet_out", 64, 0>;
def reg_PerfPacketOut : NPUReg40XX_RegisterWrapper<"perf_packet_out", 8, 0x18, ["perf_packet_out"]>;

def bf_BarrierWaitMaskHiAct : NPUReg40XX_RegFieldWrapper<"barriers_wait_mask_hi_act", 32, 0>;
def reg_BarrierWaitMaskHiAct : NPUReg40XX_RegisterWrapper<"barriers_wait_mask_hi_act", 8, 0x20, ["barriers_wait_mask_hi_act"]>;

def bf_BarrierWaitMaskLoAct : NPUReg40XX_RegFieldWrapper<"barriers_wait_mask_lo_act", 64, 0>;
def reg_BarrierWaitMaskLoAct : NPUReg40XX_RegisterWrapper<"barriers_wait_mask_lo_act", 8, 0x28, ["barriers_wait_mask_lo_act"]>;

def bf_BarrierWaitPostHiAct : NPUReg40XX_RegFieldWrapper<"barriers_post_mask_hi_act", 32, 0>;
def reg_BarrierWaitPostHiAct : NPUReg40XX_RegisterWrapper<"barriers_post_mask_hi_act", 8, 0x30, ["barriers_post_mask_hi_act"]>;

def bf_BarrierWaitPostLoAct : NPUReg40XX_RegFieldWrapper<"barriers_post_mask_lo_act", 64, 0>;
def reg_BarrierWaitPostLoAct : NPUReg40XX_RegisterWrapper<"barriers_post_mask_lo_act", 8, 0x38, ["barriers_post_mask_lo_act"]>;

def bf_BarrierGroupAct : NPUReg40XX_RegFieldWrapper<"group_act", 8, 0>;
def bf_BarrierMaskAct : NPUReg40XX_RegFieldWrapper<"mask_act", 8, 8>;
def reg_BarrierGroupMaskAct : NPUReg40XX_RegisterWrapper<"barriers_group_mask_act", 8, 0x40, ["group_act", "mask_act"]>;

def reg_SchedulingBarrierConfig : NPUReg40XX_RegisterWrapper<"act_invo_barriers_sched", 8, 0x48, ["start_after_", "clean_after_"]>;

def bf_ReservedAki : NPUReg40XX_RegFieldWrapper<"reserved_aki", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_ReservedAki : NPUReg40XX_RegisterWrapper<"reserved_aki", 4, 0x50, ["reserved_aki"]>;

def bf_InvoTile : NPUReg40XX_RegFieldWrapper<"invo_tile", 32, 0>;
def reg_InvoTile : NPUReg40XX_RegisterWrapper<"invo_tile", 4, 0x54, ["invo_tile"]>;

def bf_KernelRangeIndex : NPUReg40XX_RegFieldWrapper<"kernel_range_index", 32, 0>;
def reg_KernelRangeIndex : NPUReg40XX_RegisterWrapper<"kernel_range_index", 4, 0x58, ["kernel_range_index"]>;

def bf_NexrAkiWlAddr : NPUReg40XX_RegFieldWrapper<"next_aki_wl_addr", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_NexrAkiWlAddr : NPUReg40XX_RegisterWrapper<"next_aki_wl_addr", 4, 0x5C, ["next_aki_wl_addr"]>;

// ActKernelInvocation descriptor
def reg_mapped_ActKernelInvocationDescriptor : NPUReg40XX_RegMappedWrapper<"VpuActKernelInvocation",
    ["range", "kernel_args", "data_window_base", "perf_packet_out",
    "barriers_wait_mask_hi_act", "barriers_wait_mask_lo_act", "barriers_post_mask_hi_act", "barriers_post_mask_lo_act", "barriers_group_mask_act",
    "act_invo_barriers_sched", "reserved_aki", "invo_tile", "kernel_range_index", "next_aki_wl_addr"]>;

//
// M2I registers
//

// VpuMediaBuffDescriptor

// IN_ADDR common
def bf_InAddr : NPUReg40XX_RegFieldWrapper<"inAddr", 64, 0>;

// IN_ADDR_0
def reg_InAddr0 : NPUReg40XX_RegisterWrapper<"inAddr0", 8, 0x0, ["inAddr"]>;

// common part for InSize0,1,2
def bf_Width : NPUReg40XX_RegFieldWrapper<"width", 16, 0>;
def bf_Height : NPUReg40XX_RegFieldWrapper<"height", 16, 16>;
def bf_Ls : NPUReg40XX_RegFieldWrapper<"ls", 24, 32>;

// InSize0
def bf_PID : NPUReg40XX_RegFieldWrapper<"pid", 8, 56>;
def reg_InSize0 : NPUReg40XX_RegisterWrapper<"inSize0", 8, 0x8, ["width", "height", "ls", "pid"]>;

// IN_ADDR_1
def reg_InAddr1 : NPUReg40XX_RegisterWrapper<"inAddr1", 8, 0x10, ["inAddr"]>;

// InSize1
def bf_HWPEN : NPUReg40XX_RegFieldWrapper<"HWPEN", 1, 56>;
def bf_ExtHDR : NPUReg40XX_RegFieldWrapper<"ExtHDR", 1, 57>;
def bf_Reserved_InSize1 : NPUReg40XX_RegFieldWrapper<"Reserved_InSize1", 6, 58>;
def reg_InSize1 : NPUReg40XX_RegisterWrapper<"inSize1", 8, 0x18, ["width", "height", "ls", "HWPEN", "ExtHDR", "Reserved_InSize1"]>;

// IN_ADDR_2
def reg_InAddr2 : NPUReg40XX_RegisterWrapper<"inAddr2", 8, 0x20, ["inAddr"]>;

// InSize2
def bf_Reserved_InSize2 : NPUReg40XX_RegFieldWrapper<"Reserved_InSize2", 8, 56>;
def reg_InSize2 : NPUReg40XX_RegisterWrapper<"inSize2", 8, 0x28, ["width", "height", "ls", "Reserved_InSize2"]>;

// IOCfg
def bf_InFormat : NPUReg40XX_RegFieldWrapper<"inFormat", 8, 0>;
def bf_OutFormat : NPUReg40XX_RegFieldWrapper<"outFormat", 8, 8>;
def bf_NumRois : NPUReg40XX_RegFieldWrapper<"numRois", 16, 16>;
def bf_SampleType : NPUReg40XX_RegFieldWrapper<"sampleType", 4, 32>;
def bf_Operations : NPUReg40XX_RegFieldWrapper<"operations", 4, 36>;
def bf_IFC : NPUReg40XX_RegFieldWrapper<"IFC", 8, 40>;
def bf_IRQMask : NPUReg40XX_RegFieldWrapper<"IRQMask", 16, 48>;

def reg_IOCfg : NPUReg40XX_RegisterWrapper<"IOCfg", 8, 0x30, ["inFormat", "outFormat", "numRois", "sampleType", "operations", "IFC", "IRQMask"]>;

// NormFactor
def bf_NormFact0 : NPUReg40XX_RegFieldWrapper<"NormFact0", 16, 0>;
def bf_NormFact1 : NPUReg40XX_RegFieldWrapper<"NormFact1", 16, 16>;
def bf_NormFact2 : NPUReg40XX_RegFieldWrapper<"NormFact2", 16, 32>;
def bf_NormFact3 : NPUReg40XX_RegFieldWrapper<"NormFact3", 16, 48>;

def reg_NormFactor_0 : NPUReg40XX_RegisterWrapper<"NormFactor_0", 8, 0x38, ["NormFact0", "NormFact1", "NormFact2", "NormFact3"]>;
def reg_NormFactor_1 : NPUReg40XX_RegisterWrapper<"NormFactor_1", 8, 0x40, ["NormFact0", "NormFact1", "NormFact2", "NormFact3"]>;
def reg_NormFactor_2 : NPUReg40XX_RegisterWrapper<"NormFactor_2", 8, 0x48, ["NormFact0", "NormFact1", "NormFact2", "NormFact3"]>;

// PSOB
def bf_InPS : NPUReg40XX_RegFieldWrapper<"inPS", 32, 0>;
def bf_OutBase : NPUReg40XX_RegFieldWrapper<"outBase", 16, 32>;
def bf_HWPAddrLO : NPUReg40XX_RegFieldWrapper<"HWPAddrLO", 16, 48>;

def reg_PSOB : NPUReg40XX_RegisterWrapper<"PSOB", 8, 0x50, ["inPS", "outBase", "HWPAddrLO"]>;

// nextDesc
def bf_NextDesc : NPUReg40XX_RegFieldWrapper<"nextDesc", 48, 0>;
def reg_NextDesc : NPUReg40XX_RegisterWrapper<"nextDesc", 6, 0x58, ["nextDesc"]>;

// HWPAddrHI
def bf_HWPAddrHI : NPUReg40XX_RegFieldWrapper<"HWPAddrHI", 16, 0>;
def reg_HWPAddrHI : NPUReg40XX_RegisterWrapper<"HWPAddrHI", 2, 0x5E, ["HWPAddrHI"]>;

//VpuROIDescriptor
// RoiDef
def bf_RoiBase : NPUReg40XX_RegFieldWrapper<"roiBase", 32, 0>;
def bf_OutFormatLocal : NPUReg40XX_RegFieldWrapper<"outFormatLocal", 8, 32>;
def bf_SamlingTypeLocal : NPUReg40XX_RegFieldWrapper<"samlingTypeLocal", 4, 40>;
def bf_OFC : NPUReg40XX_RegFieldWrapper<"OFC", 8, 44>;
def bf_IRQLocal : NPUReg40XX_RegFieldWrapper<"IRQLocal", 8, 52>;
def bf_HWProfEN : NPUReg40XX_RegFieldWrapper<"HWProfEN", 1, 60>;
def bf_RoiDef_RESERVED : NPUReg40XX_RegFieldWrapper<"RoiDef_RESERVED", 3, 61>;
def reg_RoiDef : NPUReg40XX_RegisterWrapper<"RoiDef", 8, 0x60,
    ["roiBase", "outFormatLocal", "samlingTypeLocal", "OFC", "IRQLocal", "HWProfEN", "RoiDef_RESERVED"]>;

// RoiCfg
def bf_X_coord : NPUReg40XX_RegFieldWrapper<"X_coord", 16, 0>;
def bf_Y_coord : NPUReg40XX_RegFieldWrapper<"Y_coord", 16, 16>;
def bf_RoiWidth : NPUReg40XX_RegFieldWrapper<"roiWidth", 16, 32>;
def bf_RoiHeight : NPUReg40XX_RegFieldWrapper<"roiHeight", 16, 48>;
def reg_RoiCfg : NPUReg40XX_RegisterWrapper<"RoiCfg", 8, 0x68, ["X_coord", "Y_coord", "roiWidth", "roiHeight"]>;

// OutScaleSize
def bf_OutScale0_width : NPUReg40XX_RegFieldWrapper<"outScale0_width", 16, 0>;
def bf_OutScale0_height : NPUReg40XX_RegFieldWrapper<"outScale0_height", 16, 16>;
def bf_OutScale1_width : NPUReg40XX_RegFieldWrapper<"outScale1_width", 16, 32>;
def bf_OutScale1_height : NPUReg40XX_RegFieldWrapper<"outScale1_height", 16, 48>;
def reg_OutScaleSize : NPUReg40XX_RegisterWrapper<"OutScaleSize", 8, 0x70, ["outScale0_width", "outScale0_height", "outScale1_width", "outScale1_height"]>;

// ScPSY
def bf_PsSc0Y : NPUReg40XX_RegFieldWrapper<"psSc0Y", 32, 0>;
def bf_PsSc1Y : NPUReg40XX_RegFieldWrapper<"psSc1Y", 32, 32>;
def reg_ScPSY : NPUReg40XX_RegisterWrapper<"ScPSY", 8, 0x78, ["psSc0Y", "psSc1Y"]>;

// ScPSUV
def bf_PsSc0UV : NPUReg40XX_RegFieldWrapper<"psSc0UV", 32, 0>;
def bf_PsSc1UV : NPUReg40XX_RegFieldWrapper<"psSc1UV", 32, 32>;
def reg_ScPSUV : NPUReg40XX_RegisterWrapper<"ScPSUV", 8, 0x80, ["psSc0UV", "psSc1UV"]>;

// OutLS
def bf_LsSc0Y : NPUReg40XX_RegFieldWrapper<"lsSc0Y", 16, 0>;
def bf_LsSc1Y : NPUReg40XX_RegFieldWrapper<"lsSc1Y", 16, 16>;
def bf_LsSc0UV : NPUReg40XX_RegFieldWrapper<"lsSc0UV", 16, 32>;
def bf_LsSc1UV : NPUReg40XX_RegFieldWrapper<"lsSc1UV", 16, 48>;
def reg_OutLS : NPUReg40XX_RegisterWrapper<"OutLS", 8, 0x88, ["lsSc0Y", "lsSc1Y", "lsSc0UV", "lsSc1UV"]>;

// ScOffset
def bf_VSc_offset : NPUReg40XX_RegFieldWrapper<"vSc_offset", 32, 0>;
def bf_HSc_offset : NPUReg40XX_RegFieldWrapper<"hSc_offset", 32, 32>;
def reg_ScOffset : NPUReg40XX_RegisterWrapper<"ScOffset", 8, 0x90, ["vSc_offset", "hSc_offset"]>;

// ScFactor
def bf_VSc_factor : NPUReg40XX_RegFieldWrapper<"vSc_factor", 32, 0>;
def bf_HSc_factor : NPUReg40XX_RegFieldWrapper<"hSc_factor", 32, 32>;
def reg_ScFactor : NPUReg40XX_RegisterWrapper<"ScFactor", 8, 0x98, ["vSc_factor", "hSc_factor"]>;

def bf_BarGateMaskLO : NPUReg40XX_RegFieldWrapper<"barGateMaskLO", 64, 0>;
def reg_BarGateMaskLO : NPUReg40XX_RegisterWrapper<"barGateMaskLO", 8, 0xA0, ["barGateMaskLO"]>;
def bf_BarGateMaskHI : NPUReg40XX_RegFieldWrapper<"barGateMaskHI", 64, 0>;
def reg_BarGateMaskHI : NPUReg40XX_RegisterWrapper<"barGateMaskHI", 8, 0xA8, ["barGateMaskHI"]>;
def bf_BarUpdateLO : NPUReg40XX_RegFieldWrapper<"barUpdateLO", 64, 0>;
def reg_BarUpdateLO : NPUReg40XX_RegisterWrapper<"barUpdateLO", 8, 0xB0, ["barUpdateLO"]>;
def bf_BarUpdateHI : NPUReg40XX_RegFieldWrapper<"barUpdateHI", 64, 0>;
def reg_BarUpdateHI : NPUReg40XX_RegisterWrapper<"barUpdateHI", 8, 0xB8, ["barUpdateHI"]>;

// Media BarriersScheduling
def reg_MediaBarriersSchedulingCfg : NPUReg40XX_RegisterWrapper<"media_barriers_sched_", 8, 0xE0, ["start_after_", "clean_after_"]>;

// uint8_t pad0_[8];
def bf_Pad8_0 : NPUReg40XX_RegFieldWrapper<"pad8_0", 32, 0>;
def reg_Pad8_0 : NPUReg40XX_RegisterWrapper<"pad8_0", 4, 0xE8, ["pad8_0"]>;
def bf_Pad8_1 : NPUReg40XX_RegFieldWrapper<"pad8_1", 32, 0>;
def reg_Pad8_1 : NPUReg40XX_RegisterWrapper<"pad8_1", 4, 0xEC, ["pad8_1"]>;

// M2I descriptor
def reg_mapped_M2IDescriptor : NPUReg40XX_RegMappedWrapper<"VpuMediaTask",
    ["inAddr0", "inSize0", "inAddr1", "inSize1", "inAddr2", "inSize2", "IOCfg", "NormFactor_0", "NormFactor_1",
    "NormFactor_2", "PSOB", "nextDesc", "HWPAddrHI", "RoiDef", "RoiCfg", "OutScaleSize", "ScPSY", "ScPSUV", "OutLS",
    "ScOffset", "ScFactor", "barGateMaskLO", "barGateMaskHI", "barUpdateLO", "barUpdateHI",
    "media_barriers_sched_", "pad8_0", "pad8_1"]>;


// WorkItem registers

def bf_WI_desc_ptr : NPUReg40XX_RegFieldWrapper<"desc_ptr", 64, 0>;
def reg_WI_desc_ptr : NPUReg40XX_RegisterWrapper<"desc_ptr", 8, 0x0, ["desc_ptr"]>;

def bf_WI_type : NPUReg40XX_RegFieldWrapper<"wi_type", 8, 0>;
def reg_WI_type : NPUReg40XX_RegisterWrapper<"wi_type", 1, 0x8, ["wi_type"]>;

def bf_WI_unit : NPUReg40XX_RegFieldWrapper<"wi_unit", 8, 0>;
def reg_WI_unit : NPUReg40XX_RegisterWrapper<"wi_unit", 1, 0x9, ["wi_unit"]>;

def bf_WI_sub_unit : NPUReg40XX_RegFieldWrapper<"wi_sub_unit", 8, 0>;
def reg_WI_sub_unit : NPUReg40XX_RegisterWrapper<"wi_sub_unit", 1, 0xA, ["wi_sub_unit"]>;

// in API we have pas with 53 bytes, but in compiler we have sanity check for 8 byte per elem -> need 7 pads
def bf_Pad_8_0 : NPUReg40XX_RegFieldWrapper<"pad_8_0", 64, 0>;
def reg_Pad_8_0 : NPUReg40XX_RegisterWrapper<"pad_8_0", 8, 0xB, ["pad_8_0"]>;

def bf_Pad_8_1 : NPUReg40XX_RegFieldWrapper<"pad_8_1", 64, 0>;
def reg_Pad_8_1 : NPUReg40XX_RegisterWrapper<"pad_8_1", 8, 0x13, ["pad_8_1"]>;

def bf_Pad_8_2 : NPUReg40XX_RegFieldWrapper<"pad_8_2", 64, 0>;
def reg_Pad_8_2 : NPUReg40XX_RegisterWrapper<"pad_8_2", 8, 0x1B, ["pad_8_2"]>;

def bf_Pad_8_3 : NPUReg40XX_RegFieldWrapper<"pad_8_3", 64, 0>;
def reg_Pad_8_3 : NPUReg40XX_RegisterWrapper<"pad_8_3", 8, 0x23, ["pad_8_3"]>;

def bf_Pad_8_4 : NPUReg40XX_RegFieldWrapper<"pad_8_4", 64, 0>;
def reg_Pad_8_4 : NPUReg40XX_RegisterWrapper<"pad_8_4", 8, 0x2B, ["pad_8_4"]>;

def bf_Pad_8_5 : NPUReg40XX_RegFieldWrapper<"pad_8_5", 64, 0>;
def reg_Pad_8_5 : NPUReg40XX_RegisterWrapper<"pad_8_5", 8, 0x33, ["pad_8_5"]>;

def bf_Pad_8_6 : NPUReg40XX_RegFieldWrapper<"pad_5_6", 40, 0>;
def reg_Pad_8_6 : NPUReg40XX_RegisterWrapper<"pad_5_6", 5, 0x3B, ["pad_5_6"]>;


def reg_mapped_WorkItemDescriptor : NPUReg40XX_RegMappedWrapper<"WorkItem",
    ["desc_ptr", "wi_type", "wi_unit", "wi_sub_unit", "pad_8_0", "pad_8_1", "pad_8_2", "pad_8_3", "pad_8_4", "pad_8_5", "pad_5_6"]>;

// TaskInfo registers
def bf_TIDescPtr : NPUReg40XX_RegFieldWrapper<"ti_desc_ptr", 64, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TIDescPtr : NPUReg40XX_RegisterWrapper<"ti_desc_ptr", 8, 0x0, ["ti_desc_ptr"]>;

def bf_TIType : NPUReg40XX_RegFieldWrapper<"ti_type", 8, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TIType : NPUReg40XX_RegisterWrapper<"ti_type", 1, 0x8, ["ti_type"]>;

def bf_TIUnit : NPUReg40XX_RegFieldWrapper<"ti_unit", 8, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TIUnit : NPUReg40XX_RegisterWrapper<"ti_unit", 1, 0x9, ["ti_unit"]>;

def bf_TISubUnit : NPUReg40XX_RegFieldWrapper<"ti_sub_unit", 8, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TISubUnit : NPUReg40XX_RegisterWrapper<"ti_sub_unit", 1, 0xA, ["ti_sub_unit"]>;

def bf_TaskInfoPad1 : NPUReg40XX_RegFieldWrapper<"ti_pad1", 8, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TaskInfoPad1 : NPUReg40XX_RegisterWrapper<"ti_pad1", 1, 0xB, ["ti_pad1"]>;

def bf_TILinkedList : NPUReg40XX_RegFieldWrapper<"ti_linked_list_nodes", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TILinkedList : NPUReg40XX_RegisterWrapper<"ti_linked_list_nodes", 4, 0xC, ["ti_linked_list_nodes"]>;

def bf_TIDescrRefOffset : NPUReg40XX_RegFieldWrapper<"ti_descr_ref_offset", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TIDescrRefOffset : NPUReg40XX_RegisterWrapper<"ti_descr_ref_offset", 4, 0x10, ["ti_descr_ref_offset"]>;

def bf_TIParentDescrRefOffset : NPUReg40XX_RegFieldWrapper<"ti_parent_descr_ref_offset", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TIParentDescrRefOffset : NPUReg40XX_RegisterWrapper<"ti_parent_descr_ref_offset", 4, 0x14, ["ti_parent_descr_ref_offset"]>;

def bf_TIEnqueueingTaskConfig : NPUReg40XX_RegFieldWrapper<"ti_enqueueing_task_config", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TIEnqueueingTaskConfig : NPUReg40XX_RegisterWrapper<"ti_enqueueing_task_config", 4, 0x18, ["ti_enqueueing_task_config"]>;

def bf_TIWorkItemRef : NPUReg40XX_RegFieldWrapper<"ti_work_item_ref", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_TIWorkItemRef : NPUReg40XX_RegisterWrapper<"ti_work_item_ref", 4, 0x1C, ["ti_work_item_ref"]>;

def reg_mapped_TaskInfoDescriptor : NPUReg40XX_RegMappedWrapper<"VpuTaskInfo",
    ["ti_desc_ptr", "ti_type", "ti_unit", "ti_sub_unit", "ti_pad1", "ti_linked_list_nodes", "ti_descr_ref_offset", "ti_parent_descr_ref_offset", "ti_enqueueing_task_config", "ti_work_item_ref"]>;


// TaskBarrierMap registers

def bf_TaskBarrierNextSameId : NPUReg40XX_RegFieldWrapper<"tb_next_same_id", 32, 0>;
def reg_TaskBarrierNextSameId : NPUReg40XX_RegisterWrapper<"tb_next_same_id", 4, 0x0, ["tb_next_same_id"]>;

def bf_TaskBarrierProducerCount : NPUReg40XX_RegFieldWrapper<"tb_producer_count", 16, 0>;
def reg_TaskBarrierProducerCount : NPUReg40XX_RegisterWrapper<"tb_producer_count", 2, 0x4, ["tb_producer_count"]>;

def bf_TaskBarrierConsumerCount : NPUReg40XX_RegFieldWrapper<"tb_consumer_count", 16, 0>;
def reg_TaskBarrierConsumerCount : NPUReg40XX_RegisterWrapper<"tb_consumer_count", 2, 0x6, ["tb_consumer_count"]>;

def bf_TaskBarrierRealId : NPUReg40XX_RegFieldWrapper<"tb_real_id", 8, 0>;
def reg_TaskBarrierRealId : NPUReg40XX_RegisterWrapper<"tb_real_id", 1, 0x8, ["tb_real_id"]>;

def bf_TaskBarrierPad3 : NPUReg40XX_RegFieldWrapper<"tb_pad3", 24, 0>;
def reg_TaskBarrierPad3 : NPUReg40XX_RegisterWrapper<"tb_pad3", 3, 0x9, ["tb_pad3"]>;

def bf_TaskBarrierWorkItemIdx : NPUReg40XX_RegFieldWrapper<"tb_work_item_idx", 32, 0>;
def reg_TaskBarrierWorkItemIdx : NPUReg40XX_RegisterWrapper<"tb_work_item_idx", 4, 0xC, ["tb_work_item_idx"]>;

def bf_TaskBArrierEnqueueCount : NPUReg40XX_RegFieldWrapper<"tb_enqueue_count", 32, 0>;
def reg_TaskBArrierEnqueueCount : NPUReg40XX_RegisterWrapper<"tb_enqueue_count", 4, 0x10, ["tb_enqueue_count"]>;

def bf_TaskBArrierReservedNextEnqueueId : NPUReg40XX_RegFieldWrapper<"tb_reserved_next_enqueue_id", 32, 0>;
def ref_TaskBArrierReservedNextEnqueueId : NPUReg40XX_RegisterWrapper<"tb_reserved_next_enqueue_id", 4, 0x14, ["tb_reserved_next_enqueue_id"]>;

// TaskBarrierMap descriptor
def reg_mapped_TaskBarrierMapDescriptor : NPUReg40XX_RegMappedWrapper<"vpuTaskBarrierMap",
    ["tb_next_same_id", "tb_producer_count", "tb_consumer_count", "tb_real_id", "tb_pad3", "tb_work_item_idx", "tb_enqueue_count", "tb_reserved_next_enqueue_id"]>;


// BarrierReferenceMap registers

def bf_BRPhysicalBarrier : NPUReg40XX_RegFieldWrapper<"br_physical_barrier", 16, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_BRPhysicalBarrier : NPUReg40XX_RegisterWrapper<"br_physical_barrier", 2, 0x0, ["br_physical_barrier"]>;

def bf_BRProducerCount : NPUReg40XX_RegFieldWrapper<"br_producer_count", 16, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_BRProducerCount : NPUReg40XX_RegisterWrapper<"br_producer_count", 2, 0x2, ["br_producer_count"]>;

def bf_BRConsumerCount : NPUReg40XX_RegFieldWrapper<"br_consumer_count", 16, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_BRConsumerCount : NPUReg40XX_RegisterWrapper<"br_consumer_count", 2, 0x4, ["br_consumer_count"]>;

def bf_BRPad2 : NPUReg40XX_RegFieldWrapper<"br_pad2", 16, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_BRPad2 : NPUReg40XX_RegisterWrapper<"br_pad2", 2, 0x6, ["br_pad2"]>;

def bf_BRProducersRefOffset : NPUReg40XX_RegFieldWrapper<"br_producers_ref_offset", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_BRProducersRefOffset : NPUReg40XX_RegisterWrapper<"br_producers_ref_offset", 4, 0x8, ["br_producers_ref_offset"]>;

def bf_BRConsumersRefOffset : NPUReg40XX_RegFieldWrapper<"br_consumers_ref_offset", 32, 0, "UINT", NPUReg40XX_Version<11, 4, 10>>;
def reg_BRConsumersRefOffset : NPUReg40XX_RegisterWrapper<"br_consumers_ref_offset", 4, 0xC, ["br_consumers_ref_offset"]>;

def reg_mapped_BarrierReferenceMapDescriptor : NPUReg40XX_RegMappedWrapper<"BarrierReferenceMap",
    ["br_physical_barrier", "br_producer_count", "br_consumer_count", "br_pad2", "br_producers_ref_offset", "br_consumers_ref_offset"]>;

#endif
