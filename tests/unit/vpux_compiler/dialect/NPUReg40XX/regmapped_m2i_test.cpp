//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_40xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

using namespace npu40xx;

#define CREATE_HW_M2I_DESC(field, value)                                   \
    [] {                                                                   \
        nn_public::VpuMediaTask hwM2IDesc;                                 \
        memset(reinterpret_cast<void*>(&hwM2IDesc), 0, sizeof(hwM2IDesc)); \
        hwM2IDesc.field = value;                                           \
        return hwM2IDesc;                                                  \
    }()

class NPUReg40XX_M2IRegisterTest :
        public MLIR_RegMappedNPUReg40XXUnitBase<nn_public::VpuMediaTask, vpux::NPUReg40XX::RegMapped_VpuMediaTaskType> {
};

TEST_P(NPUReg40XX_M2IRegisterTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, nn_public::VpuMediaTask>> m2iValuesSet = {
        {{
                 {"inAddr0", {{"inAddr", 0xFFFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inAddr0, 0xFFFFFFFF)},
        {{
                 {"inSize0", {{"width", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize0.width, 0xFF)},
        {{
                 {"inSize0", {{"height", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize0.height, 0xFF)},
        {{
                 {"inSize0", {{"ls", 0xFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize0.ls, 0xFFF)},
        {{
                 {"inSize0", {{"pid", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize0.PID, 0xF)},
        {{
                 {"inAddr1", {{"inAddr", 0xFFFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inAddr1, 0xFFFFFFFF)},
        {{
                 {"inSize1", {{"width", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize1.width, 0xFF)},
        {{
                 {"inSize1", {{"height", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize1.height, 0xFF)},
        {{
                 {"inSize1", {{"ls", 0xFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize1.ls, 0xFFF)},
        {{
                 {"inSize1", {{"HWPEN", 0x1}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize1.HWPEN, 0x1)},
        {{
                 {"inSize1", {{"ExtHDR", 0x1}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize1.ExtHDR, 0x1)},
        {{
                 {"inAddr2", {{"inAddr", 0xFFFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inAddr2, 0xFFFFFFFF)},
        {{
                 {"inSize2", {{"width", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize2.width, 0xFF)},
        {{
                 {"inSize2", {{"height", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize2.height, 0xFF)},
        {{
                 {"inSize2", {{"ls", 0xFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.inSize2.ls, 0xFFF)},
        {{
                 {"IOCfg", {{"inFormat", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.IOCfg.inFormat, 0xF)},
        {{
                 {"IOCfg", {{"outFormat", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.IOCfg.outFormat, 0xF)},
        {{
                 {"IOCfg", {{"numRois", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.IOCfg.numRois, 0xFF)},
        {{
                 {"IOCfg", {{"sampleType", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.IOCfg.sampleType, 0xF)},
        {{
                 {"IOCfg", {{"operations", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.IOCfg.operations, 0xF)},
        {{
                 {"IOCfg", {{"IFC", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.IOCfg.IFC, 0xF)},
        {{
                 {"IOCfg", {{"IRQMask", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.IOCfg.IRQMask, 0xFF)},
        {{
                 {"NormFactor_0", {{"NormFact0", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[0].NormFact0, 0xFF)},
        {{
                 {"NormFactor_0", {{"NormFact1", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[0].NormFact1, 0xFF)},
        {{
                 {"NormFactor_0", {{"NormFact2", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[0].NormFact2, 0xFF)},
        {{
                 {"NormFactor_0", {{"NormFact3", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[0].NormFact3, 0xFF)},
        {{
                 {"NormFactor_1", {{"NormFact0", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[1].NormFact0, 0xFF)},
        {{
                 {"NormFactor_1", {{"NormFact1", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[1].NormFact1, 0xFF)},
        {{
                 {"NormFactor_1", {{"NormFact2", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[1].NormFact2, 0xFF)},
        {{
                 {"NormFactor_1", {{"NormFact3", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[1].NormFact3, 0xFF)},
        {{
                 {"NormFactor_2", {{"NormFact0", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[2].NormFact0, 0xFF)},
        {{
                 {"NormFactor_2", {{"NormFact1", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[2].NormFact1, 0xFF)},
        {{
                 {"NormFactor_2", {{"NormFact2", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[2].NormFact2, 0xFF)},
        {{
                 {"NormFactor_2", {{"NormFact3", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.normFactor[2].NormFact3, 0xFF)},
        {{
                 {"PSOB", {{"inPS", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.PSOB.inPS, 0xFFFF)},
        {{
                 {"PSOB", {{"outBase", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.PSOB.outBase, 0xFF)},
        {{
                 {"PSOB", {{"HWPAddrLO", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.PSOB.HWPAddrLO, 0xFF)},
        {{
                 {"nextDesc", {{"nextDesc", 0xFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.nextDesc, 0xFFFFFF)},
        {{
                 {"HWPAddrHI", {{"HWPAddrHI", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.buff_desc_.HWPAddrHI, 0xFF)},
        {{
                 {"RoiDef", {{"roiBase", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiDef.roiBase, 0xFFFF)},
        {{
                 {"RoiDef", {{"outFormatLocal", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiDef.outFormatLocal, 0xF)},
        {{
                 {"RoiDef", {{"samlingTypeLocal", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiDef.samlingTypeLocal, 0xF)},
        {{
                 {"RoiDef", {{"OFC", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiDef.OFC, 0xF)},
        {{
                 {"RoiDef", {{"IRQLocal", 0xF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiDef.IRQLocal, 0xF)},
        {{
                 {"RoiDef", {{"HWProfEN", 0x1}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiDef.HWProfEN, 0x1)},
        {{
                 {"RoiCfg", {{"X_coord", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiCfg.X_coord, 0xFF)},
        {{
                 {"RoiCfg", {{"Y_coord", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiCfg.Y_coord, 0xFF)},
        {{
                 {"RoiCfg", {{"roiWidth", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiCfg.roiWidth, 0xFF)},
        {{
                 {"RoiCfg", {{"roiHeight", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.roiCfg.roiHeight, 0xFF)},
        {{
                 {"OutScaleSize", {{"outScale0_width", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.outScaleSize.outScale0_width, 0xFF)},
        {{
                 {"OutScaleSize", {{"outScale0_height", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.outScaleSize.outScale0_height, 0xFF)},
        {{
                 {"OutScaleSize", {{"outScale1_width", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.outScaleSize.outScale1_width, 0xFF)},
        {{
                 {"OutScaleSize", {{"outScale1_height", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.outScaleSize.outScale1_height, 0xFF)},
        {{
                 {"ScPSY", {{"psSc0Y", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.psScY.psSc0Y, 0xFFFF)},
        {{
                 {"ScPSY", {{"psSc1Y", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.psScY.psSc1Y, 0xFFFF)},
        {{
                 {"ScPSUV", {{"psSc0UV", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.psScUV.psSc0UV, 0xFFFF)},
        {{
                 {"ScPSUV", {{"psSc1UV", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.psScUV.psSc1UV, 0xFFFF)},
        {{
                 {"OutLS", {{"lsSc0Y", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.outLS.lsSc0Y, 0xFF)},
        {{
                 {"OutLS", {{"lsSc1Y", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.outLS.lsSc1Y, 0xFF)},
        {{
                 {"OutLS", {{"lsSc0UV", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.outLS.lsSc0UV, 0xFF)},
        {{
                 {"OutLS", {{"lsSc1UV", 0xFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.outLS.lsSc1UV, 0xFF)},
        {{
                 {"ScOffset", {{"vSc_offset", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.ScOff.vSc_offset, 0xFFFF)},
        {{
                 {"ScOffset", {{"hSc_offset", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.ScOff.hSc_offset, 0xFFFF)},
        {{
                 {"ScFactor", {{"vSc_factor", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.ScFactor.vSc_factor, 0xFFFF)},
        {{
                 {"ScFactor", {{"hSc_factor", 0xFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.ScFactor.hSc_factor, 0xFFFF)},
        {{
                 {"barGateMaskLO", {{"barGateMaskLO", 0xFFFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.barGateMaskLO, 0xFFFFFFFF)},
        {{
                 {"barGateMaskHI", {{"barGateMaskHI", 0xFFFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.barGateMaskHI, 0xFFFFFFFF)},
        {{
                 {"barUpdateLO", {{"barUpdateLO", 0xFFFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.barUpdateLO, 0xFFFFFFFF)},
        {{
                 {"barUpdateHI", {{"barUpdateHI", 0xFFFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(standard.roi_desc_.barUpdateHI, 0xFFFFFFFF)},
        {{
                 {"media_barriers_sched_", {{"start_after_", 0xFFFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(barriers_sched_.start_after_, 0xFFFFFFFF)},
        {{
                 {"media_barriers_sched_", {{"clean_after_", 0xFFFFFFFF}}},
         },
         CREATE_HW_M2I_DESC(barriers_sched_.clean_after_, 0xFFFFFFFF)},
};

INSTANTIATE_TEST_CASE_P(NPUReg40XX_MappedRegs, NPUReg40XX_M2IRegisterTest, testing::ValuesIn(m2iValuesSet));
