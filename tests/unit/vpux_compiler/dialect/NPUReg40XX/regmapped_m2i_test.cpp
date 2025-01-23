//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/descriptors.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;
using namespace vpux::NPUReg40XX;

class NPUReg40XX_M2IRegisterTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuMediaTask, vpux::NPUReg40XX::Descriptors::VpuMediaTask> {};

#define TEST_NPU4_M2I_REG_FIELD(FieldType, DescriptorMember)                                                   \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg40XX_M2IRegisterTest, FieldType, vpux::NPUReg40XX::Fields::FieldType, \
                                   DescriptorMember, 0)

#define TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(ParentRegType, FieldType, DescriptorMember)             \
    HELPER_TEST_NPU_MULTIPLE_REGS_FIELD(NPUReg40XX_M2IRegisterTest, ParentRegType##__##FieldType, \
                                        vpux::NPUReg40XX::Registers::ParentRegType,               \
                                        vpux::NPUReg40XX::Fields::FieldType, DescriptorMember, 0)

TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inAddr0, inAddr, standard.buff_desc_.inAddr0)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inSize0, width, standard.buff_desc_.inSize0.width)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inSize0, height, standard.buff_desc_.inSize0.height)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inSize0, ls, standard.buff_desc_.inSize0.ls)
TEST_NPU4_M2I_REG_FIELD(pid, standard.buff_desc_.inSize0.PID)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inAddr1, inAddr, standard.buff_desc_.inAddr1)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inSize1, width, standard.buff_desc_.inSize1.width)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inSize1, height, standard.buff_desc_.inSize1.height)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inSize1, ls, standard.buff_desc_.inSize1.ls)
TEST_NPU4_M2I_REG_FIELD(HWPEN, standard.buff_desc_.inSize1.HWPEN)
TEST_NPU4_M2I_REG_FIELD(ExtHDR, standard.buff_desc_.inSize1.ExtHDR)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inAddr2, inAddr, standard.buff_desc_.inAddr2)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inSize2, width, standard.buff_desc_.inSize2.width)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inSize2, height, standard.buff_desc_.inSize2.height)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(inSize2, ls, standard.buff_desc_.inSize2.ls)
TEST_NPU4_M2I_REG_FIELD(inFormat, standard.buff_desc_.IOCfg.inFormat)
TEST_NPU4_M2I_REG_FIELD(outFormat, standard.buff_desc_.IOCfg.outFormat)
TEST_NPU4_M2I_REG_FIELD(numRois, standard.buff_desc_.IOCfg.numRois)
TEST_NPU4_M2I_REG_FIELD(sampleType, standard.buff_desc_.IOCfg.sampleType)
TEST_NPU4_M2I_REG_FIELD(operations, standard.buff_desc_.IOCfg.operations)
TEST_NPU4_M2I_REG_FIELD(IFC, standard.buff_desc_.IOCfg.IFC)
TEST_NPU4_M2I_REG_FIELD(IRQMask, standard.buff_desc_.IOCfg.IRQMask)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_0, NormFact0, standard.buff_desc_.normFactor[0].NormFact0)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_0, NormFact1, standard.buff_desc_.normFactor[0].NormFact1)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_0, NormFact2, standard.buff_desc_.normFactor[0].NormFact2)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_0, NormFact3, standard.buff_desc_.normFactor[0].NormFact3)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_1, NormFact0, standard.buff_desc_.normFactor[1].NormFact0)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_1, NormFact1, standard.buff_desc_.normFactor[1].NormFact1)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_1, NormFact2, standard.buff_desc_.normFactor[1].NormFact2)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_1, NormFact3, standard.buff_desc_.normFactor[1].NormFact3)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_2, NormFact0, standard.buff_desc_.normFactor[2].NormFact0)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_2, NormFact1, standard.buff_desc_.normFactor[2].NormFact1)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_2, NormFact2, standard.buff_desc_.normFactor[2].NormFact2)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(NormFactor_2, NormFact3, standard.buff_desc_.normFactor[2].NormFact3)
TEST_NPU4_M2I_REG_FIELD(inPS, standard.buff_desc_.PSOB.inPS)
TEST_NPU4_M2I_REG_FIELD(outBase, standard.buff_desc_.PSOB.outBase)
TEST_NPU4_M2I_REG_FIELD(HWPAddrLO, standard.buff_desc_.PSOB.HWPAddrLO)
TEST_NPU4_M2I_REG_FIELD(nextDesc, standard.buff_desc_.nextDesc)
TEST_NPU4_M2I_REG_FIELD(HWPAddrHI, standard.buff_desc_.HWPAddrHI)
TEST_NPU4_M2I_REG_FIELD(roiBase, standard.roi_desc_.roiDef.roiBase)
TEST_NPU4_M2I_REG_FIELD(outFormatLocal, standard.roi_desc_.roiDef.outFormatLocal)
TEST_NPU4_M2I_REG_FIELD(samlingTypeLocal, standard.roi_desc_.roiDef.samlingTypeLocal)
TEST_NPU4_M2I_REG_FIELD(OFC, standard.roi_desc_.roiDef.OFC)
TEST_NPU4_M2I_REG_FIELD(IRQLocal, standard.roi_desc_.roiDef.IRQLocal)
TEST_NPU4_M2I_REG_FIELD(HWProfEN, standard.roi_desc_.roiDef.HWProfEN)
TEST_NPU4_M2I_REG_FIELD(X_coord, standard.roi_desc_.roiCfg.X_coord)
TEST_NPU4_M2I_REG_FIELD(Y_coord, standard.roi_desc_.roiCfg.Y_coord)
TEST_NPU4_M2I_REG_FIELD(roiWidth, standard.roi_desc_.roiCfg.roiWidth)
TEST_NPU4_M2I_REG_FIELD(roiHeight, standard.roi_desc_.roiCfg.roiHeight)
TEST_NPU4_M2I_REG_FIELD(outScale0_width, standard.roi_desc_.outScaleSize.outScale0_width)
TEST_NPU4_M2I_REG_FIELD(outScale0_height, standard.roi_desc_.outScaleSize.outScale0_height)
TEST_NPU4_M2I_REG_FIELD(outScale1_width, standard.roi_desc_.outScaleSize.outScale1_width)
TEST_NPU4_M2I_REG_FIELD(outScale1_height, standard.roi_desc_.outScaleSize.outScale1_height)
TEST_NPU4_M2I_REG_FIELD(psSc0Y, standard.roi_desc_.psScY.psSc0Y)
TEST_NPU4_M2I_REG_FIELD(psSc1Y, standard.roi_desc_.psScY.psSc1Y)
TEST_NPU4_M2I_REG_FIELD(psSc0UV, standard.roi_desc_.psScUV.psSc0UV)
TEST_NPU4_M2I_REG_FIELD(psSc1UV, standard.roi_desc_.psScUV.psSc1UV)
TEST_NPU4_M2I_REG_FIELD(lsSc0Y, standard.roi_desc_.outLS.lsSc0Y)
TEST_NPU4_M2I_REG_FIELD(lsSc1Y, standard.roi_desc_.outLS.lsSc1Y)
TEST_NPU4_M2I_REG_FIELD(lsSc0UV, standard.roi_desc_.outLS.lsSc0UV)
TEST_NPU4_M2I_REG_FIELD(lsSc1UV, standard.roi_desc_.outLS.lsSc1UV)
TEST_NPU4_M2I_REG_FIELD(vSc_offset, standard.roi_desc_.ScOff.vSc_offset)
TEST_NPU4_M2I_REG_FIELD(hSc_offset, standard.roi_desc_.ScOff.hSc_offset)
TEST_NPU4_M2I_REG_FIELD(vSc_factor, standard.roi_desc_.ScFactor.vSc_factor)
TEST_NPU4_M2I_REG_FIELD(hSc_factor, standard.roi_desc_.ScFactor.hSc_factor)
TEST_NPU4_M2I_REG_FIELD(barGateMaskLO, standard.roi_desc_.barGateMaskLO)
TEST_NPU4_M2I_REG_FIELD(barGateMaskHI, standard.roi_desc_.barGateMaskHI)
TEST_NPU4_M2I_REG_FIELD(barUpdateLO, standard.roi_desc_.barUpdateLO)
TEST_NPU4_M2I_REG_FIELD(barUpdateHI, standard.roi_desc_.barUpdateHI)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(media_barriers_sched_, start_after_, barriers_sched_.start_after_)
TEST_NPU4_M2I_MULTIPLE_REGS_FIELD(media_barriers_sched_, clean_after_, barriers_sched_.clean_after_)
