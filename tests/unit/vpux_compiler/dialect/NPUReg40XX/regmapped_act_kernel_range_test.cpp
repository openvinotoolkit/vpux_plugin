//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_40xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

using namespace npu40xx;

#define CREATE_HW_DMA_DESC(field, value)                                                         \
    [] {                                                                                         \
        nn_public::VpuActKernelRange hwActKernelRangeDesc;                                       \
        memset(reinterpret_cast<void*>(&hwActKernelRangeDesc), 0, sizeof(hwActKernelRangeDesc)); \
        hwActKernelRangeDesc.field = value;                                                      \
        return hwActKernelRangeDesc;                                                             \
    }()

class NPUReg40XX_NpuActKernelRangeTest :
        public MLIR_RegMappedNPUReg40XXUnitBase<nn_public::VpuActKernelRange,
                                                vpux::NPUReg40XX::RegMapped_VpuActKernelRangeType> {};

TEST_P(NPUReg40XX_NpuActKernelRangeTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, nn_public::VpuActKernelRange>> actKernelRangeFieldSet = {
        {{
                 {"type", {{"type", 0x00}}},
         },
         CREATE_HW_DMA_DESC(type, nn_public::VpuActWLType::WL_KERNEL)},
        {{
                 {"kernel_entry", {{"kernel_entry", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(kernel_entry, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"text_window_base", {{"text_window_base", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(text_window_base, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"code_size", {{"code_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(code_size, 0xFFFFFFFF)},
        {{
                 {"data_sec_size", {{"data_sec_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(data_sec_size, 0xFFFFFFFF)},
        {{
                 {"kernel_invo_count", {{"kernel_invo_count", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(kernel_invo_count, 0xFFFFFFFF)},
};

INSTANTIATE_TEST_CASE_P(NPUReg40XX_MappedRegs, NPUReg40XX_NpuActKernelRangeTest,
                        testing::ValuesIn(actKernelRangeFieldSet));
