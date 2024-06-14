//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_40xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

using namespace npu40xx;

#define CREATE_HW_DMA_DESC(field, value)                                                       \
    [] {                                                                                       \
        nn_public::VpuActKernelInvocation hwActKernelInvoDesc;                                 \
        memset(reinterpret_cast<void*>(&hwActKernelInvoDesc), 0, sizeof(hwActKernelInvoDesc)); \
        hwActKernelInvoDesc.field = value;                                                     \
        return hwActKernelInvoDesc;                                                            \
    }()

class NPUReg40XX_NpuActKernelInvocationTest :
        public MLIR_RegMappedNPUReg40XXUnitBase<nn_public::VpuActKernelInvocation,
                                                vpux::NPUReg40XX::RegMapped_VpuActKernelInvocationType> {};

TEST_P(NPUReg40XX_NpuActKernelInvocationTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, nn_public::VpuActKernelInvocation>> actKernelInvoFieldSet = {
        {{
                 {"range", {{"range", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(range, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"kernel_args", {{"kernel_args", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(kernel_args, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"data_window_base", {{"data_window_base", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(data_window_base, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"perf_packet_out", {{"perf_packet_out", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(perf_packet_out, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"barriers_wait_mask_hi_act", {{"barriers_wait_mask_hi_act", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(barriers.wait_mask_hi_, 0xFFFFFFFF)},
        {{
                 {"barriers_wait_mask_lo_act", {{"barriers_wait_mask_lo_act", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(barriers.wait_mask_lo_, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"barriers_post_mask_hi_act", {{"barriers_post_mask_hi_act", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(barriers.post_mask_hi_, 0xFFFFFFFF)},
        {{
                 {"barriers_post_mask_lo_act", {{"barriers_post_mask_lo_act", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(barriers.post_mask_lo_, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"barriers_group_mask_act", {{"group_act", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(barriers.group_, 0xFF)},
        {{
                 {"barriers_group_mask_act", {{"mask_act", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(barriers.mask_, 0xFF)},
        {{
                 {"act_invo_barriers_sched", {{"start_after_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(barriers_sched.start_after_, 0xFFFFFFFF)},
        {{
                 {"act_invo_barriers_sched", {{"clean_after_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(barriers_sched.clean_after_, 0xFFFFFFFF)},
        {{
                 {"invo_index", {{"invo_index", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(invo_index, 0xFFFFFFFF)},
        {{
                 {"invo_tile", {{"invo_tile", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(invo_tile, 0xFFFFFFFF)},
        {{
                 {"kernel_range_index", {{"kernel_range_index", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(kernel_range_index, 0xFFFFFFFF)},
};

INSTANTIATE_TEST_CASE_P(NPUReg40XX_MappedRegs, NPUReg40XX_NpuActKernelInvocationTest,
                        testing::ValuesIn(actKernelInvoFieldSet));
