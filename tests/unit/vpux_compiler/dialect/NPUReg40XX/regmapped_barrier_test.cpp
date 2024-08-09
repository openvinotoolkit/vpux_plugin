//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_40xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

using namespace npu40xx;

#define CREATE_HW_DMA_DESC(field, value)                                           \
    [] {                                                                           \
        nn_public::VpuBarrierCountConfig hwBarrierDesc;                            \
        memset(reinterpret_cast<void*>(&hwBarrierDesc), 0, sizeof(hwBarrierDesc)); \
        hwBarrierDesc.field = value;                                               \
        return hwBarrierDesc;                                                      \
    }()

class NPUReg40XX_NpuBarrierCountConfigTest :
        public MLIR_RegMappedNPUReg40XXUnitBase<nn_public::VpuBarrierCountConfig,
                                                vpux::NPUReg40XX::RegMapped_VpuBarrierCountConfigType> {};

TEST_P(NPUReg40XX_NpuBarrierCountConfigTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, nn_public::VpuBarrierCountConfig>> barrierFieldSet = {
        {{
                 {"next_same_id_", {{"next_same_id_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(next_same_id_, 0xFFFFFFFF)},
        {{
                 {"producer_count_", {{"producer_count_", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(producer_count_, 0xFFFF)},
        {{
                 {"consumer_count_", {{"consumer_count_", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(consumer_count_, 0xFFFF)},
        {{
                 {"real_id_", {{"real_id_", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(real_id_, 0xFF)},
};

INSTANTIATE_TEST_CASE_P(NPUReg40XX_MappedRegs, NPUReg40XX_NpuBarrierCountConfigTest,
                        testing::ValuesIn(barrierFieldSet));
