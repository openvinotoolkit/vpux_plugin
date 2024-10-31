//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/init.hpp"

#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/types.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

#include <npu_40xx_nnrt.hpp>

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace npu40xx;

#define CREATE_HW_DMA_DESC(field, value)                                                   \
    [] {                                                                                   \
        nn_public::VpuTaskInfo hwVpuTaskInfoDesc;                                          \
        memset(reinterpret_cast<void*>(&hwVpuTaskInfoDesc), 0, sizeof(hwVpuTaskInfoDesc)); \
        hwVpuTaskInfoDesc.field = value;                                                   \
        return hwVpuTaskInfoDesc;                                                          \
    }()

using mappedRegValues = std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>;

class NPUReg40XX_VpuTaskInfoTest : public testing::TestWithParam<std::pair<mappedRegValues, nn_public::VpuTaskInfo>> {
    std::unique_ptr<mlir::MLIRContext> ctx;
    std::unique_ptr<mlir::OpBuilder> builder;

public:
    NPUReg40XX_VpuTaskInfoTest() {
        auto registry = vpux::createDialectRegistry();

        ctx = std::make_unique<mlir::MLIRContext>();
        ctx->loadDialect<vpux::NPUReg40XX::NPUReg40XXDialect>();

        builder = std::make_unique<mlir::OpBuilder>(ctx.get());
    }
    void compare() {
        const auto params = GetParam();

        // initialize regMapped register with values
        auto defValues = vpux::NPUReg40XX::RegMapped_VpuTaskInfoType::getZeroInitilizationValues();
        vpux::VPURegMapped::updateRegMappedInitializationValues(defValues, params.first);

        auto regMappedTaskInfoDesc = vpux::NPUReg40XX::RegMapped_VpuTaskInfoType::get(*builder, defValues);

        // serialize regMapped register
        auto serializedRegMappedTaskInfoDesc = regMappedTaskInfoDesc.serialize();

        // compare
        EXPECT_EQ(sizeof(params.second), serializedRegMappedTaskInfoDesc.size());
        EXPECT_TRUE(memcmp(&params.second, serializedRegMappedTaskInfoDesc.data(), sizeof(params.second)) == 0);
    }
};

TEST_P(NPUReg40XX_VpuTaskInfoTest, CheckFiledsConsistency) {
    this->compare();
}

std::vector<std::pair<mappedRegValues, nn_public::VpuTaskInfo>> TaskInfoFieldSet = {
        {{
                 {"ti_desc_ptr", {{"ti_desc_ptr", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(wi_desc_ptr, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"ti_type", {{"ti_type", 1}}},
         },
         CREATE_HW_DMA_DESC(type, nn_public::VpuWorkItem::VpuTaskType::DMA)},
        {{
                 {"ti_unit", {{"ti_unit", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(unit, 0xFF)},
        {{
                 {"ti_sub_unit", {{"ti_sub_unit", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(sub_unit, 0xFF)},
        {{
                 {"ti_linked_list_nodes", {{"ti_linked_list_nodes", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(linked_list_nodes, 0xFFFF)},
        {{
                 {"ti_descr_ref_offset", {{"ti_descr_ref_offset", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(descr_ref_offset, 0xFFFF)},
        {{
                 {"ti_parent_descr_ref_offset", {{"ti_parent_descr_ref_offset", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(parent_descr_ref_offset, 0xFFFF)},
        {{
                 {"ti_enqueueing_task_config", {{"ti_enqueueing_task_config", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(enqueueing_task_config, 0xFFFF)},
        {{
                 {"ti_work_item_ref", {{"ti_work_item_ref", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(work_item_ref, 0xFFFF)},

};

INSTANTIATE_TEST_SUITE_P(NPUReg40XX_MappedRegs, NPUReg40XX_VpuTaskInfoTest, testing::ValuesIn(TaskInfoFieldSet));
