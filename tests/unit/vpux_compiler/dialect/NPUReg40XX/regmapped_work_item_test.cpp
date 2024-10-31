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

#define CREATE_HW_DMA_DESC(field, value)                                             \
    [] {                                                                             \
        nn_public::VpuWorkItem hwWorkItemDesc;                                       \
        memset(reinterpret_cast<void*>(&hwWorkItemDesc), 0, sizeof(hwWorkItemDesc)); \
        hwWorkItemDesc.field = value;                                                \
        return hwWorkItemDesc;                                                       \
    }()

using mappedRegValues = std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>;

class NPUReg40XX_VpuWorkItemTest : public testing::TestWithParam<std::pair<mappedRegValues, nn_public::VpuWorkItem>> {
    std::unique_ptr<mlir::MLIRContext> ctx;
    std::unique_ptr<mlir::OpBuilder> builder;

public:
    NPUReg40XX_VpuWorkItemTest() {
        auto registry = vpux::createDialectRegistry();

        ctx = std::make_unique<mlir::MLIRContext>();
        ctx->loadDialect<vpux::NPUReg40XX::NPUReg40XXDialect>();

        builder = std::make_unique<mlir::OpBuilder>(ctx.get());
    }
    void compare() {
        const auto params = GetParam();

        // initialize regMapped register with values
        auto defValues = vpux::NPUReg40XX::RegMapped_WorkItemType::getZeroInitilizationValues();
        vpux::VPURegMapped::updateRegMappedInitializationValues(defValues, params.first);

        auto regMappedWorkItemDesc = vpux::NPUReg40XX::RegMapped_WorkItemType::get(*builder, defValues);

        // serialize regMapped register
        auto serializedRegMappedWorkItemDesc = regMappedWorkItemDesc.serialize();

        // compare
        EXPECT_EQ(sizeof(params.second), serializedRegMappedWorkItemDesc.size());
        EXPECT_TRUE(memcmp(&params.second, serializedRegMappedWorkItemDesc.data(), sizeof(params.second)) == 0);
    }
};

TEST_P(NPUReg40XX_VpuWorkItemTest, CheckFiledsConsistency) {
    this->compare();
}

std::vector<std::pair<mappedRegValues, nn_public::VpuWorkItem>> workItemFieldSet = {
        {{
                 {"desc_ptr", {{"desc_ptr", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(wi_desc_ptr, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"wi_type", {{"wi_type", 1}}},
         },
         CREATE_HW_DMA_DESC(type, nn_public::VpuWorkItem::VpuTaskType::DMA)},
        {{
                 {"wi_unit", {{"wi_unit", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(unit, 0xFF)},
        {{
                 {"wi_sub_unit", {{"wi_sub_unit", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(sub_unit, 0xFF)},
};

INSTANTIATE_TEST_SUITE_P(NPUReg40XX_MappedRegs, NPUReg40XX_VpuWorkItemTest, testing::ValuesIn(workItemFieldSet));
