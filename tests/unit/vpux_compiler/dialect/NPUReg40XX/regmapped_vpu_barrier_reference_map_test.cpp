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

#define CREATE_HW_DMA_DESC(field, value)                                                               \
    [] {                                                                                               \
        nn_public::BarrierReferenceMap barrierReferenceMapDesc;                                        \
        memset(reinterpret_cast<void*>(&barrierReferenceMapDesc), 0, sizeof(barrierReferenceMapDesc)); \
        barrierReferenceMapDesc.field = value;                                                         \
        return barrierReferenceMapDesc;                                                                \
    }()

using mappedRegValues = std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>;

class NPUReg40XX_BarrierReferenceMapTest :
        public testing::TestWithParam<std::pair<mappedRegValues, nn_public::BarrierReferenceMap>> {
    std::unique_ptr<mlir::MLIRContext> ctx;
    std::unique_ptr<mlir::OpBuilder> builder;

public:
    NPUReg40XX_BarrierReferenceMapTest() {
        auto registry = vpux::createDialectRegistry();

        ctx = std::make_unique<mlir::MLIRContext>();
        ctx->loadDialect<vpux::NPUReg40XX::NPUReg40XXDialect>();

        builder = std::make_unique<mlir::OpBuilder>(ctx.get());
    }
    void compare() {
        const auto params = GetParam();

        // initialize regMapped register with values
        auto defValues = vpux::NPUReg40XX::RegMapped_BarrierReferenceMapType::getZeroInitilizationValues();
        vpux::VPURegMapped::updateRegMappedInitializationValues(defValues, params.first);

        auto regMappedBarrierReferenceMapDesc =
                vpux::NPUReg40XX::RegMapped_BarrierReferenceMapType::get(*builder, defValues);

        // serialize regMapped register
        auto serializedRegMappedBarrierReferenceMapDesc = regMappedBarrierReferenceMapDesc.serialize();

        // compare
        EXPECT_EQ(sizeof(params.second), serializedRegMappedBarrierReferenceMapDesc.size());
        EXPECT_TRUE(memcmp(&params.second, serializedRegMappedBarrierReferenceMapDesc.data(), sizeof(params.second)) ==
                    0);
    }
};

TEST_P(NPUReg40XX_BarrierReferenceMapTest, CheckFiledsConsistency) {
    this->compare();
}

std::vector<std::pair<mappedRegValues, nn_public::BarrierReferenceMap>> BarrierReferenceMapFieldSet = {
        {{
                 {"br_physical_barrier", {{"br_physical_barrier", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(physical_barrier, 0xFFFF)},
        {{
                 {"br_producer_count", {{"br_producer_count", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(producer_count, 0xFFFF)},
        {{
                 {"br_consumer_count", {{"br_consumer_count", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(consumer_count, 0xFFFF)},
        {{
                 {"br_producers_ref_offset", {{"br_producers_ref_offset", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(producers_ref_offset, 0xFFFFFFFF)},
        {{
                 {"br_consumers_ref_offset", {{"br_consumers_ref_offset", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(consumers_ref_offset, 0xFFFFFFFF)},

};

INSTANTIATE_TEST_SUITE_P(NPUReg40XX_MappedRegs, NPUReg40XX_BarrierReferenceMapTest,
                         testing::ValuesIn(BarrierReferenceMapFieldSet));
