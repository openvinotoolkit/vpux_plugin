#include "vpux/compiler/init.hpp"

#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/types.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/npu_40xx_nnrt.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace npu40xx;

#define CREATE_HW_DMA_DESC(field, value)                                                                 \
    [] {                                                                                                 \
        nn_public::VpuTaskBarrierMap hwWVpuTaskBarrierMapDesc;                                           \
        memset(reinterpret_cast<void*>(&hwWVpuTaskBarrierMapDesc), 0, sizeof(hwWVpuTaskBarrierMapDesc)); \
        hwWVpuTaskBarrierMapDesc.field = value;                                                          \
        return hwWVpuTaskBarrierMapDesc;                                                                 \
    }()

using mappedRegValues = std::map<std::string, std::map<std::string, uint64_t>>;

class NPUReg40XX_VpuTaskBarrierMapTest :
        public testing::TestWithParam<std::pair<mappedRegValues, nn_public::VpuTaskBarrierMap>> {
    std::unique_ptr<mlir::MLIRContext> ctx;
    std::unique_ptr<mlir::OpBuilder> builder;

public:
    NPUReg40XX_VpuTaskBarrierMapTest() {
        mlir::DialectRegistry registry;
        vpux::registerDialects(registry);

        ctx = std::make_unique<mlir::MLIRContext>();
        ctx->loadDialect<vpux::NPUReg40XX::NPUReg40XXDialect>();

        builder = std::make_unique<mlir::OpBuilder>(ctx.get());
    }
    void compare() {
        const auto params = GetParam();

        // initialize regMapped register with values
        auto defValues = vpux::NPUReg40XX::RegMapped_vpuTaskBarrierMapType::getZeroInitilizationValues();
        vpux::VPURegMapped::updateRegMappedInitializationValues(defValues, params.first);

        auto regMappedTaskBarrierDesc = vpux::NPUReg40XX::RegMapped_vpuTaskBarrierMapType::get(*builder, defValues);

        // serialize regMapped register
        auto serializedRegMappedWTaskBarrierDesc = regMappedTaskBarrierDesc.serialize();

        // compare
        EXPECT_EQ(sizeof(params.second), serializedRegMappedWTaskBarrierDesc.size());
        EXPECT_TRUE(memcmp(&params.second, serializedRegMappedWTaskBarrierDesc.data(), sizeof(params.second)) == 0);
    }
};

TEST_P(NPUReg40XX_VpuTaskBarrierMapTest, CheckFiledsConsistency) {
    this->compare();
}

std::vector<std::pair<mappedRegValues, nn_public::VpuTaskBarrierMap>> TaskBarrierFieldSet = {
        {{
                 {"tb_next_same_id", {{"tb_next_same_id", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(next_same_id, 0xFFFF)},
        {{
                 {"tb_producer_count", {{"tb_producer_count", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(producer_count, 0xFFFF)},
        {{
                 {"tb_consumer_count", {{"tb_consumer_count", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(consumer_count, 0xFFFF)},
        {{
                 {"tb_real_id", {{"tb_real_id", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(real_id, 0xFF)},
        {{
                 {"tb_work_item_idx", {{"tb_work_item_idx", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(work_item_idx, 0xFFFFFFFF)},
        {{
                 {"tb_enqueue_count", {{"tb_enqueue_count", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(enqueue_count, 0xFFFFFFFF)},
        {{
                 {"tb_reserved_next_enqueue_id", {{"tb_reserved_next_enqueue_id", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(reserved, 0xFFFFFFFF)},
};

INSTANTIATE_TEST_CASE_P(NPUReg40XX_MappedRegs, NPUReg40XX_VpuTaskBarrierMapTest,
                        testing::ValuesIn(TaskBarrierFieldSet));
