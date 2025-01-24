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

class NPUReg40XX_NpuActKernelRangeTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuActKernelRange, vpux::NPUReg40XX::Descriptors::VpuActKernelRange> {
};

#define TEST_NPU4_ACTKERNELRANGE_REG_FIELD(FieldType, DescriptorMember)                                              \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg40XX_NpuActKernelRangeTest, FieldType, vpux::NPUReg40XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU4_ACTKERNELRANGE_REG_FIELD(kernel_entry, kernel_entry)
TEST_NPU4_ACTKERNELRANGE_REG_FIELD(text_window_base, text_window_base)
TEST_NPU4_ACTKERNELRANGE_REG_FIELD(code_size, code_size)
TEST_NPU4_ACTKERNELRANGE_REG_FIELD(kernel_invo_count, kernel_invo_count)

TEST_F(NPUReg40XX_NpuActKernelRangeTest, typeTest) {
    const auto value = nn_public::VpuActWLType::WL_DEBUG;
    actual.write<vpux::NPUReg40XX::Fields::type>(value);
    const auto actualValue = static_cast<nn_public::VpuActWLType>(actual.read<vpux::NPUReg40XX::Fields::type>());
    EXPECT_EQ(actualValue, value);

    reference.type = value;
    ASSERT_TRUE(isContentEqual());
}
