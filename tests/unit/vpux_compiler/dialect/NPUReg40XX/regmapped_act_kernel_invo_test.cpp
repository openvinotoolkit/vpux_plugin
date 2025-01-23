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

class NPUReg40XX_NpuActKernelInvocationTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuActKernelInvocation,
                                       vpux::NPUReg40XX::Descriptors::VpuActKernelInvocation> {};

#define TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(FieldType, DescriptorMember)         \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg40XX_NpuActKernelInvocationTest, FieldType, \
                                   vpux::NPUReg40XX::Fields::FieldType, DescriptorMember, 0)

#define TEST_NPU4_ACTKERNELINVOCATION_MULTIPLE_REGS_FIELD(ParentRegType, FieldType, DescriptorMember)        \
    HELPER_TEST_NPU_MULTIPLE_REGS_FIELD(NPUReg40XX_NpuActKernelInvocationTest, ParentRegType##__##FieldType, \
                                        vpux::NPUReg40XX::Registers::ParentRegType,                          \
                                        vpux::NPUReg40XX::Fields::FieldType, DescriptorMember, 0)

TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(range, range)
TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(kernel_args, kernel_args)
TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(data_window_base, data_window_base)
TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(perf_packet_out, perf_packet_out)
TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(barriers_wait_mask_hi_act, barriers.wait_mask_hi_)
TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(barriers_wait_mask_lo_act, barriers.wait_mask_lo_)
TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(barriers_post_mask_hi_act, barriers.post_mask_hi_)
TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(barriers_post_mask_lo_act, barriers.post_mask_lo_)
TEST_NPU4_ACTKERNELINVOCATION_MULTIPLE_REGS_FIELD(act_invo_barriers_sched, start_after_, barriers_sched.start_after_)
TEST_NPU4_ACTKERNELINVOCATION_MULTIPLE_REGS_FIELD(act_invo_barriers_sched, clean_after_, barriers_sched.clean_after_)
TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(invo_tile, invo_tile)
TEST_NPU4_ACTKERNELINVOCATION_REG_FIELD(kernel_range_index, kernel_range_index)
