//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

class BlobWriterTests : public ::testing::Test {
protected:
    vpux::VPUIP::BlobWriter writer;

public:
    BlobWriterTests(): writer(Logger::global(), vpux::VPU::ArchKind::NPU37XX) {
    }
};

TEST_F(BlobWriterTests, createKernelDataRef_unique_cache_if_name_is_from_substring) {
    std::string name1 = "longname1_1";
    std::string name2 = "longname1_2";
    StringRef refName1(name1.data(), 9);
    StringRef refName2(name2.data(), 9);

    uint8_t content[] = {0, 1, 2, 3};

    writer.createKernelDataRef(refName1, 0, 4, {content, 4});
    writer.createKernelDataRef(refName2, 0, 4, {content, 4});

    EXPECT_EQ(writer.getKernelData().size(), 1);
}
