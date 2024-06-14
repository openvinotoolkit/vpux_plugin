//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <gtest/gtest.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

TEST(NPUReg40XX_dialect_loader, RegFieldDclr) {
    vpux::NPUReg40XX::RegField_ac_baseType acBaseType;
    vpux::NPUReg40XX::RegField_gridType gridType;
    vpux::NPUReg40XX::RegField_odu_se_sizeType oduSeSizeType;

    auto acBaseTypeWidth = acBaseType.getRegFieldWidth();
    auto gridTypeWidth = gridType.getRegFieldWidth();
    auto oduSeSizeTypeWidth = oduSeSizeType.getRegFieldWidth();

    EXPECT_EQ(acBaseTypeWidth, 28);
    EXPECT_EQ(gridTypeWidth, 1);
    EXPECT_EQ(oduSeSizeTypeWidth, 32);
}

TEST(NPUReg40XX_dialect_loader, RegisterDclr) {
    vpux::NPUReg40XX::Register_base_offset_bType baseOffsetType;
    vpux::NPUReg40XX::Register_elop_scaleType elopScaleType;
    vpux::NPUReg40XX::Register_invar_ptrType invarPtrType;

    auto baseOffsetTypeSize = baseOffsetType.getRegSize();
    auto elopScaleTypeSize = elopScaleType.getRegSize();
    auto invarPtrTypeSize = invarPtrType.getRegSize();

    EXPECT_EQ(baseOffsetTypeSize, 32);
    EXPECT_EQ(elopScaleTypeSize, 32);
    EXPECT_EQ(invarPtrTypeSize, 32);
}
