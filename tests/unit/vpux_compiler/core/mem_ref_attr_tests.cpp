//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/init.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_MemRefAttr = MLIR_UnitBase;

TEST_F(MLIR_MemRefAttr, ImplicitConversionToMemRefLayoutAttrInterface) {
    mlir::MLIRContext ctx(registry);

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const auto memRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, &ctx);

    // implicit conversion to interface must succeed
    mlir::MemRefLayoutAttrInterface interface = memRef;
    ASSERT_NE(interface, nullptr);
    // roundtrip also succeeds
    ASSERT_EQ(mlir::cast<vpux::MemRefAttr>(interface), memRef);
}

TEST_F(MLIR_MemRefAttr, ImplicitConversionWorksOnNullptrHwSpecificFields) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const DimsOrder order = DimsOrder::NCHW;
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const auto memRef = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, &ctx);

    ASSERT_TRUE(memRef.hwSpecificFields().empty());

    // implicit conversion must not fail, even though the fields are not set
    auto implicitlyConvertedSwizzling = memRef.hwSpecificField<VPUIP::SwizzlingSchemeAttr>();
    auto implicitlyConvertedCompression = memRef.hwSpecificField<VPUIP::SparsityCompressionAttr>();
    ASSERT_EQ(implicitlyConvertedSwizzling, nullptr);
    ASSERT_EQ(implicitlyConvertedCompression, nullptr);
}
