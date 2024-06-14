//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <gtest/gtest.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/init.hpp"

class MLIR_UnitBase : public testing::Test {
public:
    MLIR_UnitBase() {
        vpux::registerDialects(registry);
        vpux::registerCommonInterfaces(registry);
    }

protected:
    mlir::DialectRegistry registry;
};

using MappedRegValues = std::map<std::string, std::map<std::string, uint64_t>>;
template <typename HW_REG_TYPE, typename REG_MAPPED_TYPE>
class MLIR_RegMappedUnitBase : public testing::TestWithParam<std::pair<MappedRegValues, HW_REG_TYPE>> {
public:
    MLIR_RegMappedUnitBase() {
        vpux::registerDialects(registry);
        vpux::registerCommonInterfaces(registry);

        ctx = std::make_unique<mlir::MLIRContext>();
    }
    void compare() {
        const auto params = this->GetParam();

        // initialize regMapped register with values
        auto defValues = REG_MAPPED_TYPE::getZeroInitilizationValues();
        vpux::VPURegMapped::updateRegMappedInitializationValues(defValues, params.first);

        auto regMappedDMADesc = REG_MAPPED_TYPE::get(*builder, defValues);

        // serialize regMapped register
        auto serializedRegMappedDMADesc = regMappedDMADesc.serialize();

        // compare
        EXPECT_EQ(sizeof(params.second), serializedRegMappedDMADesc.size());
        EXPECT_TRUE(memcmp(&params.second, serializedRegMappedDMADesc.data(), sizeof(params.second)) == 0);
    }

    mlir::DialectRegistry registry;
    std::unique_ptr<mlir::MLIRContext> ctx;
    std::unique_ptr<mlir::OpBuilder> builder;
};

template <typename HW_REG_TYPE, typename REG_MAPPED_TYPE>
class MLIR_RegMappedNPUReg37XXUnitBase : public MLIR_RegMappedUnitBase<HW_REG_TYPE, REG_MAPPED_TYPE> {
public:
    MLIR_RegMappedNPUReg37XXUnitBase() {
        this->ctx->template loadDialect<vpux::NPUReg37XX::NPUReg37XXDialect>();
        this->builder = std::make_unique<mlir::OpBuilder>(this->ctx.get());
    }
};

template <typename HW_REG_TYPE, typename REG_MAPPED_TYPE>
class MLIR_RegMappedNPUReg40XXUnitBase : public MLIR_RegMappedUnitBase<HW_REG_TYPE, REG_MAPPED_TYPE> {
public:
    MLIR_RegMappedNPUReg40XXUnitBase() {
        this->ctx->template loadDialect<vpux::NPUReg40XX::NPUReg40XXDialect>();
        this->builder = std::make_unique<mlir::OpBuilder>(this->ctx.get());
    }
};
