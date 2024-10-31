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
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"

class MLIR_UnitBase : public testing::Test {
public:
    MLIR_UnitBase() {
        registry = vpux::createDialectRegistry();
    }

protected:
    mlir::DialectRegistry registry;
};

class NPUSpecific_UnitTest : public MLIR_UnitBase {
public:
    NPUSpecific_UnitTest(vpux::VPU::ArchKind arch) {
        // We need to register hw-specific interfaces (e.g. NCEOpInterface) for VPU NCE ops
        auto interfacesRegistry = vpux::createInterfacesRegistry(arch);
        interfacesRegistry->registerInterfaces(registry);
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<vpux::VPU::VPUDialect>();
    }
    mlir::MLIRContext ctx;
};

namespace vpux::VPU::arch37xx {
class UnitTest : public NPUSpecific_UnitTest {
public:
    UnitTest(): NPUSpecific_UnitTest(vpux::VPU::ArchKind::NPU37XX) {
    }
};
}  // namespace vpux::VPU::arch37xx

namespace vpux::VPU::arch40xx {
class UnitTest : public NPUSpecific_UnitTest {
public:
    UnitTest(): NPUSpecific_UnitTest(vpux::VPU::ArchKind::NPU40XX) {
    }
};
}  // namespace vpux::VPU::arch40xx

using MappedRegValues = std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>;
template <typename HW_REG_TYPE, typename REG_MAPPED_TYPE>
class MLIR_RegMappedUnitBase : public testing::TestWithParam<std::pair<MappedRegValues, HW_REG_TYPE>> {
public:
    MLIR_RegMappedUnitBase() {
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
