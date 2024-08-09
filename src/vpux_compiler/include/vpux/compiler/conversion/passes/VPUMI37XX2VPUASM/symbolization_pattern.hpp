//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_type_converter.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/utils/symbolization.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

template <typename OperationType>
class VPUASMSymbolizationPattern : public SymbolizationPattern<OperationType> {
public:
    using Base = VPUASMSymbolizationPattern<OperationType>;
    using SymbolMapper = typename SymbolizationPattern<OperationType>::SymbolMapper;

    VPUASMSymbolizationPattern(mlir::func::FuncOp netFunc, SymbolizationTypeConverter& typeConverter,
                               SymbolMapper& mapper, mlir::MLIRContext* ctx, Logger log)
            : SymbolizationPattern<OperationType>(netFunc, typeConverter, mapper, ctx), _log(log) {
    }

    // E#69730: would be cleaner to type-check at template level if Op itself declares the OneResult interface
    llvm::SmallVector<mlir::FlatSymbolRefAttr> getSymbolicNames(OperationType op, size_t) override {
        auto fullName = OperationType::getOperationName();

        auto opName = fullName.drop_front(VPUMI37XX::VPUMI37XXDialect::getDialectNamespace().size() + 1);

        mlir::Operation* base = op.getOperation();
        VPUX_THROW_UNLESS(base->getResults().size() == 1,
                          "Default symbolic converter only supports ops with exactly one result. For {0} got {1}",
                          fullName, base->getResults().size());
        auto indexType = base->getResult(0).getType().dyn_cast<VPURegMapped::IndexType>();

        VPUX_THROW_UNLESS(indexType,
                          " Can't use the generic symbolizer if for an Op that does not return IndexType {0}",
                          fullName);

        auto index = std::to_string(indexType.getValue());
        auto symName = mlir::StringAttr::get(op.getContext(), opName + index);
        return {mlir::FlatSymbolRefAttr::get(symName)};
    }

protected:
    mlir::ArrayAttr vectorizeBarriers(mlir::Operation::operand_range&& barrierRange) const {
        mlir::MLIRContext* ctx = this->getContext();
        llvm::SmallVector<mlir::Attribute> barrierVec(barrierRange.size());

        auto u8Attr = [&ctx](uint8_t value) -> mlir::IntegerAttr {
            auto u8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
            return mlir::IntegerAttr::get(u8Type, value);
        };

        for (auto barrier : llvm::enumerate(barrierRange)) {
            auto barrierVal = barrier.value();
            auto barrierIdx = barrier.index();
            // hard-cast since it should a by-default-expected relationship
            auto barrierOp = mlir::cast<VPUMI37XX::ConfigureBarrierOp>(barrierVal.getDefiningOp());

            barrierVec[barrierIdx] = u8Attr(barrierOp.getId());
        }
        return mlir::ArrayAttr::get(ctx, barrierVec);
    };

    Logger _log;
};

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
