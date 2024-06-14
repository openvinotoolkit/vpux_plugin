//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/SmallPtrSet.h>

namespace vpux {

//
// AliasesInfoBase
//

class AliasesInfoBase {
public:
    using ValuesSet = llvm::SmallPtrSet<mlir::Value, 16>;
    using ValuesMap = DenseMap<mlir::Value, ValuesSet>;

public:
    explicit AliasesInfoBase(Logger log): _log(log) {
    }
    explicit AliasesInfoBase(Logger log, std::optional<VPU::MemoryKind> memKind): _log(log), _memKind(memKind) {
    }
    AliasesInfoBase(const AliasesInfoBase&) = default;
    AliasesInfoBase& operator=(const AliasesInfoBase&) = default;

    // Returns the sources of a value.
    // Will return an empty set if `val` is a root value.
    const ValuesSet& getSources(mlir::Value val) const;

    // Returns the source of a value. The value is expected to have only one source, otherwise will throw an error.
    // Will return NULL if `val` is a root value.
    mlir::Value getSource(mlir::Value val) const;

    // Returns the roots of a value.
    // The set will contain `val` if it is a root value.
    const ValuesSet& getRoots(mlir::Value val) const;

    virtual ~AliasesInfoBase() = default;

protected:
    void visitOp(mlir::Operation* op, bool ignoreInnerRegions /* = false */);
    virtual void addAlias(mlir::Value source, mlir::Value alias) = 0;
    void addFuncArgAlias(mlir::Value alias);

protected:
    ValuesMap _sources;  // closest source of the alias
    ValuesMap _roots;    // top-root of the alias

    Logger _log;
    std::optional<VPU::MemoryKind> _memKind;
};

//
// ValueSourceInfo
//

class ValueSourceInfo final : public AliasesInfoBase {
public:
    explicit ValueSourceInfo(mlir::Value val);

    void addAlias(mlir::Value source, mlir::Value alias) override;

private:
    void updateRoots(mlir::Value val);
};

//
// AliasesInfo
//

class AliasesInfo : public AliasesInfoBase {
public:
    explicit AliasesInfo(mlir::func::FuncOp func);
    explicit AliasesInfo(mlir::func::FuncOp func, VPU::MemoryKind memKind);

    // The `val` must be a root value.
    const ValuesSet& getAllAliases(mlir::Value val) const;
    void addAlias(mlir::Value source, mlir::Value alias) override;
    // Remove information for given alias only
    void removeAlias(mlir::Value alias);
    // Remove all information related to given value (aliases, sources, roots)
    void remove(mlir::Value val);

private:
    void init(mlir::func::FuncOp func);

    ValuesMap _allAliases;  // all aliases, direct and indirect
};

//
// AliasesInfoMemType
//
template <VPU::MemoryKind memKind>
class AliasesInfoMemType : public AliasesInfo {
public:
    explicit AliasesInfoMemType(mlir::func::FuncOp func): AliasesInfo(func, memKind) {
    }
};

}  // namespace vpux
