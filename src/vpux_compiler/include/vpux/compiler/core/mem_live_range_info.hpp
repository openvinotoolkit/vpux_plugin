//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/AnalysisManager.h>

namespace vpux {

class MemLiveRangeInfo {
    using BufferToOpsMap = ValueOrderedMap<OpOrderedSet>;
    using OpToUsedBuffersMap = OpOrderedMap<ValueOrderedSet>;

public:
    MemLiveRangeInfo(mlir::func::FuncOp funcOp, mlir::AnalysisManager& am);
    MemLiveRangeInfo(mlir::func::FuncOp funcOp, const AliasesInfo& aliasInfo);
    MemLiveRangeInfo(mlir::func::FuncOp funcOp, const AliasesInfo& aliasInfo, std::optional<VPU::MemoryKind> memKind);

public:
    ValueOrderedSet getUsedBuffers(mlir::Operation* op) const;
    ValueOrderedSet getInputBuffers(mlir::Operation* op);
    ValueOrderedSet getOutputBuffers(mlir::Operation* op);
    size_t eraseUser(mlir::Value val, mlir::Operation* op);
    bool isBufferUsedByOp(mlir::Value val, mlir::Operation* op) const;

private:
    void addNewBuffer(mlir::Value val);
    void buildRangeInfo(mlir::func::FuncOp funcOp);

private:
    Logger _log;
    const AliasesInfo& _aliasInfo;
    BufferToOpsMap _allUsersInBlock;
    OpToUsedBuffersMap _opBuffersMap;
    OpToUsedBuffersMap _opInputBuffersMap;
    OpToUsedBuffersMap _opOutputBuffersMap;
    std::optional<VPU::MemoryKind> _memKind;
};

//
// MemLiveRangeInfoMemType
//
template <VPU::MemoryKind memKind>
class MemLiveRangeInfoMemType : public MemLiveRangeInfo {
public:
    explicit MemLiveRangeInfoMemType(mlir::func::FuncOp func, mlir::AnalysisManager& am)
            : MemLiveRangeInfo(func, am.getAnalysis<AliasesInfoMemType<memKind>, mlir::func::FuncOp>(), memKind) {
    }

    explicit MemLiveRangeInfoMemType(mlir::func::FuncOp func, const AliasesInfoMemType<memKind>& aliasInfoMemType)
            : MemLiveRangeInfo(func, aliasInfoMemType, memKind) {
    }
};

}  // namespace vpux
