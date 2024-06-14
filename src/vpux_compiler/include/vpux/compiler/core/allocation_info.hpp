//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_scheduler_utils.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"

#include "vpux/compiler/utils/partitioner.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

namespace vpux {

std::tuple<LinearScanHandler, std::list<ScheduledOpOneResource>> runLinearScan(
        mlir::func::FuncOp funcOp, MemLiveRangeInfo& liveRangeInfo, const AsyncDepsInfo& depsInfo,
        VPU::MemoryKind memKind, Logger log, ArrayRef<std::pair<vpux::AddressType, vpux::AddressType>> vec = {});

//
// AllocationInfo
// This is a helper class for working with the result of runLinearScan method through MLIR's analysis mechanism
//

struct ScanResult {
    LinearScanHandler& linearScanHandler;
    std::list<ScheduledOpOneResource>& scheduledOpOneResource;
    SmallVector<std::pair<AddressType, AddressType>> moduleReservedMemVec;
};

class AllocationInfo {
public:
    AllocationInfo(mlir::func::FuncOp netFunc, mlir::AnalysisManager& am);
    AllocationInfo(mlir::func::FuncOp netFunc, const AsyncDepsInfo& depsInfo, MemLiveRangeInfo& liveRangeInfo);

    bool hasResult(VPU::MemoryKind memKind);
    ScanResult getScanResult(VPU::MemoryKind memKind);

private:
    Logger _log;
    mlir::StringRef _mainFuncName;

    LinearScanHandler _linearScanHandler;
    std::list<ScheduledOpOneResource> _scheduledOpOneResource;
    SmallVector<std::pair<AddressType, AddressType>> _moduleReservedMemVec;
};

}  // namespace vpux
