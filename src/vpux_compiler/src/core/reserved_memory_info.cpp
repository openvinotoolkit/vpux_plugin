//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/reserved_memory_info.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

ReservedMemInfo::ReservedMemInfo(mlir::ModuleOp moduleOp, mlir::AnalysisManager& am) {
    // TODO:#108991 -- for now only "main" function is supported,
    // but it is possible support multiple nested calls using a loop through call/function ops
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp netInfo;
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

    init(netFunc, am.getChildAnalysis<AllocationInfo>(netFunc),
         am.getChildAnalysis<MemLiveRangeInfoMemType<VPU::MemoryKind::DDR>>(netFunc));
}

ReservedMemInfo::ReservedMemInfo(mlir::func::FuncOp netFunc, AllocationInfo& allocationInfo,
                                 MemLiveRangeInfo& liveRangeInfo) {
    init(netFunc, allocationInfo, liveRangeInfo);
}

void ReservedMemInfo::init(mlir::func::FuncOp netFunc, AllocationInfo& allocationInfo,
                           MemLiveRangeInfo& liveRangeInfo) {
    // Only DDR is supported by this time
    if (!allocationInfo.hasResult(VPU::MemoryKind::DDR)) {
        return;
    }

    auto scanResult = allocationInfo.getScanResult(VPU::MemoryKind::DDR);
    auto& allReservedMemInfo = scanResult.linearScanHandler;
    auto& moduleReservedMemVec = scanResult.moduleReservedMemVec;

    auto updateReservedMemInfo = [&](StringRef calleeName, const ValueOrderedSet& buffers) {
        for (const auto& buffer : buffers) {
            if (!mlir::isa<mlir::BlockArgument>(buffer)) {
                _allReservedMemInfo[calleeName][VPU::MemoryKind::DDR].push_back(
                        {allReservedMemInfo.getAddress(buffer), allReservedMemInfo.getSize(buffer)});
            }
        }
    };

    netFunc.walk([&](mlir::func::CallOp callOp) {
        auto parentExecOp = callOp->getParentOfType<mlir::async::ExecuteOp>();
        VPUX_THROW_UNLESS(parentExecOp != nullptr, "func::CallOp must have async::ExecuteOp parent");

        auto calleeName = callOp.getCallee();

        // Add also reserved ranges on module level. Those reserved resource might be related
        // to handling of additional special features (e.g. DMA HW profiling)
        _allReservedMemInfo[calleeName][VPU::MemoryKind::DDR].insert(
                _allReservedMemInfo[calleeName][VPU::MemoryKind::DDR].end(), moduleReservedMemVec.begin(),
                moduleReservedMemVec.end());

        updateReservedMemInfo(calleeName, liveRangeInfo.getInputBuffers(parentExecOp));
        updateReservedMemInfo(calleeName, liveRangeInfo.getOutputBuffers(parentExecOp));
    });
}

// returns reserved addresses and sizes for func
ReservedMemInfo::MemReservedMap& ReservedMemInfo::getReservedMemInfo(mlir::StringRef funcName) {
    return _allReservedMemInfo[funcName];
}
