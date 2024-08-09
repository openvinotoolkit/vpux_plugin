//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/reserved_memory_info.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/DenseSet.h>

#include <tuple>

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

    // E#122828: current work model is that repeating calls to the same function
    // use *the same* input/output buffers. thus, it may well be that the callee
    // map is going to be populated with the same data multiple times: avoid
    // this by caching the added buffers.
    using BufferPerCalleeAndMemory = std::tuple<mlir::StringRef, VPU::MemoryKind, mlir::Value>;
    auto insertThisBuffer = [cache = mlir::DenseSet<BufferPerCalleeAndMemory>()](
                                    mlir::StringRef calleeName, VPU::MemoryKind memKind, mlir::Value buffer) mutable {
        const bool firstInsertion = cache.insert(std::make_tuple(calleeName, memKind, buffer)).second;
        return firstInsertion;
    };

    auto updateReservedMemInfo = [&](StringRef calleeName, const ValueOrderedSet& buffers) {
        auto log = Logger::global().nest("reserved-memory-info");
        for (const auto& buffer : buffers) {
            if (!mlir::isa<mlir::BlockArgument>(buffer) && insertThisBuffer(calleeName, VPU::MemoryKind::DDR, buffer)) {
                auto addrAndSize =
                        std::make_pair(allReservedMemInfo.getAddress(buffer), allReservedMemInfo.getSize(buffer));
                log.trace("New reserved buffer {0} for {1} with memKind {2}: {3}", buffer, calleeName,
                          VPU::MemoryKind::DDR, addrAndSize);
                _allReservedMemInfo[calleeName][VPU::MemoryKind::DDR].push_back(std::move(addrAndSize));
            }
        }
    };

    netFunc.walk([&](mlir::func::CallOp callOp) {
        auto parentExecOp = callOp->getParentOfType<mlir::async::ExecuteOp>();
        VPUX_THROW_UNLESS(parentExecOp != nullptr, "func::CallOp must have async::ExecuteOp parent");

        auto calleeName = callOp.getCallee();
        updateReservedMemInfo(calleeName, liveRangeInfo.getInputBuffers(parentExecOp));
        updateReservedMemInfo(calleeName, liveRangeInfo.getOutputBuffers(parentExecOp));
    });

    // Add also reserved ranges on module level. Those reserved resource might be related
    // to handling of additional special features (e.g. DMA HW profiling)
    for (auto& calleeAndMemKindMap : _allReservedMemInfo) {
        auto& reservedAddressAndSizeVec = calleeAndMemKindMap.second[VPU::MemoryKind::DDR];
        reservedAddressAndSizeVec.insert(reservedAddressAndSizeVec.end(), moduleReservedMemVec.begin(),
                                         moduleReservedMemVec.end());
    }
}

// returns reserved addresses and sizes for func
ReservedMemInfo::MemReservedMap& ReservedMemInfo::getReservedMemInfo(mlir::StringRef funcName) {
    return _allReservedMemInfo[funcName];
}
