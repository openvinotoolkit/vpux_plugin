//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/allocation_info.hpp"

#include "vpux/compiler/utils/partitioner.hpp"

#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinOps.h>

namespace vpux {

//
// ReservedMemInfo
//

class ReservedMemInfo {
public:
    using ReservedAddressAndSizeVector = SmallVector<std::pair<vpux::AddressType, vpux::AddressType>>;
    using MemReservedMap = DenseMap<VPU::MemoryKind, ReservedAddressAndSizeVector>;
    using CalleeMap = DenseMap<mlir::StringRef, MemReservedMap>;

public:
    ReservedMemInfo(mlir::ModuleOp moduleOp, mlir::AnalysisManager& am);
    ReservedMemInfo(mlir::func::FuncOp netFunc);

    // returns reserved addresses and sizes for func
    MemReservedMap& getReservedMemInfo(mlir::StringRef funcName);

private:
    ReservedAddressAndSizeVector getUniqueRanges(const ValueOrderedSet& buffers, ScanResult& scanResult);
    void reserveFunctionRanges(mlir::func::FuncOp netFunc, MemLiveRangeInfo& liveRangeInfo, ScanResult& scanResult);
    void reserveModuleRanges(ScanResult& scanResult);
    void linearizeCallOps(mlir::func::FuncOp netFunc, AsyncDepsInfo& depsInfo);

private:
    Logger _log = Logger::global().nest("reserved-mem-info", 0);

    CalleeMap _allReservedMemInfo;
};

}  // namespace vpux
