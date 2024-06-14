//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

//
// Hierarchy aware utils
//

bool isNceTile(mlir::SymbolRefAttr executor);

//
// MemoryResourceOp
//

static constexpr StringLiteral usedMemModuleName = "UsedMemory";

template <typename Enum, typename OutT = MemoryResourceOp>
using memory_resource_if = enable_t<OutT, std::is_enum<Enum>, vpux::details::HasStringifyEnum<Enum>>;

template <typename Enum>
memory_resource_if<Enum> addAvailableMemory(mlir::ModuleOp mainModule, Enum kind, Byte size) {
    return addAvailableMemory(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)),
                              size);
}

MemoryResourceOp addAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace, Byte size);

template <typename Enum, typename OutT = bool>
using bool_if = enable_t<OutT, std::is_enum<Enum>, vpux::details::HasStringifyEnum<Enum>>;

template <typename Enum>
bool_if<Enum> hasAvailableMemory(mlir::ModuleOp mainModule, Enum kind) {
    return hasAvailableMemory(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)));
}

bool hasAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

MemoryResourceOp getAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getAvailableMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getAvailableMemory(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)));
}

MemoryResourceOp setUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace, Byte size);

template <typename Enum>
memory_resource_if<Enum> setUsedMemory(mlir::ModuleOp mainModule, Enum kind, Byte size) {
    return setUsedMemory(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)), size);
}

// TODO E#105253: consider not using temporary modules to store data in functions
MemoryResourceOp setUsedMemory(mlir::func::FuncOp func, mlir::SymbolRefAttr memSpace, Byte size);

template <typename Enum>
memory_resource_if<Enum> setUsedMemory(mlir::func::FuncOp func, Enum kind, Byte size) {
    return setUsedMemory(func, mlir::SymbolRefAttr::get(func->getContext(), stringifyEnum(kind)), size);
}

MemoryResourceOp getUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getUsedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getUsedMemory(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)));
}

SmallVector<MemoryResourceOp> getUsedMemory(mlir::ModuleOp mainModule);
SmallVector<MemoryResourceOp> getUsedMemory(mlir::func::FuncOp func);

void eraseUsedMemory(mlir::func::FuncOp func);

//
// Reserved memory resource
//
static constexpr StringLiteral resMemModuleName = "ReservedMemory";

SmallVector<IE::MemoryResourceOp> getReservedMemoryResources(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

SmallVector<std::pair<uint64_t, uint64_t>> getReservedMemOffsetAndSizeVec(mlir::ModuleOp module,
                                                                          mlir::SymbolRefAttr memSpaceAttr);
//
// DMA profiling reserved memory
//
static constexpr StringLiteral dmaProfilingResMemModuleName = "DmaProfilingReservedMemory";

IE::MemoryResourceOp setDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                   int64_t size);

IE::MemoryResourceOp getDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getDmaProfilingReservedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getDmaProfilingReservedMemory(mainModule,
                                         mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

SmallVector<MemoryResourceOp> getDmaProfilingReservedMemory(mlir::ModuleOp mainModule);

//
// Compressed DMA reserved memory
//
static constexpr StringLiteral compressDmaResMemModuleName = "CompressDmaReservedMemory";

IE::MemoryResourceOp setCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                  int64_t size);

IE::MemoryResourceOp getCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getCompressDmaReservedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getCompressDmaReservedMemory(mainModule,
                                        mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

SmallVector<MemoryResourceOp> getCompressDmaReservedMemory(mlir::ModuleOp mainModule);

//
// SW Kernel prefetching reserved memory
//
static constexpr StringLiteral swKernelPrefetchingResMemModuleName = "SWKernelPrefetchingReservedMemory";

IE::MemoryResourceOp setSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                          int64_t size);

IE::MemoryResourceOp getSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

template <typename Enum>
memory_resource_if<Enum> getSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule, Enum kind) {
    return getSWKernelPrefetchingReservedMemory(mainModule,
                                                mlir::SymbolRefAttr::get(mainModule.getContext(), stringifyEnum(kind)));
}

SmallVector<MemoryResourceOp> getSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule);

//
// ExecutorResourceOp
//

namespace details {

ExecutorResourceOp addExecutor(mlir::Region& region, mlir::SymbolRefAttr executorAttr, size_t count);

bool hasExecutor(mlir::SymbolTable mainModule, mlir::SymbolRefAttr executorAttr);

}  // namespace details

template <typename Enum, typename OutT = ExecutorResourceOp>
using exec_resource_if = enable_t<OutT, std::is_enum<Enum>, vpux::details::HasStringifyEnum<Enum>>;

template <typename Enum>
exec_resource_if<Enum> addAvailableExecutor(mlir::ModuleOp mainModule, Enum kind, size_t count) {
    const auto executorAttr = mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind));
    return details::addExecutor(mainModule.getBodyRegion(), executorAttr, count);
}

template <typename Enum>
bool_if<Enum> hasExecutor(mlir::ModuleOp mainModule, Enum kind) {
    const auto executorAttr = mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind));
    return details::hasExecutor(mainModule.getOperation(), executorAttr);
}

ExecutorResourceOp getAvailableExecutor(mlir::ModuleOp mainModule, mlir::SymbolRefAttr executorAttr);

template <typename Enum>
exec_resource_if<Enum> getAvailableExecutor(mlir::ModuleOp mainModule, Enum kind) {
    return getAvailableExecutor(mainModule, mlir::SymbolRefAttr::get(mainModule->getContext(), stringifyEnum(kind)));
}

//
// ShaveResources
//

int64_t getTotalNumOfActShaveEngines(mlir::ModuleOp moduleOp);

//
// TileResourceOp
//

TileResourceOp addTileExecutor(mlir::ModuleOp mainModule, size_t count);

bool hasTileExecutor(mlir::ModuleOp mainModule);

TileResourceOp getTileExecutor(mlir::ModuleOp mainModule);

TileResourceOp getTileExecutor(mlir::func::FuncOp funcOp);

}  // namespace IE
}  // namespace vpux
