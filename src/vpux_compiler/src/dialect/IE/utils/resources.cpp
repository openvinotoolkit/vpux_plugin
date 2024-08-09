//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

using namespace vpux;

//
// Declarations
//

namespace vpux::IE::details {

MemoryResourceOp addAvailableMemory(mlir::Region& region, mlir::SymbolRefAttr memSpace, Byte size);
bool hasAvailableMemory(mlir::SymbolTable symbolTable, mlir::SymbolRefAttr memSpace);
MemoryResourceOp getAvailableMemory(mlir::SymbolTable symbolTable, mlir::SymbolRefAttr memSpace);

template <class ConcreteOp>
MemoryResourceOp setUsedMemory(ConcreteOp op, mlir::SymbolRefAttr memSpace, Byte size);
MemoryResourceOp getUsedMemory(mlir::SymbolTable symbolTable, mlir::SymbolRefAttr memSpace);

mlir::ModuleOp getTmpModule(ArrayRef<mlir::ModuleOp> modules);

bool isNceTileMemory(mlir::SymbolRefAttr memSpace);
bool isNceTileExecutor(mlir::SymbolRefAttr executor);

mlir::SymbolTable getSymbolTableContainer(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);
mlir::Region* getRegionContainer(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace);

IE::MemoryResourceOp addReservedMemoryResource(mlir::ModuleOp mainModule, mlir::StringLiteral reservedMemorySection,
                                               mlir::SymbolRefAttr memSpace, int64_t size);
IE::MemoryResourceOp getReservedMemoryResource(mlir::ModuleOp mainModule, mlir::StringLiteral reservedMemorySection,
                                               mlir::SymbolRefAttr memSpace);
SmallVector<IE::MemoryResourceOp> getReservedMemoryResource(mlir::ModuleOp mainModule,
                                                            mlir::StringLiteral reservedMemorySection);

}  // namespace vpux::IE::details

//
// Hierarchy aware utils
//

bool vpux::IE::details::isNceTileMemory(mlir::SymbolRefAttr memSpace) {
    auto memSpaceStr = memSpace.getRootReference().getValue();
    return memSpaceStr == stringifyEnum(VPU::MemoryKind::CMX_NN) || memSpaceStr == VPU::CMX_NN_FragmentationAware;
}

bool vpux::IE::details::isNceTileExecutor(mlir::SymbolRefAttr executor) {
    auto nceExecutorList =
            SmallVector<StringRef>({stringifyEnum(VPU::ExecutorKind::DPU), stringifyEnum(VPU::ExecutorKind::SHAVE_ACT),
                                    stringifyEnum(VPU::ExecutorKind::SHAVE_NN)});
    auto executorStr = executor.getLeafReference().getValue();
    return std::find(nceExecutorList.begin(), nceExecutorList.end(), executorStr) != nceExecutorList.end();
}

bool vpux::IE::isNceTile(mlir::SymbolRefAttr executor) {
    return executor.getLeafReference().getValue() == stringifyEnum(VPU::ExecutorKind::NCE);
}

//
// MemoryResourceOp
//

IE::MemoryResourceOp vpux::IE::details::addAvailableMemory(mlir::Region& region, mlir::SymbolRefAttr memSpace,
                                                           Byte size) {
    VPUX_THROW_UNLESS(size.count() > 0, "Trying to set zero size of memory kind '{0}'", memSpace);
    const auto byteSizeAttr = getIntAttr(region.getContext(), size.count());
    auto builder = mlir::OpBuilder::atBlockBegin(&region.front());
    return builder.create<IE::MemoryResourceOp>(mlir::UnknownLoc::get(region.getContext()), memSpace.getLeafReference(),
                                                byteSizeAttr, nullptr);
}

IE::MemoryResourceOp vpux::IE::addAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace, Byte size) {
    if (details::isNceTileMemory(memSpace)) {
        auto tileOp = getTileExecutor(mainModule);
        VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to add '{0}' memspace.", memSpace);
        return tileOp.addAvailableMemory(memSpace, size);
    }
    return details::addAvailableMemory(mainModule.getBodyRegion(), memSpace, size);
}

bool vpux::IE::details::hasAvailableMemory(mlir::SymbolTable symbolTable, mlir::SymbolRefAttr memSpace) {
    auto res = symbolTable.lookup<IE::MemoryResourceOp>(memSpace.getLeafReference());
    return res != nullptr;
}

bool vpux::IE::hasAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    if (details::isNceTileMemory(memSpace)) {
        auto tileOp = getTileExecutor(mainModule);
        VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query '{0}' memspace.", memSpace);
        return tileOp.hasAvailableMemory(memSpace);
    }
    return details::hasAvailableMemory(mainModule.getOperation(), memSpace);
}

IE::MemoryResourceOp vpux::IE::details::getAvailableMemory(mlir::SymbolTable symbolTable,
                                                           mlir::SymbolRefAttr memSpace) {
    return symbolTable.lookup<IE::MemoryResourceOp>(memSpace.getLeafReference());
}

IE::MemoryResourceOp vpux::IE::getAvailableMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    if (details::isNceTileMemory(memSpace)) {
        auto tileOp = getTileExecutor(mainModule);
        VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query '{0}' memspace.", memSpace);
        return tileOp.getAvailableMemory(memSpace);
    }
    return details::getAvailableMemory(mainModule.getOperation(), memSpace);
}

//
// Used memory resources
//

template <class ConcreteOp>
IE::MemoryResourceOp vpux::IE::details::setUsedMemory(ConcreteOp op, mlir::SymbolRefAttr memSpace, Byte size) {
    auto available = details::getAvailableMemory(op.getOperation(), memSpace);
    VPUX_THROW_UNLESS(available != nullptr, "Memory kind '{0}' is not registered as available", memSpace);
    VPUX_THROW_UNLESS(size <= available.size(), "Memory kind '{0}' used size '{1}' exceeds available size '{2}'",
                      memSpace, size, available.size());

    auto byteSizeAttr = getIntAttr(op->getContext(), size.count());

    auto usedMemModule = op.template lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        auto mainBuilder = mlir::OpBuilder::atBlockBegin(op.getBody());
        usedMemModule = mainBuilder.template create<mlir::ModuleOp>(mlir::UnknownLoc::get(mainBuilder.getContext()),
                                                                    usedMemModuleName);
    }

    auto res = usedMemModule.template lookupSymbol<IE::MemoryResourceOp>(memSpace);
    if (res != nullptr) {
        res.setByteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(usedMemModule.getBody());
    return innerBuilder.template create<IE::MemoryResourceOp>(mlir::UnknownLoc::get(usedMemModule.getContext()),
                                                              memSpace.getLeafReference(), byteSizeAttr, nullptr);
}

IE::MemoryResourceOp vpux::IE::setUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace, Byte size) {
    if (details::isNceTileMemory(memSpace)) {
        auto tileOp = getTileExecutor(mainModule);
        VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to set '{0}' memspace.", memSpace);
        return tileOp.setUsedMemory(memSpace, size);
    }

    return details::setUsedMemory(mainModule, memSpace, size);
}

mlir::ModuleOp vpux::IE::details::getTmpModule(ArrayRef<mlir::ModuleOp> modules) {
    VPUX_THROW_WHEN(modules.size() != 1, "Expected exactly one Temporary module. Got {0}", modules.size());
    auto tmpModule = modules[0];

    auto maybeSymName = tmpModule.getSymName();
    VPUX_THROW_UNLESS(maybeSymName.has_value(), "Temporary module must have sym name: {0}", usedMemModuleName);
    VPUX_THROW_UNLESS(maybeSymName.value() == usedMemModuleName, "Temporary module must have sym name: {0}, Got: {1}",
                      usedMemModuleName, maybeSymName.value());

    return tmpModule;
}

IE::MemoryResourceOp vpux::IE::setUsedMemory(mlir::func::FuncOp func, mlir::SymbolRefAttr memSpace, Byte size) {
    auto available = getAvailableMemory(func->getParentOfType<mlir::ModuleOp>(), memSpace);
    VPUX_THROW_UNLESS(available != nullptr, "Memory kind '{0}' is not registered as available", memSpace);
    VPUX_THROW_UNLESS(size <= available.size(), "Memory kind '{0}' used size '{1}' exceeds available size '{2}'",
                      memSpace, size, available.size());

    auto modules = to_small_vector(func.getOps<mlir::ModuleOp>());

    mlir::ModuleOp tmpModule;
    if (modules.empty()) {
        auto mainBuilder = mlir::OpBuilder::atBlockBegin(&func.getBody().front());
        tmpModule = mainBuilder.template create<mlir::ModuleOp>(mlir::UnknownLoc::get(mainBuilder.getContext()),
                                                                usedMemModuleName);
    } else {
        tmpModule = details::getTmpModule(modules);
    }

    auto res = tmpModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
    auto byteSizeAttr = getIntAttr(func->getContext(), size.count());
    if (res != nullptr) {
        res.setByteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(tmpModule.getBody());
    return innerBuilder.template create<IE::MemoryResourceOp>(mlir::UnknownLoc::get(tmpModule.getContext()),
                                                              memSpace.getLeafReference(), byteSizeAttr, nullptr);
}

IE::MemoryResourceOp vpux::IE::details::getUsedMemory(mlir::SymbolTable symbolTable, mlir::SymbolRefAttr memSpace) {
    auto usedMemModule = symbolTable.lookup<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        return nullptr;
    }

    return usedMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
}

IE::MemoryResourceOp vpux::IE::getUsedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    return details::getUsedMemory(mainModule.getOperation(), memSpace);
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getUsedMemory(mlir::ModuleOp mainModule) {
    auto usedMemModule = mainModule.lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (usedMemModule == nullptr) {
        return {};
    }
    auto usedMem = to_small_vector(usedMemModule.getOps<IE::MemoryResourceOp>());
    auto tileOp = getTileExecutor(mainModule);
    if (tileOp == nullptr) {
        return usedMem;
    }
    auto nceUsedMemModule = tileOp.lookupSymbol<mlir::ModuleOp>(usedMemModuleName);
    if (nceUsedMemModule == nullptr) {
        return usedMem;
    }
    auto nceUsedMem = to_small_vector(nceUsedMemModule.getOps<IE::MemoryResourceOp>());
    usedMem.append(nceUsedMem);

    return usedMem;
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getUsedMemory(mlir::func::FuncOp func) {
    auto modules = to_small_vector(func.getOps<mlir::ModuleOp>());

    if (modules.empty()) {
        return {};
    }

    auto tmpModule = details::getTmpModule(modules);
    return to_small_vector(tmpModule.getOps<IE::MemoryResourceOp>());
}

void vpux::IE::eraseUsedMemory(mlir::func::FuncOp func) {
    auto modules = to_small_vector(func.getOps<mlir::ModuleOp>());

    if (modules.empty()) {
        return;
    }

    auto tmpModule = details::getTmpModule(modules);
    tmpModule.erase();
}

//
// Reserved memory resources
//

mlir::SymbolTable vpux::IE::details::getSymbolTableContainer(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    if (isNceTileMemory(memSpace)) {
        auto tileOp = getTileExecutor(mainModule);
        VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query '{0}' memspace.", memSpace);
        return tileOp.getOperation();
    }

    return mainModule.getOperation();
}

mlir::Region* vpux::IE::details::getRegionContainer(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    if (isNceTileMemory(memSpace)) {
        auto tileOp = getTileExecutor(mainModule);
        VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query '{0}' memspace.", memSpace);
        return &tileOp.getRegion();
    }
    return &mainModule.getBodyRegion();
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getReservedMemoryResources(mlir::ModuleOp mainModule,
                                                                       mlir::SymbolRefAttr memSpace) {
    auto symbolTable = details::getSymbolTableContainer(mainModule, memSpace);
    SmallVector<IE::MemoryResourceOp> resMemVec;

    auto resMemModule = symbolTable.lookup<mlir::ModuleOp>(resMemModuleName);
    if (resMemModule == nullptr) {
        return {};
    }

    for (auto&& resMemModuleOp : resMemModule.getOps<mlir::ModuleOp>()) {
        resMemVec.push_back(resMemModuleOp.lookupSymbol<IE::MemoryResourceOp>(memSpace));
    }

    return resMemVec;
}

IE::MemoryResourceOp vpux::IE::details::addReservedMemoryResource(mlir::ModuleOp mainModule,
                                                                  mlir::StringLiteral reservedMemorySection,
                                                                  mlir::SymbolRefAttr memSpace, int64_t size) {
    auto region = getRegionContainer(mainModule, memSpace);
    auto symbolTable = getSymbolTableContainer(mainModule, memSpace);
    auto resMemTable = symbolTable.lookup<mlir::ModuleOp>(IE::resMemModuleName);
    if (resMemTable == nullptr) {
        auto mainBuilder = mlir::OpBuilder::atBlockBegin(&region->front());
        resMemTable = mainBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(mainBuilder.getContext()),
                                                         IE::resMemModuleName);
    }

    auto resMemBuilder = mlir::OpBuilder::atBlockBegin(resMemTable.getBody());

    auto resMemModule = resMemTable.lookupSymbol<mlir::ModuleOp>(reservedMemorySection);
    if (resMemModule == nullptr) {
        resMemModule = resMemBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(resMemBuilder.getContext()),
                                                            reservedMemorySection);
    }

    auto* ctx = symbolTable.getOp()->getContext();
    auto byteSizeAttr = getIntAttr(ctx, size);

    auto res = resMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
    if (res != nullptr) {
        res.setByteSizeAttr(byteSizeAttr);
        return res;
    }

    auto innerBuilder = mlir::OpBuilder::atBlockBegin(resMemModule.getBody());
    return innerBuilder.create<IE::MemoryResourceOp>(mlir::UnknownLoc::get(resMemModule.getContext()),
                                                     memSpace.getLeafReference(), byteSizeAttr, nullptr);
};

IE::MemoryResourceOp vpux::IE::details::getReservedMemoryResource(mlir::ModuleOp mainModule,
                                                                  mlir::StringLiteral reservedMemorySection,
                                                                  mlir::SymbolRefAttr memSpace) {
    mlir::SymbolTable symbolTable = getSymbolTableContainer(mainModule, memSpace);
    auto resMemTable = symbolTable.lookup<mlir::ModuleOp>(IE::resMemModuleName);
    if (resMemTable == nullptr) {
        return nullptr;
    }

    auto resMemModule = resMemTable.lookupSymbol<mlir::ModuleOp>(reservedMemorySection);
    if (resMemModule == nullptr) {
        return nullptr;
    }

    return resMemModule.lookupSymbol<IE::MemoryResourceOp>(memSpace);
}

SmallVector<IE::MemoryResourceOp> vpux::IE::details::getReservedMemoryResource(
        mlir::ModuleOp mainModule, mlir::StringLiteral reservedMemorySection) {
    auto resMemTable = mainModule.lookupSymbol<mlir::ModuleOp>(IE::resMemModuleName);
    if (resMemTable == nullptr) {
        return {};
    }
    auto resMemModule = resMemTable.lookupSymbol<mlir::ModuleOp>(reservedMemorySection);
    if (resMemModule == nullptr) {
        return {};
    }
    auto reservedMem = to_small_vector(resMemModule.getOps<IE::MemoryResourceOp>());

    auto tileOp = IE::getTileExecutor(mainModule);
    if (tileOp == nullptr) {
        return reservedMem;
    }
    auto nceResMemTable = tileOp.lookupSymbol<mlir::ModuleOp>(IE::resMemModuleName);
    if (nceResMemTable == nullptr) {
        return {};
    }
    auto nceResMemModule = nceResMemTable.lookupSymbol<mlir::ModuleOp>(reservedMemorySection);
    if (nceResMemModule == nullptr) {
        return {};
    }

    auto nceReservedMem = to_small_vector(nceResMemModule.getOps<IE::MemoryResourceOp>());
    reservedMem.append(nceReservedMem);

    return reservedMem;
}

// Get information about reserved resources in given memory type
// This function should be called before performing memory allocation
SmallVector<std::pair<uint64_t, uint64_t>> vpux::IE::getReservedMemOffsetAndSizeVec(mlir::ModuleOp module,
                                                                                    mlir::SymbolRefAttr memSpaceAttr) {
    SmallVector<std::pair<uint64_t, uint64_t>> reservedMemVec;
    // Check for reserved memory which memory scheduler should take into account
    // so that they not overlap with other buffers. Those reserved resource might be related
    // to handling of additional special features (e.g. DMA HW profiling)
    auto reservedMemoryResources = IE::getReservedMemoryResources(module, memSpaceAttr);
    if (!reservedMemoryResources.empty()) {
        // Put all reserved resources starting from 0 if they were not assigned any address
        size_t resMemOffset = 0;
        for (auto& resMem : reservedMemoryResources) {
            auto resMemSize = resMem.getByteSize();
            resMemOffset = resMem.getOffset().value_or(resMemOffset);
            reservedMemVec.push_back(std::make_pair(resMemOffset, resMemSize));
            if (!resMem.getOffset().has_value()) {
                resMem.setOffsetAttr(getIntAttr(module->getContext(), resMemOffset));
            }
            resMemOffset += resMemSize;
        }
    }

    return reservedMemVec;
}

//
// DMA profiling reserved memory
//

IE::MemoryResourceOp vpux::IE::setDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                             int64_t size) {
    return details::addReservedMemoryResource(mainModule, dmaProfilingResMemModuleName, memSpace, size);
}

IE::MemoryResourceOp vpux::IE::getDmaProfilingReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    return details::getReservedMemoryResource(mainModule, dmaProfilingResMemModuleName, memSpace);
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getDmaProfilingReservedMemory(mlir::ModuleOp mainModule) {
    return details::getReservedMemoryResource(mainModule, dmaProfilingResMemModuleName);
}

//
// Compressed DMA reserved memory
//

IE::MemoryResourceOp vpux::IE::setCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace,
                                                            int64_t size) {
    return details::addReservedMemoryResource(mainModule, compressDmaResMemModuleName, memSpace, size);
}

IE::MemoryResourceOp vpux::IE::getCompressDmaReservedMemory(mlir::ModuleOp mainModule, mlir::SymbolRefAttr memSpace) {
    return details::getReservedMemoryResource(mainModule, compressDmaResMemModuleName, memSpace);
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getCompressDmaReservedMemory(mlir::ModuleOp mainModule) {
    return details::getReservedMemoryResource(mainModule, compressDmaResMemModuleName);
}

//
// SW Kernel prefetching reserved memory
//

IE::MemoryResourceOp vpux::IE::setSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule,
                                                                    mlir::SymbolRefAttr memSpace, int64_t size) {
    return details::addReservedMemoryResource(mainModule, swKernelPrefetchingResMemModuleName, memSpace, size);
}

IE::MemoryResourceOp vpux::IE::getSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule,
                                                                    mlir::SymbolRefAttr memSpace) {
    return details::getReservedMemoryResource(mainModule, swKernelPrefetchingResMemModuleName, memSpace);
}

SmallVector<IE::MemoryResourceOp> vpux::IE::getSWKernelPrefetchingReservedMemory(mlir::ModuleOp mainModule) {
    return details::getReservedMemoryResource(mainModule, swKernelPrefetchingResMemModuleName);
}

//
// ExecutorResourceOp
//

IE::ExecutorResourceOp vpux::IE::details::addExecutor(mlir::Region& region, mlir::SymbolRefAttr executorAttr,
                                                      size_t count) {
    VPUX_THROW_UNLESS(count > 0, "Trying to set zero count of executor kind '{0}'", executorAttr);
    VPUX_THROW_UNLESS(!IE::isNceTile(executorAttr), "Unexpected '{0}' during executor query.", executorAttr);
    auto* ctx = region.getContext();
    const auto countAttr = getIntAttr(ctx, count);
    auto builder = mlir::OpBuilder::atBlockBegin(&region.front());
    return builder.create<IE::ExecutorResourceOp>(mlir::UnknownLoc::get(ctx), executorAttr.getLeafReference(),
                                                  countAttr, nullptr);
}

bool vpux::IE::details::hasExecutor(mlir::SymbolTable symbolTable, mlir::SymbolRefAttr executorAttr) {
    VPUX_THROW_UNLESS(!IE::isNceTile(executorAttr), "Unexpected '{0}' during executor query.", executorAttr);
    auto res = symbolTable.lookup<IE::ExecutorResourceOp>(executorAttr.getLeafReference());
    return res != nullptr;
}

IE::ExecutorResourceOp vpux::IE::getAvailableExecutor(mlir::ModuleOp mainModule, mlir::SymbolRefAttr executorAttr) {
    VPUX_THROW_UNLESS(!IE::isNceTile(executorAttr), "Unexpected '{0}' during executor query.", executorAttr);
    if (details::isNceTileExecutor(executorAttr)) {
        auto tileOp = getTileExecutor(mainModule);
        VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query '{0}' executor.",
                          executorAttr);
        return tileOp.getSubExecutor(executorAttr);
    }
    return mainModule.lookupSymbol<IE::ExecutorResourceOp>(executorAttr);
}

//
// TileResourceOp
//

IE::ExecutorResourceOp vpux::IE::TileResourceOp::addSubExecutor(mlir::SymbolRefAttr executorAttr, size_t count) {
    return details::addExecutor(getRegion(), executorAttr, count);
}

bool vpux::IE::TileResourceOp::hasSubExecutor(mlir::SymbolRefAttr executorAttr) {
    return details::hasExecutor(getOperation(), executorAttr);
}

IE::ExecutorResourceOp vpux::IE::TileResourceOp::getSubExecutor(mlir::SymbolRefAttr executorAttr) {
    return lookupSymbol<IE::ExecutorResourceOp>(executorAttr.getLeafReference());
}

IE::MemoryResourceOp vpux::IE::TileResourceOp::addAvailableMemory(mlir::SymbolRefAttr memSpace, Byte size) {
    return details::addAvailableMemory(getRegion(), memSpace, size);
}

bool vpux::IE::TileResourceOp::hasAvailableMemory(mlir::SymbolRefAttr memSpace) {
    return details::hasAvailableMemory(getOperation(), memSpace);
}

IE::MemoryResourceOp vpux::IE::TileResourceOp::getAvailableMemory(mlir::SymbolRefAttr memSpace) {
    return lookupSymbol<IE::MemoryResourceOp>(memSpace);
}

IE::MemoryResourceOp vpux::IE::TileResourceOp::setUsedMemory(mlir::SymbolRefAttr memSpace, Byte size) {
    return details::setUsedMemory(*this, memSpace, size);
}

IE::MemoryResourceOp vpux::IE::TileResourceOp::getUsedMemory(mlir::SymbolRefAttr memSpace) {
    return details::getUsedMemory(getOperation(), memSpace);
}

//
// ShaveResources
//

int64_t vpux::IE::getTotalNumOfActShaveEngines(mlir::ModuleOp mainModule) {
    auto tileOp = getTileExecutor(mainModule);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query SHAVE_ACT executor.");
    VPUX_THROW_UNLESS(tileOp.hasSubExecutor(VPU::ExecutorKind::SHAVE_ACT),
                      "No SHAVE_ACT executor found, check your arch");
    auto actShavePerTile = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT);
    return tileOp.getCount() * actShavePerTile.getCount();
}

IE::TileResourceOp IE::addTileExecutor(mlir::ModuleOp mainModule, size_t count) {
    VPUX_THROW_UNLESS(count > 0, "Trying to set zero count of tile resource kind.");

    auto* ctx = mainModule.getContext();
    auto countAttr = getIntAttr(ctx, count);
    auto builder = mlir::OpBuilder::atBlockBegin(mainModule.getBody());
    auto nameAttr = mlir::StringAttr::get(ctx, stringifyEnum(VPU::ExecutorKind::NCE));
    auto resOp = builder.create<IE::TileResourceOp>(mlir::UnknownLoc::get(ctx), nameAttr, countAttr, nullptr, nullptr);

    // Operations with a 'SymbolTable' must have exactly one block
    resOp.getRegion().emplaceBlock();
    return resOp;
}

bool IE::hasTileExecutor(mlir::ModuleOp mainModule) {
    auto res = mainModule.lookupSymbol<IE::TileResourceOp>(stringifyEnum(VPU::ExecutorKind::NCE));
    return res != nullptr;
}

IE::TileResourceOp IE::getTileExecutor(mlir::ModuleOp mainModule) {
    return mainModule.lookupSymbol<IE::TileResourceOp>(stringifyEnum(VPU::ExecutorKind::NCE));
}

IE::TileResourceOp IE::getTileExecutor(mlir::func::FuncOp funcOp) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    return moduleOp.lookupSymbol<IE::TileResourceOp>(stringifyEnum(VPU::ExecutorKind::NCE));
}
