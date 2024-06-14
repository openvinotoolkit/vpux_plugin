//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vector>
#include <vpux_headers/platform.hpp>
#include "vpux/compiler/act_kernels/nce2p7.h"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace llvm {
using RelocKey = std::pair<mlir::Value, ELFNPU37XX::CreateSymbolTableSectionOp>;
template <>
struct DenseMapInfo<RelocKey> {
    static RelocKey getEmptyKey() {
        void* pointer = llvm::DenseMapInfo<void*>::getEmptyKey();
        return RelocKey(RelocKey::first_type::getFromOpaquePointer(pointer),
                        RelocKey::second_type::getFromOpaquePointer(pointer));
    }

    static RelocKey getTombstoneKey() {
        void* pointer = llvm::DenseMapInfo<void*>::getTombstoneKey();
        return RelocKey(RelocKey::first_type::getFromOpaquePointer(pointer),
                        RelocKey::second_type::getFromOpaquePointer(pointer));
    }

    static unsigned getHashValue(RelocKey val) {
        auto h1 = hash_value(val.first.getAsOpaquePointer());
        auto h2 = hash_value(val.second.getAsOpaquePointer());

        return static_cast<unsigned>(h1 * h2);
    }

    static bool isEqual(RelocKey lhs, RelocKey rhs) {
        auto l1 = DenseMapInfo<mlir::Value>::isEqual(lhs.first, rhs.first);
        auto l2 = DenseMapInfo<mlir::Operation*>::isEqual(lhs.second.getOperation(), rhs.second.getOperation());

        return l1 && l2;
    }
};
}  // namespace llvm

namespace vpux {

// creates a linear (1D) MemrefType of dimension (memrefSize x dataType)
mlir::MemRefType getLinearMemrefType(mlir::MLIRContext* ctx, int64_t memrefSize, mlir::Type dataType,
                                     VPU::MemoryKind memKind);

namespace ELFNPU37XX {

std::pair<const uint8_t*, size_t> getDataAndSizeOfElfSection(llvm::ArrayRef<uint8_t> elfBlob,
                                                             const std::vector<std::string> possibleSecNames);

using OffsetCache = mlir::DenseMap<mlir::Value, mlir::DenseMap<mlir::Value, size_t>>;
size_t getOffsetOfOpInSection(mlir::Value op, mlir::Value section, OffsetCache& cache);
size_t getOffsetOfOpInSection(mlir::Value& op);

SmallString getSwKernelArchString(VPU::ArchKind archKind);

class RelocationManager {
public:
    RelocationManager() = default;

    RelocationManager(mlir::func::FuncOp funcOp): funcOp_(funcOp) {
    }

    void init(mlir::func::FuncOp funcOp);
    void initCMXSymTab(ELFNPU37XX::CreateSymbolTableSectionOp cmxMappingSymTab);
    ELFNPU37XX::ElfSectionInterface getSection(mlir::Value value);
    ELFNPU37XX::CreateSymbolTableSectionOp getCMXSymTab();
    ELFNPU37XX::CreateSymbolTableSectionOp getSymTab(mlir::Value value);
    ELFNPU37XX::CreateRelocationSectionOp getRelocSection(ELFNPU37XX::ElfSectionInterface section,
                                                          ELFNPU37XX::CreateSymbolTableSectionOp symTab);
    static ELFNPU37XX::SymbolOp getSymbol(ELFNPU37XX::ElfSectionInterface section);

private:
    mlir::func::FuncOp funcOp_ = nullptr;
    ELFNPU37XX::CreateSymbolTableSectionOp cmxMappingSymTab_ = nullptr;
    llvm::DenseMap<mlir::Value, ELFNPU37XX::ElfSectionInterface> sectionMap_;
    llvm::DenseMap<mlir::Value, ELFNPU37XX::CreateSymbolTableSectionOp> symTabMap_;
    llvm::DenseMap<std::pair<mlir::Value, ELFNPU37XX::CreateSymbolTableSectionOp>,
                   ELFNPU37XX::CreateRelocationSectionOp>
            relocMap_;
};

namespace math {

size_t gcd(size_t a, size_t b);
size_t lcm(size_t a, size_t b);

}  // namespace math

constexpr size_t VPUX_SHAVE_ALIGNMENT = Byte(1_KB).count();
constexpr size_t VPUX_DEFAULT_ALIGNMENT = (64_Byte).count();
constexpr size_t VPUX_NO_ALIGNMENT = (1_Byte).count();

//
// Platform Information
//

elf::platform::ArchKind mapVpuArchKindToElfArchKind(const VPU::ArchKind& archKind);

}  // namespace ELFNPU37XX
}  // namespace vpux
