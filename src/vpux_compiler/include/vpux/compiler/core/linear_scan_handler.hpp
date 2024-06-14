//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseSet.h>

namespace vpux {

//
// LinearScanHandler
//

class LinearScanHandler final {
public:
    explicit LinearScanHandler(AddressType defaultAlignment = 1);

public:
    void markAsDead(mlir::Value val);
    void markAllBuffersAsDead();
    void markAsAlive(mlir::Value val);
    void markAsDynamicSpill(mlir::Value val);
    void removeDynamicSpill(mlir::Value val);
    Byte maxAllocatedSize() const;

public:
    bool isAlive(mlir::Value val) const;
    bool isDynamicSpill(mlir::Value val) const;
    static bool isFixedAlloc(mlir::Value val);
    AddressType getSize(mlir::Value val);
    AddressType getAlignment(mlir::Value val) const;
    AddressType getAddress(mlir::Value val) const;
    void allocated(mlir::Value val, AddressType addr);
    void deallocate(mlir::Value val);
    mlir::DenseSet<mlir::Value> getAliveValues();
    void freed(mlir::Value val);
    static int getSpillWeight(mlir::Value);
    static bool spilled(mlir::Value);
    void setAddress(mlir::Value val, AddressType address);

private:
    DenseMap<mlir::Value, AddressType> _valOffsets;
    DenseMap<mlir::Value, AddressType> _sizeCache;
    llvm::DenseSet<mlir::Value> _aliveValues;
    llvm::DenseSet<mlir::Value> _dynamicSpillValues;
    AddressType _defaultAlignment = 1;
    Byte _maxAllocatedSize;
};

}  // namespace vpux
