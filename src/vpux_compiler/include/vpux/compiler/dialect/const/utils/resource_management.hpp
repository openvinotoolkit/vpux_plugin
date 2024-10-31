//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LogicalResult.h>

#include <mutex>
#include <utility>

namespace vpux::Const {

/** @brief A simple "ref-count"-like type.
 *
 * This object acts as a "reference" to some constant data (e.g.
 * dense_resource): upon creation, it adds a reference, upon destruction, it
 * drops the reference. Thus the lifetime of this object could be used to mark
 * the start and end of the constant data usage.
 *
 * @note The actual ref-counting and resource clean up are done by another
 * entity.
 */
class DataRef {
    mlir::MLIRContext* _ctx = nullptr;
    mlir::StringRef _dataKey = {};

public:
    /// @brief Creates a new object with the key being a reference to a
    /// particular constant data.
    DataRef(mlir::MLIRContext* ctx, mlir::StringRef dataKey);
    ~DataRef();

    DataRef() = default;
    DataRef(const DataRef&);
    DataRef& operator=(const DataRef&);
    DataRef(DataRef&&);
    DataRef& operator=(DataRef&&);

    friend void swap(DataRef& x, DataRef& y) {
        using std::swap;
        swap(x._ctx, y._ctx);
        swap(x._dataKey, y._dataKey);
    }
};

/** @brief A pseudo-interface that wraps blob manager from MLIR.
 */
class ConstDataManagerInterface : public mlir::DialectInterface::Base<ConstDataManagerInterface> {
    std::mutex _refMutex{};
    mlir::DenseMap<mlir::StringRef, int64_t> _refCounts{};

public:
    ConstDataManagerInterface(mlir::Dialect* dialect);

    /// @brief Increments internal ref-count for a particular key.
    void addRef(llvm::StringRef key);

    /// @brief Decrements internal ref-count for a particular key. When
    /// ref-count reaches zero, the constant data by that key is deallocated.
    void dropRef(llvm::StringRef key);
};

}  // namespace vpux::Const
