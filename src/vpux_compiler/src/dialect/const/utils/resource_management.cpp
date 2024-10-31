//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/resource_management.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"  // for ConstDialect
#include "vpux/utils/core/error.hpp"

#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectResourceBlobManager.h>

#include <cassert>

namespace vpux::Const {
namespace {
ConstDataManagerInterface& fetchDataManager(mlir::MLIRContext* ctx) {
    auto* dialect = ctx->getOrLoadDialect<vpux::Const::ConstDialect>();
    assert(dialect != nullptr && "ConstDialect must be present in the context");

    auto* iface = dialect->getRegisteredInterface<ConstDataManagerInterface>();
    assert(iface != nullptr && "ConstDataManagerInterface must be registered in the context");
    return *iface;
}
}  // namespace

DataRef::DataRef(mlir::MLIRContext* ctx, mlir::StringRef dataKey): _ctx(ctx), _dataKey(dataKey) {
    static_assert(std::is_same_v<decltype(std::declval<mlir::DenseResourceElementsAttr>().getRawHandle().getKey()),
                                 decltype(std::declval<vpux::Const::DataRef>()._dataKey)>,
                  "We rely on data keys being StringRefs. If this is not true, the DataRef should copy the data key "
                  "instead of referencing it.");

    if (!_dataKey.empty()) {
        fetchDataManager(_ctx).addRef(_dataKey);
    }
}

DataRef::~DataRef() {
    if (!_dataKey.empty()) {
        fetchDataManager(_ctx).dropRef(_dataKey);
    }
}

DataRef::DataRef(const DataRef& x): DataRef(x._ctx, x._dataKey) {
}

DataRef& DataRef::operator=(const DataRef& x) {
    DataRef tmp(x);
    swap(*this, tmp);
    return *this;
}

DataRef::DataRef(DataRef&& x): _ctx(std::exchange(x._ctx, nullptr)), _dataKey(std::exchange(x._dataKey, {})) {
}

DataRef& DataRef::operator=(DataRef&& x) {
    DataRef tmp(std::move(x));
    swap(*this, tmp);
    return *this;
}

ConstDataManagerInterface::ConstDataManagerInterface(mlir::Dialect* dialect): Base(dialect) {
}

void ConstDataManagerInterface::addRef(mlir::StringRef key) {
    std::lock_guard<std::mutex> guard(_refMutex);
    ++_refCounts[key];
}

void ConstDataManagerInterface::dropRef(mlir::StringRef key) {
    std::lock_guard<std::mutex> guard(_refMutex);
    const auto it = _refCounts.find(key);
    assert(it != _refCounts.end());
    const auto refCount = --it->second;
    if (refCount == 0) {
        _refCounts.erase(it);  // keep this hash table small, don't expect this value to ever come back

        // Note: we cannot remove entries - current behavior implicitly ensures
        // keys are collision-free this way? - but we can *replace* entries.
        // thus, replace real constant data with a dummy nullptr entry: the data
        // itself is cleared but the "metadata" (asm resource blob) still
        // remains until MLIRContext is deleted.
        mlir::AsmResourceBlob dummy(
                mlir::ArrayRef<char>{}, [](char*, size_t, size_t) {}, false);

        auto& realBlobManager = mlir::DenseResourceElementsHandle::getManagerInterface(getContext());
        realBlobManager.update(key, std::move(dummy));
    }
}
}  // namespace vpux::Const
