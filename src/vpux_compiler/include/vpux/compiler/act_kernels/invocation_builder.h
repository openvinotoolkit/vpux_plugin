//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "nce2p7.h"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/utils/schema.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>

namespace vpux {

/**
 * Helper builder for creation activation-
 * shaves invocation in memory arguments
 */
class InvocationBuilder {
public:
    using PatchCallbackType = std::function<void(MutableArrayRef<uint8_t>, size_t)>;

public:
    InvocationBuilder(size_t dataOffset, Logger log): _win_e_offset(dataOffset), _log(log) {
    }

    /**
     * register serialisation for given invocation argument, might be MemRefType or any other supported types
     * @param operand
     */
    void addArg(mlir::Attribute attr);
    void addTensorArg(mlir::Value value, const MVCNN::TensorReference* tenorRef, vpux::VPU::ArchKind archKind);

    /**
     * actual serialising routine
     */
    SmallVector<uint8_t> store() const;

private:
    template <class T>
    void appendValue(SmallVector<char>& storage, const T& anyValue) {
        ArrayRef<char> valueAsArray(reinterpret_cast<const char*>(&anyValue), sizeof(anyValue));
        storage.insert(storage.end(), valueAsArray.begin(), valueAsArray.end());
    }

    void parseBasicAttrTypes(mlir::Attribute attr);

    /**
     * create a patch entry, that can be further updated
     * @tparam U structure type
     * @tparam T field type
     */
    template <class U, class T>
    PatchCallbackType createPatchPoint(const T& patcher) {
        return [offset = _scalarStorage.size(), arrayOffset = _arrayStorage.size(), patcher](
                       MutableArrayRef<uint8_t> serialStorage, size_t updateTo) {
            auto& origObject = reinterpret_cast<U&>(*(serialStorage.begin() + offset));
            patcher(origObject, checked_cast<uint32_t>(updateTo + arrayOffset));
        };
    }

private:
    size_t _win_e_offset;  //  offset of the beginning of invocation args within expected WIN_E
    Logger _log;

    SmallVector<char> _scalarStorage;  // keeps scalar elements and memref metadata
    SmallVector<char> _arrayStorage;   // keeps arrays elements

    SmallVector<PatchCallbackType> _deferredPointers;
};

}  // namespace vpux
