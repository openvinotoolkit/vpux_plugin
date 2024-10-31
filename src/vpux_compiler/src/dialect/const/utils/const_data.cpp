//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/const_data.hpp"

#include <cstddef>

namespace vpux::Const {
ConstData ConstData::allocateBytes(std::size_t byteSize) {
    ConstData data;
    data._ptr = llvm::allocate_buffer(byteSize, alignof(std::max_align_t));
    data._delete = [](void* data, std::size_t size) {
        llvm::deallocate_buffer(data, size, alignof(std::max_align_t));
    };
    data._size = byteSize;
    data._kind = DataKind::Internal;
    return data;
}

ConstData ConstData::fromRawBuffer(const void* ptr, std::size_t byteSize) {
    ConstData data;
    // Note: const casting here is fine, DataKind::External would protect us
    // from modification.
    data._ptr = const_cast<void*>(ptr);
    data._size = byteSize;
    data._kind = DataKind::External;
    return data;
}
}  // namespace vpux::Const
