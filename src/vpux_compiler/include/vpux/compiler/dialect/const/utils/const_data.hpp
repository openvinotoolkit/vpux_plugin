//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"

#include <llvm/Support/MemAlloc.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace vpux::Const {

/*! @brief Represents constant data storage.
 */
struct ConstData {
    //! @brief Creates new object, allocating the underlying memory.
    template <typename T>
    static ConstData allocate(std::size_t size) {
        static_assert(alignof(T) <= alignof(std::max_align_t),
                      "Object alignment larger than the default is not supported");
        return ConstData::allocateBytes(sizeof(T) * size);
    }

    //! @brief Creates new object that references an external memory. In this
    //! case, the data is *not* owned by this object.
    static ConstData fromRawBuffer(const void* ptr, std::size_t byteSize);

    //! @brief Returns whether this data is "mutable". Thus far, the object is
    //! mutable if it owns its data (i.e. the data is allocated internally).
    bool isMutable() const {
        return _delete != nullptr;
    }

    template <typename T = char>
    ArrayRef<T> data() const {
        assert(_size % sizeof(T) == 0 && "Casting to a type that could not be represented");
        return ArrayRef(reinterpret_cast<const T*>(_ptr), _size / sizeof(T));
    }
    template <typename T = char>
    MutableArrayRef<T> mutableData() {
        assert(isMutable() && "This data is read-only");
        assert(_size % sizeof(T) == 0 && "Casting to a type that could not be represented");
        return MutableArrayRef(reinterpret_cast<T*>(_ptr), _size / sizeof(T));
    }
    std::size_t size() const {
        return _size;
    }

    ConstData() = default;
    ConstData(const ConstData&) = delete;
    ConstData(ConstData&& x)
            : _ptr(std::exchange(x._ptr, nullptr)),
              _size(std::exchange(x._size, 0)),
              _delete(std::exchange(x._delete, nullptr)) {
    }
    ConstData& operator=(const ConstData&) = delete;
    ConstData& operator=(ConstData&& x) {
        ConstData tmp(std::move(x));
        swap(*this, tmp);
        return *this;
    }
    ~ConstData() {
        if (_delete) {
            _delete(_ptr, _size);
        }
    }
    friend void swap(ConstData& x, ConstData& y) {
        using std::swap;
        swap(x._ptr, y._ptr);
        swap(x._size, y._size);
        swap(x._delete, y._delete);
    }

private:
    using DeleterFn = void (*)(void*, std::size_t);

    void* _ptr = nullptr;
    std::size_t _size = 0;
    DeleterFn _delete = nullptr;

    static ConstData allocateBytes(std::size_t byteSize);
};

}  // namespace vpux::Const
