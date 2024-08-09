//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/small_vector.hpp"

#include <cassert>

namespace vpux {

template <typename T>
class RingBuffer final {
public:
    RingBuffer() = default;

    explicit RingBuffer(size_t capacity): _storage(capacity), _putInd(capacity), _getInd(0), _size(0) {
    }

public:
    template <typename T1>
    void push(T1&& val) {
        ++_size;

        if (++_putInd >= _storage.size()) {
            _putInd = 0;
        }

        _storage[_putInd] = std::forward<T1>(val);
    }

    void pop() {
        assert(_size > 0);
        --_size;

        if (++_getInd >= _storage.size()) {
            _getInd = 0;
        }
    }

    void clear() {
        _putInd = _storage.size();
        _getInd = 0;
        _size = 0;
    }

    void reset(size_t capacity) {
        _storage.resize(capacity);
        clear();
    }

public:
    T& front() {
        return _storage[_getInd];
    }
    const T& front() const {
        return _storage[_getInd];
    }

    T& back() {
        return _storage[_putInd];
    }
    const T& back() const {
        return _storage[_putInd];
    }

    bool empty() const {
        return _size == 0;
    }
    bool full() const {
        return _size == _storage.size();
    }

    size_t size() const {
        return _size;
    }
    size_t capacity() const {
        return _storage.size();
    }

private:
    SmallVector<T> _storage;
    size_t _putInd = 0;
    size_t _getInd = 0;
    size_t _size = 0;
};

}  // namespace vpux
