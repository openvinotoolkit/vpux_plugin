//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"

#include <cstdint>

namespace vpux {

class Slice final {
public:
    Slice() = default;

    Slice(int64_t begin, int64_t end): _begin(begin), _end(end) {
        VPUX_THROW_UNLESS(end >= begin, "Wrong slice range '[{0}, {1})'", begin, end);
    }

public:
    int64_t begin() const {
        return _begin;
    }

    int64_t end() const {
        return _end;
    }

    int64_t length() const {
        return end() - begin();
    }

public:
    bool intersects(const Slice& other) const {
        return (begin() < other.end()) && (other.begin() < end());
    }

    bool contains(const Slice& other) const {
        return (begin() <= other.begin()) && (_end >= other.end());
    }

    // Represents `this` range as sub-slice of the `parent`.
    Slice asSubSlice(const Slice& parent) const {
        VPUX_THROW_UNLESS(parent.contains(*this), "Slice '{0}' is not a sub-slice of '{1}'", *this, parent);

        return {begin() - parent.begin(), end() - parent.begin()};
    }

public:
    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "[{0}, {1})", begin(), end());
    }

private:
    int64_t _begin = 0;
    int64_t _end = 0;
};

inline bool operator==(const Slice& s1, const Slice& s2) {
    return s1.begin() == s2.begin() && s1.end() == s2.end();
}
inline bool operator!=(const Slice& s1, const Slice& s2) {
    return !(s1 == s2);
}

}  // namespace vpux
