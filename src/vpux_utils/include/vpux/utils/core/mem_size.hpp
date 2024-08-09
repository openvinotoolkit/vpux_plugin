//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Typed representation for memory sizes.
//

#pragma once

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <climits>

namespace vpux {

//
// MemType
//

enum class MemType {
    Bit,
    Byte,
    KB,
    MB,
    GB,
};

StringLiteral stringifyEnum(MemType val);

//
// MemMultiplier
//

template <MemType FROM, MemType TO>
struct MemMultiplier;

template <MemType TYPE>
struct MemMultiplier<TYPE, TYPE> final {
    static constexpr int64_t value = 1;
};
template <>
struct MemMultiplier<MemType::Byte, MemType::Bit> final {
    static constexpr int64_t value = CHAR_BIT;
};
template <>
struct MemMultiplier<MemType::KB, MemType::Byte> final {
    static constexpr int64_t value = 1024;
};
template <>
struct MemMultiplier<MemType::MB, MemType::KB> final {
    static constexpr int64_t value = 1024;
};
template <>
struct MemMultiplier<MemType::GB, MemType::MB> final {
    static constexpr int64_t value = 1024;
};

template <>
struct MemMultiplier<MemType::MB, MemType::Byte> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::MB, MemType::KB>::value * MemMultiplier<MemType::KB, MemType::Byte>::value;
};
template <>
struct MemMultiplier<MemType::MB, MemType::Bit> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::MB, MemType::Byte>::value * MemMultiplier<MemType::Byte, MemType::Bit>::value;
};

template <>
struct MemMultiplier<MemType::GB, MemType::KB> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::GB, MemType::MB>::value * MemMultiplier<MemType::MB, MemType::KB>::value;
};
template <>
struct MemMultiplier<MemType::GB, MemType::Byte> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::GB, MemType::KB>::value * MemMultiplier<MemType::KB, MemType::Byte>::value;
};
template <>
struct MemMultiplier<MemType::GB, MemType::Bit> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::GB, MemType::Byte>::value * MemMultiplier<MemType::Byte, MemType::Bit>::value;
};

//
// MemSize
//

namespace details {

template <MemType FROM, MemType TO, typename = void>
struct HasMemMultiplier {
    static constexpr bool value = false;
};

template <MemType FROM, MemType TO>
struct HasMemMultiplier<FROM, TO, std::enable_if_t<MemMultiplier<FROM, TO>::value != 0>> {
    static constexpr bool value = true;
};

}  // namespace details

template <MemType TYPE>
class MemSize final {
public:
    constexpr MemSize() = default;
    constexpr explicit MemSize(int64_t size): _size(size) {
    }

public:
    template <MemType OTHER>
    constexpr MemSize(const MemSize<OTHER>& size, require_t<details::HasMemMultiplier<OTHER, TYPE>>* = nullptr)
            : _size(size.count() * MemMultiplier<OTHER, TYPE>::value) {
    }

    template <MemType OTHER>
    MemSize(const MemSize<OTHER>& size, require_t<details::HasMemMultiplier<TYPE, OTHER>>* = nullptr) {
        constexpr auto mult = MemMultiplier<TYPE, OTHER>::value;

        VPUX_THROW_UNLESS(size.count() % mult == 0, "Can't convert {0} {1} to {2}", size.count(), OTHER, TYPE);
        _size = size.count() / mult;
    }

    template <class OTHER>
    constexpr OTHER to() const {
        return OTHER(*this);
    }

public:
    constexpr auto count() const {
        return _size;
    }

public:
    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "{0} {1}", count(), TYPE);
    }

private:
    int64_t _size = 0;
};

template <MemType TYPE>
constexpr MemSize<TYPE> operator+(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return MemSize<TYPE>(size1.count() + size2.count());
}
template <MemType TYPE>
constexpr MemSize<TYPE> operator-(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return MemSize<TYPE>(size1.count() - size2.count());
}
template <MemType TYPE>
constexpr int64_t operator%(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() % size2.count();
}

template <MemType TYPE>
MemSize<TYPE>& operator+=(MemSize<TYPE>& size1, MemSize<TYPE> size2) {
    return size1 = size1 + size2;
}
template <MemType TYPE>
MemSize<TYPE>& operator-=(MemSize<TYPE>& size1, MemSize<TYPE> size2) {
    return size1 = size1 - size2;
}
template <MemType TYPE>
MemSize<TYPE>& operator%=(MemSize<TYPE>& size1, MemSize<TYPE> size2) {
    return size1 = size1 % size2;
}

template <MemType TYPE>
constexpr MemSize<TYPE> operator*(MemSize<TYPE> size, int64_t mult) {
    return MemSize<TYPE>(size.count() * mult);
}
template <MemType TYPE>
constexpr MemSize<TYPE> operator*(int64_t mult, MemSize<TYPE> size) {
    return MemSize<TYPE>(size.count() * mult);
}
template <MemType TYPE>
constexpr MemSize<TYPE> operator/(MemSize<TYPE> size, int64_t div) {
    return MemSize<TYPE>(size.count() / div);
}

template <MemType TYPE>
MemSize<TYPE>& operator*=(MemSize<TYPE>& size, int64_t mult) {
    return size = size * mult;
}
template <MemType TYPE>
MemSize<TYPE>& operator/=(MemSize<TYPE>& size, int64_t div) {
    return size = size / div;
}

template <MemType TYPE>
constexpr bool operator==(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() == size2.count();
}
template <MemType TYPE>
constexpr bool operator!=(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() != size2.count();
}
template <MemType TYPE>
constexpr bool operator>(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() > size2.count();
}
template <MemType TYPE>
constexpr bool operator>=(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() >= size2.count();
}
template <MemType TYPE>
constexpr bool operator<(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() < size2.count();
}
template <MemType TYPE>
constexpr bool operator<=(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() <= size2.count();
}

using Bit = MemSize<MemType::Bit>;
using Byte = MemSize<MemType::Byte>;
using KB = MemSize<MemType::KB>;
using MB = MemSize<MemType::MB>;
using GB = MemSize<MemType::GB>;

inline constexpr Bit operator""_Bit(unsigned long long const size) {
    return Bit(static_cast<int64_t>(size));
}
inline constexpr Byte operator""_Byte(unsigned long long const size) {
    return Byte(static_cast<int64_t>(size));
}
inline constexpr KB operator""_KB(unsigned long long const size) {
    return KB(static_cast<int64_t>(size));
}
inline constexpr MB operator""_MB(unsigned long long const size) {
    return MB(static_cast<int64_t>(size));
}
inline constexpr GB operator""_GB(unsigned long long const size) {
    return GB(static_cast<int64_t>(size));
}

//
// align memSize
//

template <MemType TYPE, MemType OTHER>
constexpr MemSize<TYPE> alignMemSize(MemSize<TYPE> memSize, MemSize<OTHER> alignment) {
    auto alignedSize = memSize.count();
    if (auto mult = vpux::MemMultiplier<OTHER, TYPE>::value) {
        auto alignmetCount = alignment.template to<MemSize<TYPE>>().count();
        if (memSize.count() % alignmetCount) {
            alignedSize = memSize.count() + (alignmetCount - (memSize.count() % alignmetCount));
        }
    }
    return MemSize<TYPE>(alignedSize);
}

}  // namespace vpux
