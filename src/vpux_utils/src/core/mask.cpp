//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/mask.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

//
// Mask
//

constexpr size_t vpux::Mask::NUM_BITS;

Mask vpux::Mask::fromCode(StorageType code) {
    Mask mask;
    mask._code = code;
    return mask;
}

Mask vpux::Mask::fromCount(int32_t count) {
    VPUX_THROW_UNLESS(count >= 0, "Can't create Mask from count '{0}'", count);
    VPUX_THROW_UNLESS(static_cast<size_t>(count) <= NUM_BITS, "Can't create Mask from count '{0}'", count);

    Mask mask;
    mask._code = static_cast<StorageType>((static_cast<int64_t>(1) << count) - 1);
    return mask;
}

Mask vpux::Mask::fromRange(int32_t start, int32_t end) {
    VPUX_THROW_UNLESS(start >= 0 && end >= 0 && start <= end, "Can't create Mask from range '[{0}, {1})'", start, end);
    VPUX_THROW_UNLESS(static_cast<size_t>(end) <= NUM_BITS, "Can't create Mask from range '[{0}, {1})'", start, end);

    Mask mask;
    mask._code = static_cast<StorageType>(((1 << (end - start)) - 1) << start);
    return mask;
}

Mask vpux::Mask::fromIndexes(ArrayRef<int32_t> indexes) {
    Mask mask;
    for (auto ind : indexes) {
        VPUX_THROW_UNLESS(ind >= 0, "Can't create Mask from indexes '{0}'", indexes);
        VPUX_THROW_UNLESS(static_cast<size_t>(ind) < NUM_BITS, "Can't create Mask from indexes '{0}'", indexes);

        mask._code |= static_cast<StorageType>(1 << ind);
    }
    return mask;
}

size_t vpux::Mask::size() const {
    size_t res = 0;

    for (auto temp = _code; temp > 0; temp >>= 1) {
        if (temp & 1) {
            ++res;
        }
    }

    return res;
}

int32_t vpux::Mask::operator[](size_t ind) const {
    VPUX_THROW_UNLESS(ind < size(), "Mask index '{0}' is out of range '{1}'", ind, size());

    int32_t res = 0;

    for (auto temp = _code; temp > 0; temp >>= 1, ++res) {
        if (temp & 1) {
            if (ind == 0) {
                return res;
            }

            --ind;
        }
    }

    return -1;
}

bool vpux::Mask::isContinous() const {
    bool metSet = false;
    bool metUnsetAfterSet = false;

    for (auto temp = _code; temp > 0; temp >>= 1) {
        if ((temp & 1) && metUnsetAfterSet) {
            return false;
        }

        if ((temp & 1) == 0 && metSet) {
            metUnsetAfterSet = true;
        }

        if (temp & 1) {
            metSet = true;
        }
    }

    return true;
}
