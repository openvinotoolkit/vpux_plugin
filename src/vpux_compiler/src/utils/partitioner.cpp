//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/partitioner.hpp"

#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include <cassert>

using namespace vpux;

namespace {

class PartitionerValidator final {
public:
    explicit PartitionerValidator(const Partitioner& p): _p(p) {
        validate();
    }
    PartitionerValidator(const PartitionerValidator& other) = delete;
    PartitionerValidator& operator=(const PartitionerValidator&) = delete;
    ~PartitionerValidator() {
        validate();
    }

    void validate() const {
#ifndef NDEBUG
        const auto& gaps = _p.gaps();
        for (size_t i = 0; i < gaps.size(); ++i) {
            auto& g = gaps[i];

            assert(g.begin != InvalidAddress);
            assert(g.end != InvalidAddress);
            assert(g.end > g.begin);

            if (i != 0) {
                assert(g.begin >= gaps[i - 1].end);
            }
        }
#endif
    }

    void checkNewGap(AddressType addr, AddressType size) const {
        VPUX_UNUSED(addr);
        VPUX_UNUSED(size);

#ifndef NDEBUG
        for (const auto& gap : _p.gaps()) {
            assert(!Partitioner::intersects(addr, size, gap.begin, gap.size()));
        }
#else
        VPUX_UNUSED(_p);
#endif
    }

private:
    const Partitioner& _p;
};

}  // namespace

vpux::Partitioner::Partitioner(AddressType totalSize): _totalSize(totalSize) {
    assert(_totalSize > 0);
    _gaps.push_back({0, _totalSize});
}

AddressType vpux::Partitioner::alloc(AddressType size, AddressType alignment, Direction dir) {
    assert(size > 0);
    assert(alignment > 0);

    const PartitionerValidator v(*this);

    return chooseMinimalGap(size, alignment, dir);
}

void vpux::Partitioner::allocFixed(AddressType addr, AddressType size) {
    assert(addr != InvalidAddress);
    assert(size > 0);
    assert(addr + size <= _totalSize);
    assert(!_gaps.empty());

    const PartitionerValidator v(*this);

    auto it = std::lower_bound(_gaps.begin(), _gaps.end(), Gap{addr, 0}, [](const Gap& g1, const Gap& g2) {
        return g1.begin < g2.begin;
    });

    const bool hasEq = (it != _gaps.end() && it->begin == addr);

    if (hasEq) {
        auto& elem = *it;
        assert(elem.size() >= size);  // client is aware of this demand

        if (elem.size() == size) {
            _gaps.erase(it);
        } else {
            elem.begin += size;
        }
    } else {
        assert(it != _gaps.begin());
        --it;

        auto& elem = *it;
        const auto end = addr + size;

        assert(elem.begin < addr);
        assert(elem.end >= end);

        if (elem.end == end) {
            elem.end = addr;
        } else {
            assert(elem.end >= end);

            const auto tailSize = elem.end - end;
            assert(tailSize > 0);

            elem.end = addr;
            _gaps.insert(it + 1, Gap{end, end + tailSize});
        }
    }
}

void vpux::Partitioner::free(AddressType addr, AddressType size) {
    assert(addr != InvalidAddress);
    assert(size > 0);

    const PartitionerValidator v(*this);

    v.checkNewGap(addr, size);

    const auto end = addr + size;
    if (_gaps.empty()) {
        _gaps.push_back(Gap{addr, end});
        return;
    }

    const auto prepend = [](Gap& gap, AddressType addr, AddressType size) {
        VPUX_UNUSED(addr);

        assert((addr + size) == gap.begin);
        assert(gap.begin >= size);

        gap.begin -= size;
    };
    const auto append = [](Gap& gap, AddressType addr, AddressType size) {
        VPUX_UNUSED(addr);

        assert(addr == gap.end);

        gap.end += size;
    };

    auto& firstGap = _gaps.front();
    if (end == firstGap.begin) {
        prepend(firstGap, addr, size);
        return;
    } else if (end < firstGap.begin) {
        _gaps.insert(_gaps.begin(), Gap{addr, end});
        return;
    }

    for (size_t i = 0; i < _gaps.size() - 1; ++i) {
        auto& gap = _gaps[i];
        auto& nextGap = _gaps[i + 1];

        if (addr >= gap.end) {
            if (end <= nextGap.begin) {
                if (addr == gap.end) {
                    if (end == nextGap.begin) {
                        append(gap, addr, size + nextGap.size());
                        _gaps.erase(_gaps.begin() + static_cast<ptrdiff_t>(i + 1));
                    } else {
                        append(gap, addr, size);
                    }
                } else if (end == nextGap.begin) {
                    prepend(nextGap, addr, size);
                } else {
                    _gaps.insert(_gaps.begin() + static_cast<ptrdiff_t>(i + 1), Gap{addr, end});
                }

                return;
            }
        }
    }

    auto& lastGap = _gaps.back();
    if (end < lastGap.begin) {
        _gaps.insert(_gaps.begin() + static_cast<ptrdiff_t>(_gaps.size() - 1), Gap{addr, end});
    } else if (end == lastGap.begin) {
        prepend(lastGap, addr, size);
    } else if (addr == lastGap.end) {
        append(lastGap, addr, size);
    } else {
        _gaps.push_back(Gap{addr, end});
    }
}

AddressType vpux::Partitioner::totalFreeSize() const {
    return std::accumulate(_gaps.begin(), _gaps.end(), AddressType{0}, [](AddressType res, const Gap& g) {
        return res + g.size();
    });
}

AddressType vpux::Partitioner::maxFreeSize() const {
    return std::accumulate(_gaps.begin(), _gaps.end(), AddressType{0}, [](AddressType res, const Gap& g) {
        return std::max(res, g.size());
    });
}

AddressType vpux::Partitioner::getAddrFromGap(size_t pos, AddressType size, AddressType alignment, Direction dir) {
    const auto& g = _gaps[pos];

    if (g.size() < size) {
        return InvalidAddress;
    }

    if (dir == Direction::Up) {
        const auto alignedBegin = alignValUp(g.begin, alignment);

        if ((alignedBegin + size) > g.end) {
            return InvalidAddress;
        }

        return alignedBegin;
    } else {
        auto alignedBegin = alignValUp(g.end - size, alignment);

        if ((alignedBegin + size) > g.end) {
            if (g.begin + alignment > alignedBegin) {
                return InvalidAddress;
            }

            alignedBegin -= alignment;
        }

        if ((alignedBegin + size) > g.end) {
            return InvalidAddress;
        }

        return alignedBegin;
    }
}

AddressType vpux::Partitioner::useGap(size_t pos, AddressType alignedBegin, AddressType size) {
    auto& g = _gaps[pos];

    assert(alignedBegin >= g.begin);
    assert(alignedBegin + size <= g.end);

    const auto newGapLeft = alignedBegin - g.begin;
    const auto newGapRight = g.end - (alignedBegin + size);

    if (bool allocFitsExactlyIntoThisGap = (newGapLeft == 0 && newGapRight == 0); allocFitsExactlyIntoThisGap) {
        _gaps.erase(_gaps.begin() + static_cast<ptrdiff_t>(pos));
    } else if (bool allocIsAdjacentToRightBorder = (newGapLeft > 0 && newGapRight == 0); allocIsAdjacentToRightBorder) {
        g.end = alignedBegin;
    } else if (bool allocIsAdjacentToLeftBorder = (newGapLeft == 0 && newGapRight > 0); allocIsAdjacentToLeftBorder) {
        g.begin = alignedBegin + size;
    } else {  // add new gap
        const auto origBegin = g.begin;
        g.begin = alignedBegin + size;
        _gaps.insert(_gaps.begin() + static_cast<ptrdiff_t>(pos), Gap{origBegin, alignedBegin});
    }

    return alignedBegin;
}

AddressType vpux::Partitioner::chooseMinimalGap(AddressType size, AddressType alignment, Direction dir) {
    if (_gaps.empty()) {
        return InvalidAddress;
    }

    const auto numGaps = _gaps.size();

    int minGapInd = -1;
    auto minGapSize = std::numeric_limits<AddressType>::max();

    // The last gap in current direction has the lowest priority,
    // it is checked only if there is no other suitable gap.

    if (dir == Direction::Up) {
        for (size_t i = 0; i < numGaps - 1; ++i) {
            const auto alignedBegin = getAddrFromGap(i, size, alignment, dir);
            if (alignedBegin != InvalidAddress) {
                if (_gaps[i].size() < minGapSize) {
                    minGapInd = static_cast<int>(i);
                    minGapSize = _gaps[i].size();
                }
            }
        }

        if (minGapInd == -1) {
            const auto alignedBegin = getAddrFromGap(numGaps - 1, size, alignment, dir);
            if (alignedBegin != InvalidAddress) {
                minGapInd = static_cast<int>(numGaps - 1);
            }
        }
    } else {
        for (size_t i = numGaps - 1; i >= 1; --i) {
            const auto alignedBegin = getAddrFromGap(i, size, alignment, dir);
            if (alignedBegin != InvalidAddress) {
                if (_gaps[i].size() < minGapSize) {
                    minGapInd = static_cast<int>(i);
                    minGapSize = _gaps[i].size();
                }
            }
        }

        if (minGapInd == -1) {
            const auto alignedBegin = getAddrFromGap(0, size, alignment, dir);
            if (alignedBegin != InvalidAddress) {
                minGapInd = 0;
            }
        }
    }

    if (minGapInd != -1) {
        const auto alignedBegin = getAddrFromGap(static_cast<size_t>(minGapInd), size, alignment, dir);
        return useGap(static_cast<size_t>(minGapInd), alignedBegin, size);
    }

    return InvalidAddress;
}

bool vpux::Partitioner::intersects(AddressType addr1, AddressType size1, AddressType addr2, AddressType size2) {
    assert(size1 > 0);
    assert(size2 > 0);

    const auto end1 = addr1 + size1;
    const auto end2 = addr2 + size2;

    const auto inRange = [](AddressType begin, AddressType end, AddressType val) {
        return val >= begin && val < end;
    };

    return inRange(addr1, end1, addr2) || inRange(addr1, end1, end2 - 1) || inRange(addr2, end2, addr1) ||
           inRange(addr2, end2, end1 - 1);
}
