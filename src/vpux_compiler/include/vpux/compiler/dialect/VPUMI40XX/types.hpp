//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"

#include <cstddef>
#include <functional>
#include <iterator>

namespace vpux::VPUMI40XX {

namespace details {

template <class Iterator, auto NextFunction>
class TaskIterator {
public:
    // TaskIterator depends on pointee method to proceed to next element
    // end (post-last element) iterator has empty task / nullptr as pointee
    // so std::prev(end) is ill-formed, this is why forward iterator tag
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = VPURegMapped::TaskOpInterface;
    // TaskOpInterface is already a wrapper around a pointer
    // no sense in TaskOpInterface*, especially that it'd require to keep
    // storage of TaskOpInterface objects in addition to ValueRange's in OpRanges
    using pointer = value_type;
    using reference = value_type&;
    using const_reference = const value_type&;

    TaskIterator() = default;
    TaskIterator(pointer task): pointee(task) {
    }

    const_reference operator*() const {
        return pointee;
    }

    reference operator*() {
        return pointee;
    }

    pointer operator->() {
        return pointee;
    }

    auto operator++() {
        if (pointee) {
            pointee = (pointee.*NextFunction)();
        }
        return static_cast<Iterator&>(*this);
    }

    Iterator operator++(int) {
        auto self = static_cast<Iterator&>(*this);
        const auto current = self;
        ++self;
        return current;
    }

protected:
    pointer pointee;
};

template <class Iterator>
class TaskRange {
public:
    using value_type = typename Iterator::value_type;

    TaskRange() = default;
    explicit TaskRange(mlir::Value begin): beginIterator(begin.getDefiningOp<VPURegMapped::TaskOpInterface>()) {
    }

    Iterator begin() {
        return beginIterator;
    }
    Iterator end() {
        return endIterator;
    }
    bool empty() const {
        return beginIterator == endIterator;
    }

private:
    // MLIR framework doesn't implement const model (see https://mlir.llvm.org/docs/Rationale/UsageOfConst/)
    // inherently, mlir::Operation const* isn't supported, so const task iterator doesn't make sense
    // as dereference operator'd either'd attempt to return const TaskOpInterface which goes against MLIR design
    // or behave the same way as mutable task iterator, that's confusing
    Iterator begin() const = delete;
    Iterator end() const = delete;

    Iterator beginIterator;
    Iterator endIterator;
};

}  // namespace details

template <class DirectIterator, auto NextFunction>
inline bool operator==(const details::TaskIterator<DirectIterator, NextFunction>& lhs,
                       const details::TaskIterator<DirectIterator, NextFunction>& rhs) {
    return (*lhs) == (*rhs);
}

template <class DirectIterator, auto NextFunction>
inline bool operator!=(const details::TaskIterator<DirectIterator, NextFunction>& lhs,
                       const details::TaskIterator<DirectIterator, NextFunction>& rhs) {
    return !(lhs == rhs);
}

class TaskForwardIterator :
        public details::TaskIterator<TaskForwardIterator, &VPURegMapped::TaskOpInterface::getNextTask> {
public:
    using Base = details::TaskIterator<TaskForwardIterator, &VPURegMapped::TaskOpInterface::getNextTask>;
    using Base::Base;
};

class TaskBackwardIterator :
        public details::TaskIterator<TaskBackwardIterator, &VPURegMapped::TaskOpInterface::getPreviousTask> {
public:
    using Base = details::TaskIterator<TaskBackwardIterator, &VPURegMapped::TaskOpInterface::getPreviousTask>;
    using Base::Base;
};

using TaskForwardRange = details::TaskRange<TaskForwardIterator>;
using TaskBackwardRange = details::TaskRange<TaskBackwardIterator>;

}  // namespace vpux::VPUMI40XX
