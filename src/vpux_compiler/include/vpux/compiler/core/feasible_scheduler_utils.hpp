//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cassert>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>

namespace vpux {

struct ScheduledOpOneResource {
    enum class EResRelation { PRODUCER = 0, CONSUMER = 1 };

    struct ResourceView {
        mlir::Value buffer;
        llvm::SmallVector<int64_t> offset;
        llvm::SmallVector<int64_t> shape;
        llvm::SmallVector<int64_t> staticStrides;

        bool operator==(const ResourceView& o) const {
            return (buffer == o.buffer) && (offset == o.offset) && (shape == o.shape) &&
                   (staticStrides == o.staticStrides);
        }
    };

    using OperationType = size_t;
    ScheduledOpOneResource() = default;

    ScheduledOpOneResource(OperationType op, size_t start, size_t end, EResRelation resRelation,
                           const std::optional<ResourceView>& resourceView = std::nullopt)
            : _op(op), _addressStart(start), _addressEnd(end), _resRelation(resRelation), _resourceView(resourceView) {
    }

    ScheduledOpOneResource(const ScheduledOpOneResource& o) = default;

    ScheduledOpOneResource& operator=(const ScheduledOpOneResource& o) = default;

    bool operator==(const ScheduledOpOneResource& o) const {
        return (_op == o._op) && (_addressStart == o._addressStart) && (_addressEnd == o._addressEnd) &&
               (_resRelation == o._resRelation) && (_resourceView == o._resourceView);
    }

    bool operator<(const ScheduledOpOneResource& o) const {
        if (_addressStart == o._addressStart && _addressEnd == o._addressEnd) {
            return _op < o._op;
        }
        return (_addressStart != o._addressStart) ? (_addressStart < o._addressStart) : (_addressEnd < o._addressEnd);
    }
    ~ScheduledOpOneResource() = default;

    // Check if operation resource can overlap in execution with other operation
    // Example:
    // Root buffer range [0 100]
    //  DMA1 produces first half [0 50]
    //  DMA2 produces second half [51 100]
    // Above tasks can execute in parralel even though they write to single root buffer
    bool canOverlap(const ScheduledOpOneResource& o) const {
        // no overlap
        if (_addressEnd < o._addressStart || _addressStart > o._addressEnd) {
            return true;
        }

        // Limit overlap to cases where whole root buffer have same start and end
        // as this is what can be seen in our compilation pipeline where subviews of buffer
        // refer to same root buffer (same range).
        if (_addressStart == o._addressStart && _addressEnd == o._addressEnd) {
            if (!_resourceView.has_value() || !o._resourceView.has_value()) {
                return false;
            }

            // If same range and buffer, check if range access details specify non-overlapping views of same buffer
            // Require same staticStrides but different offset. Shape can be ignored
            // It is possible to have different strides but such case would require more detailed check which
            // would include all sub view properties (offset, shape, stride) and is not something that would be
            // spotted in real compilation besides some artificial test case scenarios
            auto resVal1 = _resourceView.value();
            auto resVal2 = o._resourceView.value();
            if (resVal1.buffer == resVal2.buffer && resVal1.staticStrides == resVal2.staticStrides &&
                resVal1.offset != resVal2.offset) {
                return true;
            }
        }

        return false;
    }

    OperationType _op{};
    size_t _addressStart{};
    size_t _addressEnd{};
    EResRelation _resRelation{EResRelation::PRODUCER};
    std::optional<ResourceView> _resourceView{};
};  // struct ScheduledOpOneResource //

}  // namespace vpux
