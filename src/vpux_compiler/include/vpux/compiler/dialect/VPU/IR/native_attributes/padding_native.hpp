//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux {
namespace VPU {
class Padding {
private:
    int64_t left;
    int64_t right;
    int64_t top;
    int64_t bottom;

public:
    Padding() = default;
    Padding(int64_t leftPad, int64_t rightPad, int64_t topPad, int64_t bottomPad)
            : left(leftPad), right(rightPad), top(topPad), bottom(bottomPad) {
    }
    ~Padding() = default;

    friend bool operator==(const Padding& lhs, const Padding& rhs) {
        return lhs.left == rhs.left && lhs.right == rhs.right && lhs.top == rhs.top && lhs.bottom == rhs.bottom;
    }

    int64_t getTopPad() const {
        return top;
    }
    int64_t getBottomPad() const {
        return bottom;
    }
    int64_t getLeftPad() const {
        return left;
    }
    int64_t getRightPad() const {
        return right;
    }

    static Padding getClassFromAttr(PaddingAttr paddingAttr) {
        if (paddingAttr == nullptr) {
            return {};
        }

        auto left = paddingAttr.getLeft().getInt();
        auto right = paddingAttr.getRight().getInt();
        auto top = paddingAttr.getTop().getInt();
        auto bottom = paddingAttr.getBottom().getInt();

        return Padding(left, right, top, bottom);
    }

    static PaddingAttr getAttrFromClass(mlir::MLIRContext* ctx, const Padding& padding) {
        auto topAttr = vpux::getIntAttr(ctx, padding.top);
        auto bottomAttr = vpux::getIntAttr(ctx, padding.bottom);
        auto leftAttr = vpux::getIntAttr(ctx, padding.left);
        auto rightAttr = vpux::getIntAttr(ctx, padding.right);

        return PaddingAttr::get(ctx, leftAttr, rightAttr, topAttr, bottomAttr);
    };

    void printFormat(llvm::raw_ostream& stream) const {
        std::unordered_map<std::string, int64_t> map;
        map["left"] = left;
        map["right"] = right;
        map["top"] = top;
        map["bottom"] = bottom;
        printTo(stream, "pads = ");
        vpux::MapFormatProvider::format(map, stream, {});
    }
};
}  // namespace VPU
}  // namespace vpux
