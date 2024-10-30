//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/batch.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

DebatchedCallOpData DebatchedCallOpData::deserialize(const SmallVector<ValueType>& array) {
    static constexpr uint32_t minElementsInArray = 2;
    VPUX_THROW_UNLESS(array.size() >= minElementsInArray,
                      "Cannot deserialzie DebatchedCallOpData from array, expected elements count: {0}, got {1}",
                      minElementsInArray, array.size());
    return DebatchedCallOpData{array[0], array[1]};
}

SmallVector<DebatchedCallOpData::ValueType> DebatchedCallOpData::serialize() const {
    return SmallVector<ValueType>({callOpIndex, totalBatchSize});
}

std::string DebatchedCallOpData::to_string() const {
    return "call Index: " + std::to_string(callOpIndex) + ", batch size: " + std::to_string(totalBatchSize);
}

const DebatchedCallOpData& DebatchedCallOpAttributeView::getCallData() const {
    return data;
}

std::optional<DebatchedCallOpAttributeView> DebatchedCallOpAttributeView::extract(mlir::func::CallOp callOp) {
    if (!callOp->hasAttr(DebatchedCallOpAttributeView::name())) {
        return {};
    }
    auto attr = callOp->getAttr(DebatchedCallOpAttributeView::name()).dyn_cast_or_null<mlir::ArrayAttr>();
    VPUX_THROW_UNLESS(attr != nullptr, "Unexpected type for \"{0}\", only \"mlir::ArrayAttr\" supported",
                      DebatchedCallOpAttributeView::name());
    return DebatchedCallOpAttributeView(
            DebatchedCallOpData::deserialize(parseIntArrayAttr<DebatchedCallOpData::ValueType>(attr)));
}

void DebatchedCallOpAttributeView::injectImpl(mlir::func::CallOp callOp) const {
    auto serializedArray = data.serialize();

    auto debatchedAttr = getIntArrayAttr(callOp->getContext(), serializedArray);
    VPUX_THROW_UNLESS(debatchedAttr != nullptr, "Cannot create 'DebatchedCallOpAttributeView' attribute \"{0}\"",
                      DebatchedCallOpAttributeView::name());
    callOp->setAttr(DebatchedCallOpAttributeView::name(), debatchedAttr);
}

bool DebatchedCallOpAttributeView::hasAvailableTilesAttr(mlir::func::CallOp callOp) {
    return callOp->hasAttr(DebatchedCallOpAttributeView::availableTilesAttrName());
}

void DebatchedCallOpAttributeView::setAvailableTilesAttr(mlir::func::CallOp callOp,
                                                         DebatchedCallOpData::ValueType val) {
    VPUX_THROW_UNLESS(!hasAvailableTilesAttr(callOp),
                      "Detected existing 'DebatchedCallOpAttributeView' attribute \"{0}\", cannot create new attribute",
                      DebatchedCallOpAttributeView::availableTilesAttrName());

    auto newAttr = getIntAttr(callOp->getContext(), val);

    VPUX_THROW_UNLESS(newAttr != nullptr, "Failed to create new 'DebatchedCallOpAttributeView' attribute \"{0}\"",
                      DebatchedCallOpAttributeView::availableTilesAttrName());

    callOp->setAttr(DebatchedCallOpAttributeView::availableTilesAttrName(), newAttr);
}

void DebatchedCallOpAttributeView::removeAvailableTilesAttr(mlir::func::CallOp callOp) {
    VPUX_THROW_UNLESS(hasAvailableTilesAttr(callOp),
                      "'DebatchedCallOpAttributeView' attribute \"{0}\" not found, cannot remove",
                      DebatchedCallOpAttributeView::availableTilesAttrName());

    callOp->removeAttr(DebatchedCallOpAttributeView::availableTilesAttrName());
}

DebatchedCallOpData::ValueType DebatchedCallOpAttributeView::getAvailableTilesVal(mlir::func::CallOp callOp) {
    VPUX_THROW_UNLESS(hasAvailableTilesAttr(callOp), "'DebatchedCallOpAttributeView' attribute \"{0}\" not found",
                      DebatchedCallOpAttributeView::availableTilesAttrName());
    return static_cast<DebatchedCallOpData::ValueType>(
            callOp->getAttr(DebatchedCallOpAttributeView::availableTilesAttrName())
                    .cast<mlir::IntegerAttr>()
                    .getValue()
                    .getSExtValue());
}
