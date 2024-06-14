//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
using namespace IE;

namespace {

mlir::FailureOr<SmallVector<int64_t>> parseIntVector(mlir::Location loc, const mlir::Value value) {
    if (value == nullptr) {
        return errorAt(loc, "Parameter were not provided");
    }

    auto valueConst = value.getDefiningOp<Const::DeclareOp>();
    if (valueConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for interpolate attribute");
    }

    const auto valueContent = valueConst.getContent();
    return to_small_vector(valueContent.getValues<int64_t>());
}

RollShiftAndAxes adjustSpatialShiftAxes(ArrayRef<int64_t> shiftRef, ArrayRef<int64_t> axesRef) {
    // Adjust Axes to the spatial dimension [H,W]
    auto shift = to_small_vector(shiftRef);
    auto axes = to_small_vector(axesRef);
    if (axes.size() == 1) {
        // Axes:  H -> H,W
        // or
        // Axes:  W -> H,W
        if (axes[0] == Dims4D::Act::H.ind()) {
            shift = SmallVector<int64_t>{shift[0], 0};
            axes = SmallVector<int64_t>{Dims4D::Act::H.ind(), Dims4D::Act::W.ind()};
        } else if (axes[0] == Dims4D::Act::W.ind()) {
            shift = SmallVector<int64_t>{0, shift[0]};
            axes = SmallVector<int64_t>{Dims4D::Act::H.ind(), Dims4D::Act::W.ind()};
        }
    } else if (axes.size() == 2) {
        // Axes: [W,H] -> [H,W]
        if (axes[0] == Dims4D::Act::W.ind() && axes[1] == Dims4D::Act::H.ind()) {
            shift = SmallVector<int64_t>{shift[1], shift[0]};
            axes = SmallVector<int64_t>{Dims4D::Act::H.ind(), Dims4D::Act::W.ind()};
        }
    }

    return RollShiftAndAxes(std::move(shift), std::move(axes));
}

}  // namespace

mlir::FailureOr<RollShiftAndAxes> vpux::IE::getShiftAndAxesForRollOp(mlir::Location loc, mlir::Value shiftValue,
                                                                     mlir::Value axesValue, ShapeRef inputShape) {
    // Get shift
    auto shiftsOrFail = parseIntVector(loc, shiftValue);
    if (mlir::failed(shiftsOrFail)) {
        return mlir::failure();
    }

    // Get axes
    auto axesOrFail = parseIntVector(loc, axesValue);
    if (mlir::failed(axesOrFail)) {
        return mlir::failure();
    }

    auto shifts = shiftsOrFail.value();
    auto axes = axesOrFail.value();
    const auto inputShapeRank = inputShape.size();

    // handle the negative shift/axes.
    SmallVector<int64_t> positiveAxes;
    std::transform(axes.begin(), axes.end(), std::back_inserter(positiveAxes), [&](const auto axis) {
        return axis < 0 ? axis + inputShapeRank : axis;
    });

    SmallVector<int64_t> positiveShift;
    if (shifts.size() == 1) {
        shifts = SmallVector<int64_t>(positiveAxes.size(), shifts[0]);
    }
    std::transform(shifts.begin(), shifts.end(), positiveAxes.begin(), std::back_inserter(positiveShift),
                   [&](const auto shift, const auto axis) {
                       return shift < 0 ? shift + inputShape[Dim(axis)] : shift;
                   });

    return adjustSpatialShiftAxes(std::move(positiveShift), std::move(positiveAxes));
}
