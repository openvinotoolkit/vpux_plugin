//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

namespace vpux {
namespace IE {

mlir::ArrayAttr getNewConcatOffsetsParameters(mlir::ArrayAttr oldOffsets, mlir::ArrayAttr dimsMappingAttr,
                                              mlir::OperandRange oldInputs, ArrayRef<vpux::ShapeRef> newInputShapes,
                                              ShapeRef reshapeShape, mlir::DenseSet<int64_t> modifiedAxes) {
    const auto oldOffsetsList = parseIntArrayOfArrayAttr<int64_t>(oldOffsets);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(dimsMappingAttr);

    SmallVector<SmallVector<int64_t>> newOffsetsList;
    newOffsetsList.reserve(oldOffsetsList.size());

    for (auto inputIndex : irange(oldOffsetsList.size())) {
        const auto oldInputShape = getShape(oldInputs[inputIndex]).raw();
        const auto newInputShape = newInputShapes[inputIndex].raw();
        const auto oldOffset = oldOffsetsList[inputIndex];

        SmallVector<int64_t> newOffset(newInputShape.size(), 0);
        for (const auto oldConcatDim : modifiedAxes) {
            for (const auto& dim : dimMapping[oldConcatDim]) {
                // Condition "reshapeShape[Dim(dim)] != 1" is added to handle the following case:
                // Concat on a dimension and then unsqueeze that dimension, e.g.:
                // 2 x ([1, 3]) -> Concat -> ([2, 3]) -> AffineReshape: dimMapping={[0, 1, 2], [3]} -> ([1, 2, 1, 3])
                if (oldInputShape[oldConcatDim] == newInputShape[dim] && reshapeShape[Dim(dim)] != 1) {
                    newOffset[dim] = oldOffset[oldConcatDim];
                    break;
                }
            }
        }

        newOffsetsList.push_back(newOffset);
    }

    // Make sure that there is at least one offset is set
    bool isOffsetSet = std::any_of(newOffsetsList.begin(), newOffsetsList.end(), [](ArrayRef<int64_t> v) {
        return std::any_of(v.begin(), v.end(), [](int64_t i) {
            return i != 0;
        });
    });
    VPUX_THROW_UNLESS(isOffsetSet == true, "No valid concat offset was found during ConcatReshapeConcat rewritten");

    return getIntArrayOfArray(dimsMappingAttr.getContext(), ArrayRef(newOffsetsList));
}

mlir::DenseSet<int64_t> getConcatModifiedAxis(IE::ConcatOp origOp) {
    mlir::DenseSet<int64_t> modifiedAxes;
    const auto offsets = parseIntArrayOfArrayAttr<int64_t>(origOp.getStaticOffsetsAttr());

    for (size_t i = 0; i < offsets.size(); i++) {
        for (size_t j = 0; j < offsets[i].size(); ++j) {
            if (offsets[i][j] != 0) {
                modifiedAxes.insert(j);
            }
        }
    }

    return modifiedAxes;
}

SmallVector<int64_t> calculateInputShapeAfterSwitchConcatAndAffineReshape(mlir::Value input, IE::ConcatOp concatOp,
                                                                          IE::AffineReshapeOp reshapeOp) {
    const auto affineOutShape = getShape(reshapeOp.getOutput());
    const auto modifiedAxes = getConcatModifiedAxis(concatOp);

    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMapping());
    SmallVector<int64_t> newShapeVec(affineOutShape.size());
    for (size_t dimIdx = 0; dimIdx < affineOutShape.size(); dimIdx++) {
        auto axisIt = llvm::find_if(modifiedAxes, [&](int64_t modifiedAxis) {
            for (auto& mappedIdx : dimMapping[modifiedAxis]) {
                if (affineOutShape[Dim(mappedIdx)] == 1) {
                    continue;
                } else {
                    return dimIdx == checked_cast<size_t>(mappedIdx);
                }
            }
            return false;
        });
        if (axisIt != modifiedAxes.end() && affineOutShape[Dim(dimIdx)] != 1) {
            newShapeVec[dimIdx] = getShape(input)[Dim(*axisIt)];
        } else if (affineOutShape[Dim(dimIdx)] == 1) {
            newShapeVec[dimIdx] = 1;
        } else {
            newShapeVec[dimIdx] = affineOutShape[Dim(dimIdx)];
        }
    }
    return newShapeVec;
}

mlir::Value createPaddingConstForConcat(ArrayRef<int64_t> constShape, mlir::Location loc,
                                        vpux::NDTypeInterface inputType, double padValue,
                                        mlir::PatternRewriter& rewriter) {
    const auto origElemType = inputType.getElementType();
    const auto padDataStorageType =
            mlir::RankedTensorType::get(constShape, mlir::Float32Type::get(rewriter.getContext()));
    const auto padDataStorage = Const::createConstContent(padDataStorageType, ArrayRef(static_cast<float>(padValue)));

    const auto padDataType = mlir::RankedTensorType::get(constShape, origElemType);
    auto padDataAttr =
            Const::ContentAttr::get(padDataStorage, Const::ContentSetup(padDataStorageType).castElemType(origElemType));

    auto constant = rewriter.create<Const::DeclareOp>(loc, padDataType, std::move(padDataAttr));

    const auto dataOrder = inputType.getDimsOrder();
    const auto orderMap = dataOrder.toAffineMap(rewriter.getContext());
    return rewriter.createOrFold<IE::ReorderOp>(loc, constant.getOutput(), orderMap);
}

const mlir::ArrayAttr inferOffsetsAttrWithAxis(IE::ConcatOp origOp, int64_t& axis) {
    auto rank = origOp.getOutput().getType().cast<vpux::NDTypeInterface>().getRank();

    SmallVector<SmallVector<int64_t>> finalOffsets;
    finalOffsets.reserve(origOp.getInputs().size());
    finalOffsets.push_back(SmallVector<int64_t>(rank, 0));
    if (axis < 0) {
        axis += rank;
    }

    auto inputs = llvm::drop_end(origOp.getInputs(), 1);
    for (auto input : llvm::enumerate(inputs)) {
        auto inputShape = getShape(input.value());
        auto offsets = SmallVector<int64_t>(rank, 0);
        offsets[axis] = inputShape[Dim(axis)] + finalOffsets.back()[axis];
        finalOffsets.push_back(offsets);
    }

    return getIntArrayOfArray(origOp.getContext(), finalOffsets);
}

std::optional<vpux::Dim> getConcatAxis(IE::ConcatOp concatOp) {
    if (concatOp.getPerAxisAttr()) {
        if (concatOp.getPerAxisAttr().getStride()) {
            return std::nullopt;
        }
        return Dim(concatOp.getPerAxisAttr().getAxis().getValue().getSExtValue());
    }

    const auto concatAxes =
            vpux::IE::getDiffInOutSizeDims(getShape(concatOp.getOperands()[0]), getShape(concatOp.getResult()));
    if (concatAxes.empty() || concatAxes.size() != 1) {
        return std::nullopt;
    }

    const auto concatAxis = concatAxes.front();
    // Should to ensure there is no data overlapped
    VPUX_THROW_UNLESS(concatOp.getStaticOffsetsAttr() != nullptr, "Cannot get StaticOffsetsAttr");
    const auto allOffsets = concatOp.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();

    int64_t accumulator = 0;
    for (const auto& p : zip(concatOp.getInputs(), allOffsets)) {
        const auto inputShape = getShape(std::get<0>(p));
        const auto offsets = parseIntArrayAttr<int64_t>(std::get<1>(p));

        if (accumulator != offsets[concatAxis.ind()]) {
            return std::nullopt;
        }
        accumulator += inputShape[concatAxis];
    }

    if (accumulator != getShape(concatOp.getResult())[concatAxis]) {
        return std::nullopt;
    }

    return concatAxis;
}

}  // namespace IE
}  // namespace vpux
