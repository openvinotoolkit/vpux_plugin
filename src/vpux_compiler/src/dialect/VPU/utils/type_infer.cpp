//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

namespace vpux {
namespace VPU {

mlir::LogicalResult inferReduceReturnTypes(mlir::Location loc, mlir::Value input, bool keepDims,
                                           SmallVector<int64_t>& axes,
                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto inType = input.getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();

    bool isAllUnique = std::unique(axes.begin(), axes.end()) == axes.end();
    if (!isAllUnique) {
        return errorAt(loc, "Axes values should be unique");
    }

    // Add to outShape the values with indices not found in axes_set.
    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inShape.size(); i++) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            outShape.push_back(inShape[i]);
        } else if (keepDims) {
            outShape.push_back(1);
        }
    }

    // If axes contains all dimensions of input data, a single reduction value is calculated for the entire input tensor
    if (outShape.empty()) {
        outShape = {1};
    }

    const auto newOutputType =
            TypeComponents()
                    .setDimsOrder(keepDims ? inType.getDimsOrder()
                                           : vpux::IE::calculateReducedOutputLayout(inType.getDimsOrder(), axes))
                    .setShape(Shape(outShape));
    const auto outType = inType.changeTypeComponents(newOutputType);

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

void inferPermuteReturnTypes(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                             SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dst_order);
    const auto inType = input.getType().cast<vpux::NDTypeInterface>();

    const auto inShape = getShape(input);
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto outMemShape = applyPerm(inMemShape, mem_perm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);
    const auto outBoundsAttr = permuteBounds(input.getContext(), inType.dyn_cast_or_null<vpux::BoundedTypeInterface>(),
                                             inOrder, outOrder, mem_perm);

    auto outType = inType.changeDimsOrder(outOrder).changeShape(outShape);

    if (auto boundedOutType = outType.dyn_cast<vpux::BoundedTypeInterface>()) {
        outType = boundedOutType.changeBounds(outBoundsAttr);
    }

    inferredReturnTypes.push_back(outType);
}

}  // namespace VPU
}  // namespace vpux
