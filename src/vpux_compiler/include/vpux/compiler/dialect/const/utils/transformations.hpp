//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace Const {
namespace details {

namespace optimization {

using TransformAttrPos = SmallVector<Const::TransformAttrInterface>::iterator;

}  // namespace optimization

/**
 *
 * Fuses consecutive transformations of the same type into a single transformation. For example:
 *   SubView + SubView ---> SubView
 *
 * Parameters:
 *  transformations: list of transformations
 *  currPos: current position of transformation that might be fused with previous one
 *  baseType: original data type
 * Result:
 *  if optimization has been applied: returns the position of the new transformation,
 *              which is the result of a combination of two consecutive transformations and true
 *  otherwise: returns the current position and false
 */
std::pair<optimization::TransformAttrPos, bool> fuseConsecutiveTransformations(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos,
        NDTypeInterface baseType);

/**
 *
 * Fold transformation that does not affect either the type or the data
 *   e.g. remove NCHW -> Reorder -> NCHW
 *
 * Parameters:
 *  transformations: list of transformations
 *  currPos: current position of transformation that might be folded
 *  baseType: original data type
 *
 * Result:
 *  if optimization has been applied: returns the position of the new previous transformation and true
 *  otherwise: returns the current position and false
 */
std::pair<optimization::TransformAttrPos, bool> foldTransformation(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos,
        NDTypeInterface baseType);

/*
 * Check compatible transformations that are placed before SubView and swaps them. Transformations are considered
 * compatible if they perform element-wise computation, only change the metadata of the constant or the information
 * of the transformation can be reconstructed when moving SubView before. For example:
 *     Add + SubView => SubView + Add
 *
 * The benefit of this change is that less computation and memory are necessary when folding constants.
 *
 * Parameters:
 *  transformations: list of transformations
 *  currPos: current position of SubView that might be swapped with previous one
 *  baseType: original data type
 * Result:
 *  if optimization has been applied: returns the new position of the SubView and true;
 *                                      end position means that the transformation was folded
 *  otherwise returns the current position and false
 */

std::pair<optimization::TransformAttrPos, bool> moveSubViewBefore(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos,
        NDTypeInterface baseType);

/*
 * Check compatible transformations that are placed before Reshape and swaps them. Although the Reshape transformation
 * does not do any computation, moving it before other transformations allows the possibility for other optimizations to
 * be done. For example, in the following pattern, the Reshape is moved before SubView:
 *     Add + Reshape + SubView => Reshape + Add + SubView
 * This allows the possibility of also moving SubView before Add, so that Add only computes the relevant slice of data:
 *     Add + Reshape + SubView => Reshape + SubView + Add
 *
 * Parameters:
 *  transformations: list of transformations
 *  currPos: current position of Reshape that might be swapped with previous one
 *  baseType: original data type
 * Result:
 *  if optimization has been applied: returns the new position of the Reshape and true;
 *                                      end position means that the transformation was folded
 *  otherwise returns the current position and false
 */

std::pair<optimization::TransformAttrPos, bool> moveReshapeBefore(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos,
        NDTypeInterface baseType);

//
// memPermuteTransformation
//

vpux::Const::Content memPermuteTransformation(vpux::Const::Content& input, vpux::NDTypeInterface outType,
                                              mlir::AffineMap memPerm);

/**
 *
 * Move applicable transformations inside the Fuse transformation, e.g.:
 *   Fuse {weights_table = Y} + RelocateWeightsTable ---> Fuse {weights_table = Y
 * [RelocateWeightsTable]}
 *
 * Parameters:
 *  transformations: list of transformations
 *  currPos: current position of transformation that might be moved into fuse
 * Result:
 *  if optimization has been applied: returns the position of the new transformation,
 *              which is the result of a combination of two consecutive transformations and true
 *  otherwise: returns the current position and false
 */
std::pair<optimization::TransformAttrPos, bool> moveTransformationIntoFuse(
        SmallVector<Const::TransformAttrInterface>& transformations, optimization::TransformAttrPos& currPos);

/** @brief Returns a shift of value range between two quantized types.

    Returns a shift in value range such that for [x0; y0) -> [x1, y1)
    transformation, value == (y1 - y0) is returned. This is primarily used in
    cases when one wants to convert between signed and unsigned quantized types.

    @note At present, only supports single-zero-point integer-storage types.
*/
int64_t getValueRangeOffset(mlir::quant::QuantizedType inType, mlir::quant::QuantizedType outType);

}  // namespace details
}  // namespace Const
}  // namespace vpux
