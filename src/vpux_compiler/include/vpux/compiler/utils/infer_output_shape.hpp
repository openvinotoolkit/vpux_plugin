//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/array_ref.hpp"

namespace vpux {

struct ShapeInfo {
    SmallVector<int64_t> shape;
    SmallVector<int64_t> bounds;
};

/**
 * @brief                        Infers the output shape for a StridedSlice operation
 *                               with the given parameters
 * @param inDataShape:           The shape of the input data
 * @param begins:                1D tensor with begin indexes for input blob slicing. Use for constant begins
 * @param ends:                  1D tensor with end indexes for input blob slicing. Use for constant ends
 * @param strides:               1D tensor of the slicing strides. Use for constant strides
 * @param beginsSize:            Shape of begin indexes for input blob slicing. Use for non-constant begins
 * @param endsSize:              Shape of end indexes for input blob slicing. Use for non-constant ends
 * @param stridesSize:           Shape of the slicing strides. Use for non-constant strides
 * @param begin_mask:            Bitmask corresponding to the dimensions of the begin input
 * @param end_mask:              Bitmask corresponding to the dimensions of the end input
 * @param new_axis_mask:         Bitmask which specifies the insertion of 1 dimension
 * @param shrink_axis_mask:      Bitmask which specifies the deletion of 1 dimension
 * @param ellipsis_mask:         Bitmask which inserts missing dimensions on a position
 *                               of a non-zero bit
 * @return                       The output shape info as ShapeInfo
 */
ShapeInfo inferStridedSliceOutputShape(ArrayRef<int64_t> inDataShape, ArrayRef<int64_t> begins, ArrayRef<int64_t> ends,
                                       ArrayRef<int64_t> strides, ArrayRef<int64_t> beginsShape,
                                       ArrayRef<int64_t> endsShape, ArrayRef<int64_t> stridesShape,
                                       ArrayRef<int64_t> beginMask, ArrayRef<int64_t> endMask,
                                       ArrayRef<int64_t> newAxisMask, ArrayRef<int64_t> shrinkAxisMask,
                                       ArrayRef<int64_t> ellipsisMask);

/**
 * @brief                        Infers the output shape for a MaxPool operation
 *                               with the given parameters
 * @param inDataShape:           The shape of the input data
 * @param windowStrides:         The strides
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowShape:           The kernel window
 * @param roundingType:          Whether to use ceiling or floor rounding type while
 *                               computing output shape
 * @return                       The output shape as SmallVector
 */
SmallVector<int64_t> inferMaxPoolOutputShape(ArrayRef<int64_t> inDataShape, ArrayRef<int64_t> windowStrides,
                                             ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove,
                                             ArrayRef<int64_t> windowShape,
                                             IE::RoundingType roundingType = IE::RoundingType::FLOOR);

/**
 * @brief                        Infers the output shape for a MaxPool8 operation
 *                               with the given parameters
 * @param inDataShape:           The shape of the input data
 * @param windowStrides:         The strides
 * @param windowDilations:       The dilations of the pooling filter
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowShape:           The kernel window
 * @param roundingType:          Whether to use ceiling or floor rounding type while
 *                               computing output shape
 * @return                       The output shape as SmallVector
 */
SmallVector<int64_t> inferMaxPool8OutputShape(ArrayRef<int64_t> inDataShape, ArrayRef<int64_t> windowStrides,
                                              ArrayRef<int64_t> windowDilations, ArrayRef<int64_t> dataPaddingBelow,
                                              ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowShape,
                                              IE::RoundingType roundingType = IE::RoundingType::FLOOR);

/**
 * @brief                        Infers the output shape for a AvgPool operation
 *                               with the given parameters
 * @param inDataShape:           The shape of the input data
 * @param windowStrides:         The strides
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowShape:           The kernel window
 * @param roundingType:          Whether to use ceiling or floor rounding type while
 *                               computing output shape
 * @return                       The output shape as SmallVector
 */
SmallVector<int64_t> inferAvgPoolOutputShape(ArrayRef<int64_t> inDataShape, ArrayRef<int64_t> windowStrides,
                                             ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove,
                                             ArrayRef<int64_t> windowShape,
                                             IE::RoundingType roundingType = IE::RoundingType::FLOOR);

/**
 * @brief                        Infers the output shape for a ConvolutionBackpropData operation
 *                               with the given parameters
 * @param inputShape:            The shape of the input data
 * @param filterShape:           The shape of the filter
 * @param windowStrides:         The strides
 * @param dataPaddingBelow:      Builds the beginning of padding shape
 * @param dataPaddingAbove:      Builds the end of padding shape
 * @param windowDilations:       The dilations
 * @param outputPadding:         The output padding
 *
 * @return                       The output shape as SmallVector
 */
SmallVector<int64_t> inferConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                  ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> dataPaddingBelow,
                                                  ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
                                                  ArrayRef<int64_t> outputPadding);

SmallVector<int64_t> inferGroupConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                       ArrayRef<int64_t> windowStrides,
                                                       ArrayRef<int64_t> dataPaddingBelow,
                                                       ArrayRef<int64_t> dataPaddingAbove,
                                                       ArrayRef<int64_t> windowDilations,
                                                       ArrayRef<int64_t> outputPadding);

SmallVector<int64_t> inferTransposedConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                            ArrayRef<int64_t> windowStrides,
                                                            ArrayRef<int64_t> dataPaddingBelow,
                                                            ArrayRef<int64_t> dataPaddingAbove,
                                                            ArrayRef<int64_t> windowDilations,
                                                            ArrayRef<int64_t> outputPadding);

SmallVector<int64_t> inferTransposedGroupConvBackpropOutputShape(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape, ArrayRef<int64_t> windowStrides,
        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
        ArrayRef<int64_t> outputPadding);

}  // namespace vpux
