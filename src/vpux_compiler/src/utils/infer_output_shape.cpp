//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/infer_output_shape.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/range.hpp"

#include <openvino/op/avg_pool.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/max_pool.hpp>
#include <openvino/op/strided_slice.hpp>

#include <mlir/IR/BuiltinTypes.h>

#include <cstddef>
#include <optional>

using namespace vpux;

ShapeInfo vpux::inferStridedSliceOutputShape(ArrayRef<int64_t> inDataShape, ArrayRef<int64_t> begins,
                                             ArrayRef<int64_t> ends, ArrayRef<int64_t> strides,
                                             ArrayRef<int64_t> beginsShape, ArrayRef<int64_t> endsShape,
                                             ArrayRef<int64_t> stridesShape, ArrayRef<int64_t> beginMask,
                                             ArrayRef<int64_t> endMask, ArrayRef<int64_t> newAxisMask,
                                             ArrayRef<int64_t> shrinkAxisMask, ArrayRef<int64_t> ellipsisMask) {
    auto extractPaddedMask = [](ArrayRef<int64_t> mask, std::size_t expandSize) -> std::vector<int64_t> {
        auto maskVector = to_std_vector(mask);
        if (maskVector.size() < expandSize) {
            maskVector.insert(maskVector.end(), expandSize - maskVector.size(), 0);
        }
        return maskVector;
    };

    SmallVector<ArrayRef<int64_t>> opMasks{beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask};
    const auto padSize = std::max_element(opMasks.begin(), opMasks.end(), [](auto const& lhs, auto const& rhs) {
                             return lhs.size() < rhs.size();
                         })->size();

    const auto paddedBeginMask = extractPaddedMask(opMasks[0], padSize);
    const auto paddedEndMask = extractPaddedMask(opMasks[1], padSize);
    const auto paddedNewAxisMask = extractPaddedMask(opMasks[2], padSize);
    const auto paddedShrinkAxisMask = extractPaddedMask(opMasks[3], padSize);
    const auto paddedEllipsisMask = extractPaddedMask(opMasks[4], padSize);

    ov::Output<ov::Node> ovBegins = {};
    if (!begins.empty()) {
        const auto beginsVec = to_std_vector(begins);
        ovBegins = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({beginsVec.size()}), beginsVec);
    } else if (!beginsShape.empty()) {
        ovBegins = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                           ov::Shape(beginsShape.begin(), beginsShape.end()));
    }

    ov::Output<ov::Node> ovEnds = {};
    if (!ends.empty()) {
        const auto endsVec = to_std_vector(ends);
        ovEnds = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({endsVec.size()}), endsVec);
    } else if (!endsShape.empty()) {
        ovEnds = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                         ov::Shape(endsShape.begin(), endsShape.end()));
    }

    ov::Output<ov::Node> ovStrides = {};
    if (!strides.empty()) {
        const auto stridesVec = to_std_vector(strides);
        ovStrides =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({stridesVec.size()}), stridesVec);
    } else if (!stridesShape.empty()) {
        ovStrides = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                            ov::Shape(stridesShape.begin(), stridesShape.end()));
    }

    const auto ovOp =
            ov::op::v1::StridedSlice(std::make_shared<ov::op::v0::Parameter>(
                                             ov::element::f16, ov::Shape(inDataShape.begin(), inDataShape.end())),
                                     ovBegins, ovEnds, ovStrides, paddedBeginMask, paddedEndMask, paddedNewAxisMask,
                                     paddedShrinkAxisMask, paddedEllipsisMask);

    const auto& outputShape = ovOp.get_output_partial_shape(0);

    auto resultShape = to_small_vector(outputShape | transformed([](const ov::Dimension& val) {
                                           if (val.is_static()) {
                                               return checked_cast<int64_t>(val.get_length());
                                           }
                                           return checked_cast<int64_t>(mlir::ShapedType::kDynamic);
                                       }));

    if (outputShape.is_dynamic()) {
        const auto getUpperBound = [](const ov::Dimension& val) -> int64_t {
            if (val.is_static()) {
                return checked_cast<int64_t>(val.get_length());
            }
            return checked_cast<int64_t>(val.get_max_length());
        };
        auto resultBounds = to_small_vector(outputShape | transformed(getUpperBound));
        return {std::move(resultShape), std::move(resultBounds)};
    }

    return {std::move(resultShape), {}};
}

SmallVector<int64_t> vpux::inferAvgPoolOutputShape(ArrayRef<int64_t> inDataShape, ArrayRef<int64_t> windowStrides,
                                                   ArrayRef<int64_t> dataPaddingBelow,
                                                   ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowShape,
                                                   IE::RoundingType roundingType) {
    const auto padsBegin = ov::Shape(dataPaddingBelow.begin(), dataPaddingBelow.end());
    const auto padsEnd = ov::Shape(dataPaddingAbove.begin(), dataPaddingAbove.end());
    const auto ovOp = ov::op::v1::AvgPool(std::make_shared<ov::op::v0::Parameter>(
                                                  ov::element::i32, ov::Shape(inDataShape.begin(), inDataShape.end())),
                                          ov::Strides(windowStrides.begin(), windowStrides.end()), padsBegin, padsEnd,
                                          ov::Shape(windowShape.begin(), windowShape.end()), false,
                                          static_cast<ov::op::RoundingType>(roundingType), ov::op::PadType::EXPLICIT);
    return to_small_vector(ovOp.get_output_shape(0) | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

SmallVector<int64_t> vpux::inferMaxPoolOutputShape(ArrayRef<int64_t> inDataShape, ArrayRef<int64_t> windowStrides,
                                                   ArrayRef<int64_t> dataPaddingBelow,
                                                   ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowShape,
                                                   IE::RoundingType roundingType) {
    const auto padsBegin = ov::Shape(dataPaddingBelow.begin(), dataPaddingBelow.end());
    const auto padsEnd = ov::Shape(dataPaddingAbove.begin(), dataPaddingAbove.end());
    const auto ovOp = ov::op::v1::MaxPool(std::make_shared<ov::op::v0::Parameter>(
                                                  ov::element::i32, ov::Shape(inDataShape.begin(), inDataShape.end())),
                                          ov::Strides(windowStrides.begin(), windowStrides.end()), padsBegin, padsEnd,
                                          ov::Shape(windowShape.begin(), windowShape.end()),
                                          static_cast<ov::op::RoundingType>(roundingType), ov::op::PadType::EXPLICIT);
    return to_small_vector(ovOp.get_output_shape(0) | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

SmallVector<int64_t> vpux::inferMaxPool8OutputShape(ArrayRef<int64_t> inDataShape, ArrayRef<int64_t> windowStrides,
                                                    ArrayRef<int64_t> windowDilations,
                                                    ArrayRef<int64_t> dataPaddingBelow,
                                                    ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowShape,
                                                    IE::RoundingType roundingType) {
    const auto padsBegin = ov::Shape(dataPaddingBelow.begin(), dataPaddingBelow.end());
    const auto padsEnd = ov::Shape(dataPaddingAbove.begin(), dataPaddingAbove.end());
    const auto ovOp = ov::op::v8::MaxPool(std::make_shared<ov::op::v0::Parameter>(
                                                  ov::element::i32, ov::Shape(inDataShape.begin(), inDataShape.end())),
                                          ov::Strides(windowStrides.begin(), windowStrides.end()),
                                          ov::Strides(windowDilations.begin(), windowDilations.end()), padsBegin,
                                          padsEnd, ov::Shape(windowShape.begin(), windowShape.end()),
                                          static_cast<ov::op::RoundingType>(roundingType), ov::op::PadType::EXPLICIT,
                                          ov::element::i64, 0);
    return to_small_vector(ovOp.get_output_shape(0) | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

ov::PartialShape getConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                            ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> dataPaddingBelow,
                                            ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
                                            ArrayRef<int64_t> outputPadding) {
    return ov::op::v1::ConvolutionBackpropData(
                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape(inputShape.begin(), inputShape.end())),
                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape(filterShape.begin(), filterShape.end())),
                   ov::Strides(windowStrides.begin(), windowStrides.end()),
                   ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                   ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                   ov::Strides(windowDilations.begin(), windowDilations.end()), ov::op::PadType::EXPLICIT,
                   ov::CoordinateDiff(outputPadding.begin(), outputPadding.end()))
            .get_output_partial_shape(0);
}

ov::PartialShape getGroupConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                 ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> dataPaddingBelow,
                                                 ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
                                                 ArrayRef<int64_t> outputPadding) {
    return ov::op::v1::GroupConvolutionBackpropData(
                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape(inputShape.begin(), inputShape.end())),
                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape(filterShape.begin(), filterShape.end())),
                   ov::Strides(windowStrides.begin(), windowStrides.end()),
                   ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                   ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                   ov::Strides(windowDilations.begin(), windowDilations.end()), ov::op::PadType::EXPLICIT,
                   ov::CoordinateDiff(outputPadding.begin(), outputPadding.end()))
            .get_output_partial_shape(0);
}

SmallVector<int64_t> vpux::inferConvBackpropOutputShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape,
                                                        ArrayRef<int64_t> windowStrides,
                                                        ArrayRef<int64_t> dataPaddingBelow,
                                                        ArrayRef<int64_t> dataPaddingAbove,
                                                        ArrayRef<int64_t> windowDilations,
                                                        ArrayRef<int64_t> outputPadding) {
    auto backpropFilter = to_std_vector(filterShape);
    backpropFilter[Dims4D::Filter::OC.ind()] = inputShape[Dims4D::Act::C.ind()];
    auto ovOpShape = getConvBackpropOutputShape(inputShape, backpropFilter, windowStrides, dataPaddingBelow,
                                                dataPaddingAbove, windowDilations, outputPadding)
                             .get_shape();

    ovOpShape[Dims4D::Act::N.ind()] = inputShape[Dims4D::Act::N.ind()];
    ovOpShape[Dims4D::Act::C.ind()] = filterShape[Dims4D::Filter::IC.ind()];

    return to_small_vector(ovOpShape | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

SmallVector<int64_t> vpux::inferGroupConvBackpropOutputShape(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape, ArrayRef<int64_t> windowStrides,
        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
        ArrayRef<int64_t> outputPadding) {
    auto groups = filterShape[0];
    auto IC = filterShape[1];
    auto OC = filterShape[2];

    auto backpropIn = to_std_vector(inputShape);
    backpropIn[Dims4D::Act::C.ind()] = groups * IC;
    auto ovOpShape = getGroupConvBackpropOutputShape(backpropIn, filterShape, windowStrides, dataPaddingBelow,
                                                     dataPaddingAbove, windowDilations, outputPadding)
                             .get_shape();

    ovOpShape[Dims4D::Act::N.ind()] = inputShape[Dims4D::Act::N.ind()];
    ovOpShape[Dims4D::Act::C.ind()] = groups * OC;

    return to_small_vector(ovOpShape | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

SmallVector<int64_t> vpux::inferTransposedConvBackpropOutputShape(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape, ArrayRef<int64_t> windowStrides,
        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
        ArrayRef<int64_t> outputPadding) {
    auto backpropFilter = to_std_vector(filterShape);
    backpropFilter[Dims4D::Filter::OC.ind()] = inputShape[Dims4D::Act::C.ind()];
    auto ovOpShape = getConvBackpropOutputShape(inputShape, backpropFilter, windowStrides, dataPaddingBelow,
                                                dataPaddingAbove, windowDilations, outputPadding)
                             .get_shape();

    ovOpShape[Dims4D::Act::N.ind()] = inputShape[Dims4D::Act::N.ind()];
    ovOpShape[Dims4D::Act::C.ind()] = filterShape[Dims4D::Filter::OC.ind()];

    return to_small_vector(ovOpShape | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}

SmallVector<int64_t> vpux::inferTransposedGroupConvBackpropOutputShape(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> filterShape, ArrayRef<int64_t> windowStrides,
        ArrayRef<int64_t> dataPaddingBelow, ArrayRef<int64_t> dataPaddingAbove, ArrayRef<int64_t> windowDilations,
        ArrayRef<int64_t> outputPadding) {
    // For 2D GroupTransposedConvolution:
    // input tensor layout is [N, C_IN * GROUPS, H, W]
    // kernel tensor layout is [GROUPS, C_OUT, C_IN, kH, kW]
    auto groups = filterShape[IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX];
    auto OC = filterShape[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX];
    auto groupedChannels = groups * OC;

    auto transposedBackpropIn = to_std_vector(inputShape);
    transposedBackpropIn[Dims4D::Act::C.ind()] = groupedChannels;
    auto ovOpShape = getGroupConvBackpropOutputShape(transposedBackpropIn, filterShape, windowStrides, dataPaddingBelow,
                                                     dataPaddingAbove, windowDilations, outputPadding)
                             .get_shape();

    ovOpShape[Dims4D::Act::N.ind()] = inputShape[Dims4D::Act::N.ind()];
    ovOpShape[Dims4D::Act::C.ind()] = groupedChannels;

    return to_small_vector(ovOpShape | transformed([](size_t val) {
                               return checked_cast<int64_t>(val);
                           }));
}
