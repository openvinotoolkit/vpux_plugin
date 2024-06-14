//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <openvino/op/convolution.hpp>

using namespace vpux;

mlir::LogicalResult vpux::VPU::GroupConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GroupConvolutionOpAdaptor conv(operands, attrs);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    auto inShape = to_small_vector(conv.getInput().getType().cast<vpux::NDTypeInterface>().getShape().raw());
    const auto inType = conv.getInput().getType().cast<vpux::NDTypeInterface>();
    auto filterShape = to_small_vector(conv.getFilter().getType().cast<vpux::NDTypeInterface>().getShape().raw());

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.getDilations());

    int64_t groups = 0;
    if (conv.getGroups().value_or(0) != 0) {
        if (filterShape.size() != inShape.size()) {
            return errorAt(loc, "Input size '{0}' does not match filter size '{1}'. (groups != 0)", inShape.size(),
                           filterShape.size());
        }

        groups = conv.getGroups().value();
    } else {
        if (filterShape.size() != inShape.size() + 1) {
            return errorAt(loc, "Input size '{0}' does not match filter size '{1}'. (groups == 0)", inShape.size() + 1,
                           filterShape.size());
        }

        groups = filterShape[0];

        // we need to adjust filters_shape to reuse helpers for normal convolution
        filterShape[1] *= groups;
        filterShape.erase(filterShape.begin());
    }

    inShape[1] /= groups;

    const auto op = ov::op::v1::Convolution(
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape(inShape.begin(), inShape.end())),
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                    ov::Shape(filterShape.begin(), filterShape.end())),
            ov::Strides(windowStrides.begin(), windowStrides.end()),
            ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ov::Strides(windowDilations.begin(), windowDilations.end()));

    const auto& outputShape = op.get_output_partial_shape(0);
    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));

    const auto outType = inType.changeShape(Shape(shapeI64));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

InputTiling vpux::VPU::GroupConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto origInputShape = getShape(getInput());
    const auto origFilterShape = getShape(getFilter());
    const auto origBiasShape = getBias() != nullptr ? getShape(getBias()) : ShapeRef();
    const auto origPadding = PadInfo(getPadsBegin(), getPadsEnd());
    const auto origGroups = getGroups().value_or(1);

    return backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, getStrides(), origPadding,
                                  origGroups);
}

//
// fitIntoCMX
//

bool vpux::VPU::GroupConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                               vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

bool vpux::VPU::GroupConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                               vpux::NDTypeInterface output, Byte reservedMem) {
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize()};

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

void vpux::VPU::GroupConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    const auto& inputTiles = inputTiling.tiles;
    VPUX_THROW_UNLESS(inputTiles.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                      inputTiles.size());

    IE::adjustPaddings(this, inputTiling);

    const auto& inputTile = inputTiles[0];
    const auto& filterTile = inputTiles[1];
    const auto groups = inputTile.shape[Dims4D::Act::C] / filterTile.shape[Dims4D::Filter::IC];
    const auto groupsNewAttr = getIntAttr(getContext(), groups);

    setGroupsAttr(groupsNewAttr);
}

mlir::FailureOr<OutputTiling> vpux::VPU::GroupConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
