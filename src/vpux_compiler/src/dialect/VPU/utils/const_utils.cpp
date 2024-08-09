//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include <vpux/utils/core/numeric.hpp>

namespace vpux {
namespace VPU {

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity) {
    const auto elemType = getUInt8Type(builder.getContext());
    const auto fakeSparsityShape = NCESparsity::inferActivationWindowShape(static_cast<int64_t>(fakeSparsity.size()));

    const auto dataStorageType = mlir::RankedTensorType::get(fakeSparsityShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, fakeSparsity);
}

std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                            Const::ContentAttr bias, int64_t OC, vpux::VPU::PPETaskAttr ppeTaskAttr,
                                            VPU::ArchKind _arch, vpux::IE::PostOpAttr postOpAttr,
                                            mlir::FloatAttr constScale) {
    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = VPU::NCESparsity::getWeightPtrStep(weights);
    const auto sparsityPtrStep = 0;

    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;

    return VPU::NCESparsity::getWeightsTable(inElemType, outElemType, weightPtrOffset, weightPtrStep, sparsityPtrOffset,
                                             sparsityPtrStep, _arch, OC, weightsElemType, bias, ppeTaskAttr, postOpAttr,
                                             constScale);
}

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable) {
    const int64_t OC = weightsTable.size() / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;

    const auto elemType = getSInt32Type(builder.getContext());
    const auto weightTableShape = NCESparsity::inferWeightsTableShape(OC);

    const auto dataStorageType = mlir::RankedTensorType::get(weightTableShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, weightsTable);
}

std::optional<SmallVector<int32_t>> createInstructionListTableData(mlir::Value opOutput, vpux::IE::PostOpAttr postOp,
                                                                   VPU::ArchKind _arch) {
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();

    if (postOp == nullptr) {
        return std::nullopt;
    }

    if (_arch == VPU::ArchKind::NPU37XX || _arch == VPU::ArchKind::NPU40XX) {
        return std::nullopt;
    }

    const auto pwlTable = findCustomPWLTable(postOp, outElemType);

    if (!pwlTable.has_value()) {
        return std::nullopt;
    }

    const auto& pwlTableRange = pwlTable.value().range;
    const auto& pwlTableShift = pwlTable.value().shift;
    const auto& pwlTableBias = pwlTable.value().bias;

    return VPU::NCESparsity::getInstructionListTable(pwlTableRange, pwlTableShift, pwlTableBias);
}

mlir::Value createInstructionListTableTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                             const std::optional<SmallVector<int32_t>>& instructionList) {
    if (!instructionList.has_value()) {
        return nullptr;
    }
    const auto instructionListArrayRef = ArrayRef(instructionList.value());
    const auto elemType = getSInt32Type(builder.getContext());
    const auto instructionListTableShape = Shape{1, 1, 1, static_cast<int64_t>(instructionListArrayRef.size())};

    const auto dataStorageType = mlir::RankedTensorType::get(instructionListTableShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, instructionListArrayRef);
}

namespace {

mlir::Value getAlignedConstWeights(mlir::OpBuilder& builder, mlir::Location loc, Const::DeclareOp weightsConst,
                                   Shape flatWeightShape, int64_t padding) {
    auto weightsContentAttr = weightsConst.getContentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, padding, 0, 0});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto origFilterType = weightsConst.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto outAllocType = mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType())
                                      .cast<vpux::NDTypeInterface>();
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.getOutput();
}

Const::ContentAttr buildPadData(const mlir::Type type, ArrayRef<int64_t> shape) {
    VPUX_THROW_UNLESS(shape.size() == 4, "Unsupported shape size {0}", shape.size());
    const auto OC = shape[Dims4D::Filter::OC.ind()];
    const auto alignment = shape[Dims4D::Filter::IC.ind()];

    if (const auto quantizedType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto padType = mlir::RankedTensorType::get(shape, normalizeQuantStorageType(quantizedType));
        std::vector<int64_t> padValues;

        if (const auto uniformType = quantizedType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
            padValues.assign(OC * alignment, uniformType.getZeroPoint());
        } else if (const auto perAxisType = quantizedType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto zeroPoints = perAxisType.getZeroPoints();
            VPUX_THROW_UNLESS(checked_cast<size_t>(OC) == zeroPoints.size(),
                              "Number of zero points {0} and channels {1} don't match", zeroPoints.size(), OC);

            // assuming all zero points are equal to broadcast
            VPUX_THROW_UNLESS(
                    zeroPoints.size() == 1 || std::equal(zeroPoints.begin() + 1, zeroPoints.end(), zeroPoints.begin()),
                    "All zero points should be equal");
            padValues.assign(OC * alignment, zeroPoints.front());

        } else {
            VPUX_THROW("Unsupported Quantized Type '{0}'", quantizedType);
        }
        std::vector<uint8_t> padValuesUint8;
        std::transform(padValues.begin(), padValues.end(), std::back_inserter(padValuesUint8),
                       [](int64_t value) -> uint8_t {
                           return static_cast<uint8_t>(value);
                       });
        const auto padAttr = mlir::DenseElementsAttr::get(padType, ArrayRef(padValuesUint8));

        return Const::ContentAttr::get(padAttr).quantCast(quantizedType);
    } else {
        const auto ndType = mlir::RankedTensorType::get(shape, type).cast<vpux::NDTypeInterface>();
        const auto padType = ndType.changeDimsOrder(DimsOrder::NCHW).cast<mlir::RankedTensorType>();
        const auto padValues = std::vector<vpux::type::float16>(OC * alignment, 0.f);
        const auto padAttr = mlir::DenseElementsAttr::get(padType, ArrayRef(padValues));

        return Const::ContentAttr::get(padAttr);
    }
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      Shape flatWeightShape, int64_t padding) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
    const auto origOrder = origFilterType.getDimsOrder();
    const auto flatWeightType = origFilterType.changeShape(flatWeightShape).changeDimsOrder(origOrder);
    auto flatWeightsOp =
            builder.create<IE::ShapeCastOp>(loc, flatWeightType, origFilter, getIntArrayAttr(ctx, flatWeightShape));

    // Step 2: Permute flat input to NCHW.
    auto flatWeightTypeNCHWType = flatWeightType.changeDimsOrder(DimsOrder::NCHW);
    const auto nchwAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    const auto flatWeightsDimsAttr =
            mlir::AffineMapAttr::get(getPermutationFromOrders(origOrder, DimsOrder::NCHW, ctx));
    auto flatWeightsNCHW = builder.create<IE::PermuteCastOp>(loc, flatWeightTypeNCHWType, flatWeightsOp->getResult(0),
                                                             nchwAttr, flatWeightsDimsAttr);

    // Step 3: Create padding for flat NCHW input. IC must be a multiple of 16.
    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto outShapedType = mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType())
                                       .cast<vpux::NDTypeInterface>();
    const auto outAllocType = outShapedType.changeDimsOrder(DimsOrder::NHWC);

    const auto padShape = SmallVector<int64_t>{OC, padding, 1, 1};
    const auto padContentAttr = buildPadData(origFilterType.getElementType(), padShape);

    const auto padAllocType =
            mlir::RankedTensorType::get(padShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    const auto padAllocTypeNHWC = padAllocType.changeDimsOrder(DimsOrder::NCHW);
    auto paddedTensor = builder.create<Const::DeclareOp>(loc, padAllocTypeNHWC, padContentAttr);

    // Step 4: Concatenate flat NCHW input with padding.

    auto concatViewOp =
            builder.create<IE::ConcatOp>(loc, SmallVector<mlir::Value>{flatWeightsNCHW, paddedTensor}, Dims4D::Act::C);

    // Step 5: Permute the result to NHWC.
    const auto nhwcAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(DimsOrder::NCHW, DimsOrder::NHWC, ctx));

    auto outOpNCHW =
            builder.create<IE::PermuteCastOp>(loc, outAllocType, concatViewOp.getOutput(), nhwcAttr, memPermAttr);

    return outOpNCHW.getOutput();
}

}  // namespace

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
    const auto alignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());

    const auto remainder = (filtersPerInChan * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    if (remainder == 0) {
        return origFilter;
    }

    const auto padding = alignment - remainder;

    const auto flatWeightChannelsCount = filtersPerInChan * KY * KX;
    const auto flatWeightShape = Shape{OC, flatWeightChannelsCount, 1, 1};

    if (auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>()) {
        return getAlignedConstWeights(builder, loc, weightsConst, flatWeightShape, padding);
    } else {
        return getAlignedNonConstWeights(builder, loc, origFilter, flatWeightShape, padding);
    }
}

mlir::Value alignConvWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                   const bool isCMajorConv) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
    const auto alignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());

    const auto remainder = (IC * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    if (remainder == 0) {
        return origFilter;
    }

    const auto padding = alignment - remainder;

    auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(weightsConst != nullptr, "Convolution does not provide constant weights");

    const auto weightsContentAttr = weightsConst.getContentAttr();

    const auto flatWeightShape = Shape{OC, 1, 1, IC * KY * KX};
    const auto flatWeightsContentAttr = isCMajorConv
                                                ? weightsContentAttr.reorder(DimsOrder::NCHW).reshape(flatWeightShape)
                                                : weightsContentAttr.reshape(flatWeightShape);
    const auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, 0, 0, padding});
    const auto nhwcWeightsContentAttr =
            isCMajorConv ? alignedWeightsContentAttr.reorder(DimsOrder::NHWC) : alignedWeightsContentAttr;

    const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, IC * KY * KX + padding};
    const auto outAllocType = mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType())
                                      .cast<vpux::NDTypeInterface>();
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);

    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);
    return alignedWeightsOp.getOutput();
}

Byte calculateAlignedBuffersMemoryRequirement(VPU::ArchKind arch, SmallVector<Byte>& bufferSizes) {
    Byte offsetAlignment = Byte(vpux::DEFAULT_CMX_ALIGNMENT);
    Byte sizeAlignment = Byte(1);
    if (arch == VPU::ArchKind::NPU37XX || arch == VPU::ArchKind::NPU40XX) {
        offsetAlignment = Byte(getAddressAlignmentForSwizzling(SWIZZLING_KEY_5, arch));
        sizeAlignment = Byte(vpux::getSizeAlignmentForSwizzling(arch));
    }
    return vpux::calculateAlignedBuffersMemoryRequirement(bufferSizes, offsetAlignment, sizeAlignment);
}

bool isNullOrConstWithSingleValue(mlir::Value value) {
    if (value == nullptr) {
        return true;
    }

    auto declareOp = mlir::dyn_cast_or_null<Const::DeclareOp>(value.getDefiningOp());
    if (declareOp == nullptr) {
        return false;
    }

    return declareOp.getContentAttr().isSplat();
}

vpux::TensorAttr createTensorAttrFromType(vpux::NDTypeInterface inType, mlir::MLIRContext* ctx) {
    const auto bounds =
            mlir::isa<BoundedTypeInterface>(inType) ? mlir::cast<BoundedTypeInterface>(inType).getBounds() : nullptr;
    return vpux::getTensorAttr(inType.getDimsOrder().toAffineMap(ctx), inType.getMemSpace(), bounds);
}

}  // namespace VPU
}  // namespace vpux
