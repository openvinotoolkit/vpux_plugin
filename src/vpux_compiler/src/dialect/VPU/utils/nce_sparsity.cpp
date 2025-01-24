//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes_properties_conversion.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/algo.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <limits>
#include <numeric>

#include <llvm/ADT/bit.h>

using namespace vpux;

namespace {

using namespace VPU::NCESparsity;

constexpr std::int32_t ALIGNMENT_REQUIREMENT_IN_ELEMENTS = 16;

template <typename ScaleElemType>
llvm::unique_function<ScaleElemType(size_t)> getBiasFunc(mlir::Type inElemType, mlir::Type outElemType,
                                                         mlir::Type weightsElemType, const Const::ContentAttr& bias,
                                                         VPU::NCESparsity::BiasConverterCb biasConverter, size_t OC) {
    if (bias == nullptr) {
        return [](int64_t) -> double {
            return 0.0f;
        };
    }

    auto biasContent = bias.fold();

    const auto isInQuantized = mlir::isa<mlir::quant::QuantizedType>(inElemType);
    const auto isOutQuantized = mlir::isa<mlir::quant::QuantizedType>(outElemType);
    const auto isWeightsQuantized = mlir::isa<mlir::quant::QuantizedType>(weightsElemType);
    const auto isQuant = isInQuantized && isOutQuantized;
    const auto isFloat = !isInQuantized && !isOutQuantized;
    const auto isMixed = !isQuant && !isFloat;
    const auto isQuantInFloatOut = isInQuantized && isMixed;
    const auto isFloatInQuantOut = isOutQuantized && isMixed;

    const auto filterQuantScales =
            isWeightsQuantized ? extractScalesAndZeroPoints(weightsElemType).first : SmallVector<double>{1.0};

    if (isQuant || isQuantInFloatOut) {
        // PPE engages float by-pass in this case. Apply re-scaling.
        auto rescaledBias = VPU::NCESparsity::getRescaledBias(bias, inElemType, weightsElemType, OC);
        VPUX_THROW_WHEN(mlir::failed(rescaledBias), "Rescaled bias value is out of range");

        return [rescaledBiasValue = std::move(rescaledBias.value()), inElemType,
                biasConverter](size_t oc) -> ScaleElemType {
            return std::get<ScaleElemType>(biasConverter(rescaledBiasValue[oc], inElemType));
        };
    } else if (isFloat || isFloatInQuantOut) {
        return [biasContent = std::move(biasContent), inElemType, isWeightsQuantized, filterQuantScales,
                biasConverter](int64_t oc) -> ScaleElemType {
            auto getBiasValue = [&]() {
                if (biasContent.isSplat()) {
                    return biasContent.getSplatValue<float>();
                } else {
                    return biasContent.getValues<float>()[oc];
                }
            };
            auto biasVal = getBiasValue();
            if (isWeightsQuantized) {
                // check if filter is quantized per axis
                if (filterQuantScales.size() != 1) {
                    biasVal /= filterQuantScales[oc];
                } else {
                    biasVal /= filterQuantScales.front();
                }
            }
            return std::get<ScaleElemType>(biasConverter(biasVal, inElemType));
        };
    }

    VPUX_THROW("In/Out element type of NCE op mismatch. quant-quant, quant-float, float-quant or float-float type "
               "pairs required. Got: in type {0}, out type {1}",
               inElemType, outElemType);
}

template <typename ScaleElemType>
llvm::unique_function<ScaleElemType(size_t)> getMultShiftFunc(mlir::Type inElemType, mlir::Type outElemType,
                                                              mlir::Type weightsType,
                                                              VPU::NCESparsity::PPEConverterCb ppeConverter, size_t OC,
                                                              mlir::FloatAttr constMultiplyFpScale) {
    if (weightsType != nullptr) {
        auto inStorageType = mlir::quant::QuantizedType::castToStorageType(inElemType);
        if ((mlir::isa<mlir::quant::QuantizedType>(inElemType) && !mlir::isa<mlir::quant::QuantizedType>(weightsType) &&
             !inStorageType.isFloat8E5M2() && !inStorageType.isFloat8E4M3FN())) {
            VPUX_THROW("Unsupported In/Wt mixed precision. Got: in type {0}, wt type {1}", inElemType, weightsType);
        }
    }

    auto inQuantScale = mlir::isa<mlir::quant::QuantizedType>(inElemType) ? extractScalesAndZeroPoints(inElemType).first
                                                                          : SmallVector<double>{1.0};
    auto outQuantScale = mlir::isa<mlir::quant::QuantizedType>(outElemType)
                                 ? extractScalesAndZeroPoints(outElemType).first
                                 : SmallVector<double>{1.0};
    auto weightsQuantScales = exractWeightsScales(weightsType);
    const auto constMultiplyScale = constMultiplyFpScale ? constMultiplyFpScale.getValueAsDouble() : 1.0;

    broadcast(inQuantScale, OC);
    broadcast(outQuantScale, OC);
    broadcast(weightsQuantScales, OC);

    std::vector<double> rescale(OC, 1.0);
    for (size_t i = 0; i < rescale.size(); i++) {
        rescale[i] = (weightsQuantScales[i] * inQuantScale[i]) / outQuantScale[i] * constMultiplyScale;
    }

    return [rescale = std::move(rescale), inElemType, ppeConverter](size_t oc) {
        const auto quantScale = rescale[oc];

        const QuantizationApproximation scaleApproximation(quantScale);
        auto multShift = ppeConverter(checked_cast<uint8_t>(scaleApproximation.shift()),
                                      checked_cast<int16_t>(scaleApproximation.mult()), rescale[oc], inElemType);

        return std::get<ScaleElemType>(multShift);
    };
}

}  // namespace

int32_t vpux::VPU::NCESparsity::toFixedPoint(const double realVal) {
    const double mult = 1 << 16;
    return std::lround(realVal * mult);
}

int32_t vpux::VPU::NCESparsity::toHex(double realVal) {
    return llvm::bit_cast<int32_t>(static_cast<float>(realVal));
}

int32_t vpux::VPU::NCESparsity::getWeightPtrStep(mlir::Value weights) {
    if (weights == nullptr) {
        return 0;
    }

    const auto filterShape = getShape(weights);

    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = weights.getType().cast<vpux::NDTypeInterface>();
    const auto convAlignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());
    const auto weightsElementCount = IC * KY * KX;
    VPUX_THROW_UNLESS(weightsElementCount % convAlignment == 0,
                      "Convolution and Depthwise convolution weights size must be a multiple of {0}, got {1}",
                      convAlignment, weightsElementCount);

    const Bit eltSize = getElemTypeSize(weights.getType());
    return checked_cast<int32_t>(Byte(eltSize * IC * KY * KX).count());
}

std::vector<int32_t> vpux::VPU::NCESparsity::getWeightsTable(
        mlir::Type inElemType, mlir::Type outElemType, std::optional<int32_t> weightsPtr, int32_t weightsPtrStep,
        std::optional<int32_t> sparsityPtr, int32_t sparsityPtrStep, VPU::NCESparsity::PPEConverterCb ppeConverter,
        VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC, mlir::Type weightsElemType,
        const Const::ContentAttr& bias, mlir::FloatAttr constScale) {
    auto weightsPtrOffset = weightsPtr.has_value() ? weightsPtr.value() : 0;

    // In case of dense operation use sparsityPtrOffset beyond CMX memory range to satisfy HW requirements
    auto sparsityPtrOffset = sparsityPtr.has_value() ? sparsityPtr.value() : SPARSITY_PTR_WHEN_NO_SPARSITY;

    SmallVector<int32_t> weightsPtrs(OC, 0);
    SmallVector<int32_t> sparsityPtrs(OC, 0);
    for (auto oc : irange(OC)) {
        weightsPtrs[oc] = weightsPtrOffset;
        weightsPtrOffset += weightsPtrStep;

        sparsityPtrs[oc] = sparsityPtrOffset;
        sparsityPtrOffset += sparsityPtrStep;
    }

    return getWeightsTable(inElemType, outElemType, weightsPtrs, sparsityPtrs, ppeConverter, biasConverter, OC,
                           weightsElemType, bias, constScale);
}

std::vector<int32_t> vpux::VPU::NCESparsity::getWeightsTable(
        mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> weightsPtrs, ArrayRef<int32_t> sparsityPtrs,
        VPU::NCESparsity::PPEConverterCb ppeConverter, VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
        mlir::Type weightsElemType, const Const::ContentAttr& bias, mlir::FloatAttr constScale) {
    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create weights table without operation input/output types");
    VPUX_THROW_WHEN(static_cast<int64_t>(weightsPtrs.size()) != OC,
                    "Weights pointers size {0} different than output channels {1}", weightsPtrs.size(), OC);
    VPUX_THROW_WHEN(static_cast<int64_t>(sparsityPtrs.size()) != OC,
                    "Sparsity pointers size {0} different than output channels {1}", sparsityPtrs.size(), OC);

    auto getMultShift = getMultShiftFunc<int32_t>(inElemType, outElemType, weightsElemType, ppeConverter,
                                                  checked_cast<size_t>(OC), constScale);
    auto getBiasFP = getBiasFunc<int32_t>(inElemType, outElemType, weightsElemType, bias, biasConverter,
                                          checked_cast<size_t>(OC));

    std::vector<std::int32_t> weightsTableVals(OC * VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, 0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        VPUX_THROW_UNLESS(weightsPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "weightsPtrs[{0}] must be multiple of {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          weightsPtrs[oc]);
        VPUX_THROW_UNLESS(sparsityPtrs[oc] == SPARSITY_PTR_WHEN_NO_SPARSITY ||
                                  sparsityPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "sparsityPtrs[{0}] must be aligned to {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          sparsityPtrs[oc]);

        weightsTableVals[wtInd + 0] = weightsPtrs[oc];
        weightsTableVals[wtInd + 1] = sparsityPtrs[oc];
        weightsTableVals[wtInd + 2] = getMultShift(oc);
        weightsTableVals[wtInd + 3] = getBiasFP(oc);
    });

    return weightsTableVals;
}

std::vector<float> vpux::VPU::NCESparsity::getScaleTable(mlir::Type inElemType, mlir::Type outElemType,
                                                         VPU::NCESparsity::PPEConverterCb ppeConverter, int64_t OC,
                                                         mlir::Type weightsElemType, mlir::FloatAttr constScale) {
    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create weights table without operation input/output types");

    auto getMultShift = getMultShiftFunc<float>(inElemType, outElemType, weightsElemType, ppeConverter,
                                                checked_cast<size_t>(OC), constScale);

    std::vector<float> scaleTableVals(OC, 0.0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        scaleTableVals[oc] = getMultShift(oc);
    });

    return scaleTableVals;
}

std::vector<float> vpux::VPU::NCESparsity::getBiasTable(mlir::Type inElemType, mlir::Type outElemType,
                                                        VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                                        mlir::Type weightsElemType, const Const::ContentAttr& bias) {
    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create weights table without operation input/output types");

    auto getBiasFP =
            getBiasFunc<float>(inElemType, outElemType, weightsElemType, bias, biasConverter, checked_cast<size_t>(OC));

    std::vector<float> biasTableVals(OC, 0.0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        biasTableVals[oc] = getBiasFP(oc);
    });

    return biasTableVals;
}

std::vector<int32_t> vpux::VPU::NCESparsity::patchWeightsTableSparsityPtrs(
        const std::vector<std::int32_t>& weightsTableVals, const int32_t sparsityPtrOffset,
        const int32_t sparsityPtrStep) {
    const int64_t OC = weightsTableVals.size() / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;

    std::vector<std::int32_t> newWeightsTableVals(weightsTableVals.begin(), weightsTableVals.end());

    VPUX_THROW_UNLESS(sparsityPtrOffset % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                      "sparsityPtrOffset must be aligned to {0}, got {1}", ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                      sparsityPtrOffset);

    VPUX_THROW_UNLESS(sparsityPtrStep % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                      "sparsityPtrStep must be aligned to {0}, got {1}", ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                      sparsityPtrStep);

    int32_t offset = sparsityPtrOffset;
    for (auto oc : irange(checked_cast<size_t>(OC))) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        newWeightsTableVals[wtInd + 1] = offset;

        offset += sparsityPtrStep;
    }

    return newWeightsTableVals;
}

Shape vpux::VPU::NCESparsity::inferWeightsTableShape(int64_t OC) {
    return Shape{OC, 1, 1, VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};
}

Shape vpux::VPU::NCESparsity::inferScaleTableShape(int64_t OC) {
    return Shape{OC, 1, 1, 1};
}

Shape vpux::VPU::NCESparsity::inferBiasTableShape(int64_t OC) {
    return Shape{OC, 1, 1, 1};
}

Shape vpux::VPU::NCESparsity::inferWeightsSparsityMapShape(ShapeRef dataShape) {
    VPUX_THROW_UNLESS(dataShape.size() == 4, "Expected data shape to be 4D, while shape is {0}", dataShape);
    const auto workloadSize = std::accumulate(dataShape.begin() + 1, dataShape.end(), static_cast<int64_t>(1),
                                              std::multiplies<int64_t>());
    const auto alignment = Byte(16).to<Bit>().count();
    const auto alignedWorkloadSize = vpux::alignValUp(workloadSize, alignment);
    return Shape({dataShape.raw()[0], 1, 1, alignedWorkloadSize});
}

mlir::FailureOr<SmallVector<double>> vpux::VPU::NCESparsity::getRescaledBias(const Const::ContentAttr& biasAttr,
                                                                             mlir::Type inElemType,
                                                                             mlir::Type filterElemType, int64_t OC) {
    auto inQuantScale = inElemType.isa<mlir::quant::QuantizedType>() ? extractScalesAndZeroPoints(inElemType).first
                                                                     : SmallVector<double>{1.0};
    auto filterQuantScales = filterElemType.isa<mlir::quant::QuantizedType>()
                                     ? extractScalesAndZeroPoints(filterElemType).first
                                     : SmallVector<double>{1.0};
    broadcast(inQuantScale, OC);
    broadcast(filterQuantScales, OC);

    SmallVector<double> rescaledBias(OC, 1.0);
    std::transform(filterQuantScales.begin(), filterQuantScales.end(), inQuantScale.begin(), rescaledBias.begin(),
                   std::multiplies<>());

    auto biasContent = biasAttr.fold();
    auto biasValueRange = biasContent.getValues<double>();
    VPUX_THROW_UNLESS(biasValueRange.size() >= static_cast<size_t>(OC), "bias size {} is less than OC {}",
                      biasValueRange.size(), OC);

    std::transform(biasValueRange.begin(), biasValueRange.begin() + OC, rescaledBias.begin(), rescaledBias.begin(),
                   std::divides<>());

    const auto isValueOutOfRange = llvm::any_of(rescaledBias, [](double newBiasData) {
        return newBiasData <= std::numeric_limits<int32_t>::min() || newBiasData >= std::numeric_limits<int32_t>::max();
    });
    if (isValueOutOfRange) {
        return mlir::failure();
    }
    return rescaledBias;
}

/*
 Compute sparsification ratio of weights. It computes effective compression ratio of weights in case of weights
 sparsification. Ratio depends on number of non-zero elements and HW requirements to alignment. Acceleration depends
 mostly on memory footprint saving therefore alignment must be taken into account while computing ratio. Weights are
 grouped into sets and have the format OCx(HxWxIC) where:
 - OC is output channels that is the number of weights sets
 - HxWxIC is weights set size, its size must be aligned according to HW requirements
 Ratio is computed as follows:
 - Count number of non-zero elements in each output channel, compute their size and align up to alignment value
 - Sum the size of all output channels/sets of weights
 - Effective ratio is: 1 - (size of non-zero vals)/(size of tensor)
*/
double vpux::VPU::NCESparsity::getSparsityRatio(vpux::NDTypeInterface weightsType,
                                                ArrayRef<int64_t> numNonSparseElemsPerOC) {
    const auto elemType = weightsType.getElementType();
    const auto elemByteSize = getElemTypeSize(elemType).to<Byte>().count();
    const auto alignedChanSizeDenseVals = [&](auto sum, auto elemsInChan) {
        return sum + vpux::alignValUp(elemsInChan * elemByteSize, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
    };
    const auto actualSize = std::accumulate(numNonSparseElemsPerOC.begin(), numNonSparseElemsPerOC.end(), 0LL,
                                            alignedChanSizeDenseVals);

    const auto uncompressedSize = weightsType.getShape().totalSize() * elemByteSize;
    const auto actualSparsityRatio = 1.0 - checked_cast<double>(actualSize) / checked_cast<double>(uncompressedSize);
    VPUX_THROW_WHEN(actualSparsityRatio < 0.0, "Sparsity ratio is negative");
    return actualSparsityRatio;
}

bool vpux::VPU::NCESparsity::isSparsifiableWeightsOperand(mlir::Value operand) {
    const auto operandType = operand.getType();
    // already sparse
    if (operandType.isa<VPU::SparseTensorType>()) {
        return false;
    }
    auto sourceOp = operand.getDefiningOp<Const::DeclareOp>();
    if (!sourceOp) {
        return false;
    }
    for (const auto transformation : sourceOp.getContentAttr().getTransformations()) {
        if (transformation.isa<Const::SparsifyAttr, Const::GetSparsityMapAttr>()) {
            VPUX_THROW("Trying to sparsify already sparsity related content at '{0}'", sourceOp->getLoc());
        }
    }
    return true;
}

bool vpux::VPU::NCESparsity::isSuperdenseRequired(const VPU::ArchKind arch, const DimsOrder outOrder,
                                                  const ShapeRef outShape, const mlir::Type outElemType) {
    if (!VPU::NCEInvariant::isSuperdenseSupported(arch)) {
        return false;
    }
    // If the inner-most dimension of output shape is aligned, super-dense mode is not required.
    const auto outputMemShape = outOrder.toMemoryOrder(outShape);
    const auto outputInnerDim = outputMemShape.back();
    const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
    const auto outputInnerDimRemainder = outputInnerDim % alignment;
    return outputInnerDimRemainder != 0;
}

vpux::VPU::NCESparsity::RuntimeSparsityStatsProvider::RuntimeSparsityStatsProvider(mlir::func::FuncOp func,
                                                                                   vpux::Logger log)
        : _logger(log), _lookup({}) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto statOps = to_small_vector(module.getOps<IE::SparsityStatisticsOp>());
    VPUX_THROW_UNLESS(statOps.size() <= 1, "Module must contains 0 or 1 sparsity statistics, but got {0}",
                      statOps.size());
    if (statOps.empty()) {
        return;
    }

    auto stats = statOps.front();
    auto& infos = stats.getSparsityInfo().front().getOperations();
    for (auto& info : infos) {
        auto asOp = mlir::cast<IE::SparsityInfoOp>(info);
        const auto key = asOp.getName().str();
        _lookup.emplace(key, asOp);
    }
}

bool vpux::VPU::NCESparsity::RuntimeSparsityStatsProvider::containsStatistics() const {
    return _lookup.size() > 0;
}

bool vpux::VPU::NCESparsity::RuntimeSparsityStatsProvider::likelySparsityConsumer(mlir::Operation* op,
                                                                                  int64_t requestedInputId) const {
    auto loc = op->getLoc().dyn_cast<mlir::FusedLoc>();
    if (loc == nullptr) {
        return false;
    }
    auto locParts = loc.getLocations();
    if (locParts.empty()) {
        return false;
    }
    auto keyNameLoc = locParts.front().dyn_cast<mlir::NameLoc>();
    if (keyNameLoc == nullptr) {
        return false;
    }
    const auto key = keyNameLoc.getName().strref().data();
    for (auto it = _lookup.find(key); it != _lookup.end() && it->first == key; ++it) {
        auto opStats = it->second;
        auto inputId = opStats.getInputId();
        if (inputId != requestedInputId) {
            continue;
        }
        const auto ratio = opStats.getRatioAttr().getValueAsDouble();
        _logger.trace("Found RT stats for input {0} of '{1}'.  Sparsity ratio is {2}", requestedInputId, op->getLoc(),
                      ratio);
        return ratio >= MINIMAL_SPARSITY_THRESHOLD;
    }
    return false;
}

//
// NewWeightsTableFormatMapper
//

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::normalizeKAndReturnCurrentGroupOf128Sets(int32_t index,
                                                                                                      int32_t& k) {
    // the pattern is repeating after 128 elements
    int32_t currentGroupOf128Sets = index / 128;
    int32_t countGroupsOf128Sets = k / 128;

    // check if the current set is in the last group of 128 sets;
    // you may see the following corner case: if k is a multiple of 128, then
    // countGroupsOf128Sets == currentGroupOf128Sets will be always false
    // that's intended because otherwise the first branch will be executed and k will become 0 (it has to be 128)
    if (currentGroupOf128Sets == countGroupsOf128Sets)
        k %= 128;
    else
        k = 128;

    return currentGroupOf128Sets;
}

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::mathematicallyEncodePositionInNewZeroPointOnlyTableLayout(
        int32_t zeroPointIndex, int32_t k) {
    // the pattern is repeating after 128 elements
    int32_t currentGroupOf128Sets = normalizeKAndReturnCurrentGroupOf128Sets(zeroPointIndex, k);
    zeroPointIndex %= 128;

    int32_t elementsInASequence = k / 8;
    int32_t sequenceNumber = ((zeroPointIndex % 16) / 4) * 2 + zeroPointIndex % 2;
    int32_t positionInSequence = (zeroPointIndex / 16) * 2 + (zeroPointIndex % 4) / 2;

    int32_t elementPositionInTable = (elementsInASequence * sequenceNumber) + positionInSequence;
    return elementPositionInTable + currentGroupOf128Sets * 128;
}

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::mathematicallyDecodePositionInNewZeroPointOnlyTableLayout(
        int32_t position, int32_t k) {
    // the pattern is repeating after 128 elements
    int32_t currentGroupOf128Sets = normalizeKAndReturnCurrentGroupOf128Sets(position, k);
    position %= 128;

    int32_t elementsInASequence = k / 8;
    int32_t sequenceNumber = position / elementsInASequence;
    int32_t positionInSequence = position % elementsInASequence;

    int32_t elementPositionInTable =
            sequenceNumber / 2 * 4 + positionInSequence / 2 * 16 + (positionInSequence % 2) * 2 + sequenceNumber % 2;
    return elementPositionInTable + currentGroupOf128Sets * 128;
}

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::computeInversePermutation(
        std::vector<int32_t> v) {
    VPUX_THROW_WHEN(v.size() == 0, "Input vector is empty");

    std::unordered_set checkForDuplicates(v.begin(), v.end());
    int32_t minValue = v.size();
    int32_t maxValue = 0;
    for (auto elem : v) {
        if (elem > maxValue) {
            maxValue = elem;
        }
        if (elem < minValue) {
            minValue = elem;
        }
    }

    VPUX_THROW_WHEN(
            checkForDuplicates.size() != v.size() || minValue != 0 || maxValue != static_cast<int32_t>(v.size() - 1),
            "Input vector should contain the values from 0 to v.size() - 1");

    std::vector<int32_t> result(v.size());
    for (auto index = 0; index < static_cast<int32_t>(v.size()); index++) {
        result[v[index]] = index;
    }

    return result;
}

// zeroPointsKx[index] = value means that the zero point that is at the position value
// will be mapped to position index in the new format
// for example, zeroPointsK64[3] = 18 means that the zero point that is at position 18
// will be mapped to position 3 in the new format
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK16 =
        std::vector<int32_t>{0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK32 =
        std::vector<int32_t>{0, 2,  16, 18, 1, 3,  17, 19, 4,  6,  20, 22, 5,  7,  21, 23,
                             8, 10, 24, 26, 9, 11, 25, 27, 12, 14, 28, 30, 13, 15, 29, 31};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK48 = std::vector<int32_t>{
        0, 2,  16, 18, 32, 34, 1, 3,  17, 19, 33, 35, 4,  6,  20, 22, 36, 38, 5,  7,  21, 23, 37, 39,
        8, 10, 24, 26, 40, 42, 9, 11, 25, 27, 41, 43, 12, 14, 28, 30, 44, 46, 13, 15, 29, 31, 45, 47};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK64 =
        std::vector<int32_t>{0,  2,  16, 18, 32, 34, 48, 50, 1,  3,  17, 19, 33, 35, 49, 51, 4,  6,  20, 22, 36, 38,
                             52, 54, 5,  7,  21, 23, 37, 39, 53, 55, 8,  10, 24, 26, 40, 42, 56, 58, 9,  11, 25, 27,
                             41, 43, 57, 59, 12, 14, 28, 30, 44, 46, 60, 62, 13, 15, 29, 31, 45, 47, 61, 63};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK80 = std::vector<int32_t>{
        0,  2,  16, 18, 32, 34, 48, 50, 64, 66, 1,  3,  17, 19, 33, 35, 49, 51, 65, 67, 4,  6,  20, 22, 36, 38, 52,
        54, 68, 70, 5,  7,  21, 23, 37, 39, 53, 55, 69, 71, 8,  10, 24, 26, 40, 42, 56, 58, 72, 74, 9,  11, 25, 27,
        41, 43, 57, 59, 73, 75, 12, 14, 28, 30, 44, 46, 60, 62, 76, 78, 13, 15, 29, 31, 45, 47, 61, 63, 77, 79};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK96 = std::vector<int32_t>{
        0,  2,  16, 18, 32, 34, 48, 50, 64, 66, 80, 82, 1,  3,  17, 19, 33, 35, 49, 51, 65, 67, 81, 83,
        4,  6,  20, 22, 36, 38, 52, 54, 68, 70, 84, 86, 5,  7,  21, 23, 37, 39, 53, 55, 69, 71, 85, 87,
        8,  10, 24, 26, 40, 42, 56, 58, 72, 74, 88, 90, 9,  11, 25, 27, 41, 43, 57, 59, 73, 75, 89, 91,
        12, 14, 28, 30, 44, 46, 60, 62, 76, 78, 92, 94, 13, 15, 29, 31, 45, 47, 61, 63, 77, 79, 93, 95};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK112 = std::vector<int32_t>{
        0,   2,  16, 18, 32,  34,  48, 50, 64,  66,  80, 82, 96, 98,  1,   3,  17, 19,  33,  35, 49, 51, 65,
        67,  81, 83, 97, 99,  4,   6,  20, 22,  36,  38, 52, 54, 68,  70,  84, 86, 100, 102, 5,  7,  21, 23,
        37,  39, 53, 55, 69,  71,  85, 87, 101, 103, 8,  10, 24, 26,  40,  42, 56, 58,  72,  74, 88, 90, 104,
        106, 9,  11, 25, 27,  41,  43, 57, 59,  73,  75, 89, 91, 105, 107, 12, 14, 28,  30,  44, 46, 60, 62,
        76,  78, 92, 94, 108, 110, 13, 15, 29,  31,  45, 47, 61, 63,  77,  79, 93, 95,  109, 111};
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK128 = std::vector<int32_t>{
        0,   2,   16,  18,  32,  34,  48,  50,  64,  66,  80,  82,  96,  98,  112, 114, 1,   3,   17,  19,  33,  35,
        49,  51,  65,  67,  81,  83,  97,  99,  113, 115, 4,   6,   20,  22,  36,  38,  52,  54,  68,  70,  84,  86,
        100, 102, 116, 118, 5,   7,   21,  23,  37,  39,  53,  55,  69,  71,  85,  87,  101, 103, 117, 119, 8,   10,
        24,  26,  40,  42,  56,  58,  72,  74,  88,  90,  104, 106, 120, 122, 9,   11,  25,  27,  41,  43,  57,  59,
        73,  75,  89,  91,  105, 107, 121, 123, 12,  14,  28,  30,  44,  46,  60,  62,  76,  78,  92,  94,  108, 110,
        124, 126, 13,  15,  29,  31,  45,  47,  61,  63,  77,  79,  93,  95,  109, 111, 125, 127};

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK16InversePermutation =
        computeInversePermutation(zeroPointsK16);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK32InversePermutation =
        computeInversePermutation(zeroPointsK32);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK48InversePermutation =
        computeInversePermutation(zeroPointsK48);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK64InversePermutation =
        computeInversePermutation(zeroPointsK64);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK80InversePermutation =
        computeInversePermutation(zeroPointsK80);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK96InversePermutation =
        computeInversePermutation(zeroPointsK96);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK112InversePermutation =
        computeInversePermutation(zeroPointsK112);
std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointsK128InversePermutation =
        computeInversePermutation(zeroPointsK128);

std::vector<std::vector<int32_t>> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointTables =
        std::vector<std::vector<int32_t>>{zeroPointsK16, zeroPointsK32, zeroPointsK48,  zeroPointsK64,
                                          zeroPointsK80, zeroPointsK96, zeroPointsK112, zeroPointsK128};

std::vector<std::vector<int32_t>>
        vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::zeroPointInversePermutationTables =
                std::vector<std::vector<int32_t>>{zeroPointsK16InversePermutation,  zeroPointsK32InversePermutation,
                                                  zeroPointsK48InversePermutation,  zeroPointsK64InversePermutation,
                                                  zeroPointsK80InversePermutation,  zeroPointsK96InversePermutation,
                                                  zeroPointsK112InversePermutation, zeroPointsK128InversePermutation};

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::getZeroPointTableByK(int32_t k) {
    return zeroPointTables[k / 16 - 1];
}

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::getZeroPointInversePermutationTableByK(
        int32_t k) {
    return zeroPointInversePermutationTables[k / 16 - 1];
}

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::encodePositionInNewZeroPointOnlyTableLayout(
        int32_t zeroPointIndex, int32_t k) {
    // the pattern is repeating after 128 elements
    normalizeKAndReturnCurrentGroupOf128Sets(zeroPointIndex, k);

    auto map = getZeroPointInversePermutationTableByK(k);
    auto oldPosOffset = zeroPointIndex - zeroPointIndex % 128;
    auto newPos = oldPosOffset + map[zeroPointIndex % 128];
    return newPos;
}

int32_t vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::decodePositionInNewZeroPointOnlyTableLayout(
        int32_t position, int32_t k) {
    // the pattern is repeating after 128 elements
    normalizeKAndReturnCurrentGroupOf128Sets(position, k);

    auto map = getZeroPointTableByK(k);
    auto oldPosOffset = position - position % 128;
    auto newPos = oldPosOffset + map[position % 128];
    return newPos;
}

void vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::mapElementsToNewFormat(std::vector<int32_t>& table,
                                                                                 int32_t start, int32_t end,
                                                                                 std::vector<int32_t>& result) {
    int32_t range = end - start;

    VPUX_THROW_WHEN(start % 128 != 0, "The starting index of the range ({0}) is not a multiple of 128", start);
    VPUX_THROW_WHEN(range % 16 != 0, "Range length ({0}) is not a multiple of 16", range);
    VPUX_THROW_WHEN(range < 16 || range > 128, "Range ({0}) should be between 16 and 128", range);
    VPUX_THROW_WHEN(
            start / 128 != (end - 1) / 128,
            "All weight sets have to be from the same group of (at most) 128 weight sets: {0} / 128 != {1} / 128",
            start, end - 1);

    auto map = getZeroPointTableByK(range);
    for (auto index = start; index < end; index++) {
        auto newPos = start + map[index % 128];
        result[index] = table[newPos];
    }
}

std::vector<int32_t> vpux::VPU::NCESparsity::NewWeightsTableFormatMapper::constructNewZeroPointOnlyTable(
        std::vector<int32_t> table) {
    int32_t k = table.size();
    std::vector<int32_t> mappedTable(k, -1);

    int32_t countGroupsOf128Sets = k / 128;
    int32_t remainingSetsInLastGroup = k % 128;

    for (int index = 0; index < countGroupsOf128Sets; index++) {
        mapElementsToNewFormat(table, index * 128, index * 128 + 128, mappedTable);
    }

    if (remainingSetsInLastGroup) {
        mapElementsToNewFormat(table, countGroupsOf128Sets * 128, countGroupsOf128Sets * 128 + remainingSetsInLastGroup,
                               mappedTable);
    }

    return mappedTable;
}

//
// 5D weights
//

int32_t vpux::VPU::NCESparsity::get5DWeightPtrStep(mlir::Value weights) {
    if (weights == nullptr) {
        return 0;
    }

    const auto filterShape = getShape(weights);

    const auto IC = filterShape[DimsGroups5D::Filter::IC];
    const auto KY = filterShape[DimsGroups5D::Filter::KY];
    const auto KX = filterShape[DimsGroups5D::Filter::KX];

    const auto origFilterType = weights.getType().cast<vpux::NDTypeInterface>();
    const auto convAlignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());
    const auto weightsElementCount = IC * KY * KX;

    VPUX_THROW_WHEN((weightsElementCount % convAlignment) != 0,
                    "NCEMatMul weights size must be a multiple of {0} but got {1}", convAlignment, weightsElementCount);

    const Bit eltSize = getElemTypeSize(weights.getType());

    return checked_cast<int32_t>(Byte(eltSize * IC * KY * KX).count());
}

std::vector<int32_t> vpux::VPU::NCESparsity::create5DWeightsTableData(mlir::Value opInput, mlir::Value opOutput,
                                                                      mlir::Value weights,
                                                                      const Const::ContentAttr& bias,
                                                                      int64_t outputChannels,
                                                                      VPU::NCESparsity::PPEConverterCb ppeConverter,
                                                                      VPU::NCESparsity::BiasConverterCb biasConverter) {
    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = VPU::NCESparsity::get5DWeightPtrStep(weights);
    const auto sparsityPtrStep = 0;

    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;

    return VPU::NCESparsity::getWeightsTable(inElemType, outElemType, weightPtrOffset, weightPtrStep, sparsityPtrOffset,
                                             sparsityPtrStep, ppeConverter, biasConverter, outputChannels,
                                             weightsElemType, bias);
}

mlir::Value vpux::VPU::NCESparsity::create5DWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                               ArrayRef<int32_t> weightsTable, int64_t outputChannels,
                                                               int64_t groups) {
    const auto elemType = getSInt32Type(builder.getContext());
    const Shape weightTableShape = {groups, outputChannels, 1, 1, VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};

    const auto dataStorageType = mlir::RankedTensorType::get(weightTableShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, weightsTable);
}
