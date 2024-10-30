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

llvm::unique_function<int32_t(size_t)> getBiasFunc(mlir::Type inElemType, mlir::Type outElemType,
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

        return [rescaledBiasValue = std::move(rescaledBias.value()), inElemType, biasConverter](size_t oc) -> int32_t {
            return biasConverter(rescaledBiasValue[oc], inElemType);
        };
    } else if (isFloat || isFloatInQuantOut) {
        return [biasContent = std::move(biasContent), inElemType, isWeightsQuantized, filterQuantScales,
                biasConverter](int64_t oc) -> int32_t {
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
            return biasConverter(biasVal, inElemType);
        };
    }

    VPUX_THROW("In/Out element type of NCE op mismatch. quant-quant, quant-float, float-quant or float-float type "
               "pairs required. Got: in type {0}, out type {1}",
               inElemType, outElemType);
}

llvm::unique_function<int32_t(size_t)> getMultShiftFunc(mlir::Type inElemType, mlir::Type outElemType,
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

        const auto scaleApproximation = QuantizationApproximation(quantScale);
        auto multShift = ppeConverter(checked_cast<uint8_t>(scaleApproximation.shift()),
                                      checked_cast<uint16_t>(scaleApproximation.mult()), rescale[oc], inElemType);

        return multShift;
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

    auto getMultShift = getMultShiftFunc(inElemType, outElemType, weightsElemType, ppeConverter,
                                         checked_cast<size_t>(OC), constScale);
    auto getBiasFP =
            getBiasFunc(inElemType, outElemType, weightsElemType, bias, biasConverter, checked_cast<size_t>(OC));

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

SmallVector<int32_t> vpux::VPU::NCESparsity::getInstructionListTable(ArrayRef<int> rangeAttr, ArrayRef<int> shiftAttr,
                                                                     ArrayRef<int> biasAttr) {
    // NOTE : The instruction list has 5 bits of addresses so the biggest count of instructions is 11111 = 27
    // 27 of course will be aligned to 32 and will contain NOPS inside

    auto range = to_small_vector(rangeAttr);
    auto shift = to_small_vector(shiftAttr);
    auto bias = to_small_vector(biasAttr);

    VPUX_THROW_UNLESS(range.size() == shift.size() + 1 && bias.size() == shift.size(),
                      "One instruction list table is incomplet: range={0} shift={1} bias={2}", range.size(),
                      shift.size(), bias.size());

    range.resize(9, range.back());
    shift.resize(8, shift.back());
    bias.resize(8, shift.back());

    // NOTE: first 2 addresses are hardware reserved areas
    const int32_t ADDR_OF_RESERVED = 6;
    const int32_t VALUE_RESERVED = 0;
    const int32_t ADDR_OF_ADDR_FLEX = 11;
    const int32_t VALUE_ADDR_FLEX = 8;
    // NOTE-END
    const int32_t ADDR_OF_FIRST2_BITS = 9;
    const int32_t ADDR_OF_REST_BITS = 16;
    const int32_t ADDR_OF_VALUE = 19;
    const int32_t MASK_FIRST2_BITS = 3;
    const int32_t ALU_HALT_OPCODE = 6;
    const int32_t ALU_LOAD = 2;
    const int32_t INSTRUCTION_END = 0;
    int32_t first2Bits, first3Bits;
    const int32_t sizeRange = static_cast<int32_t>(range.size());
    const int32_t sizeShift = static_cast<int32_t>(shift.size());
    const int32_t sizeBias = static_cast<int32_t>(bias.size());
    const int32_t fullSize = sizeRange + sizeShift + sizeBias;
    const int32_t noopCount = fullSize >> 4;

    const int32_t size = alignValUp<int32_t>(fullSize + noopCount, 16);

    SmallVector<int32_t> templateTable(size - noopCount - 1, ALU_HALT_OPCODE);

    const auto generateTableElement = [&](const int32_t input, const int32_t first2Bits,
                                          const int32_t first3Bits) -> int32_t {
        return ((input << ADDR_OF_VALUE) | (first3Bits << ADDR_OF_REST_BITS) | (VALUE_ADDR_FLEX << ADDR_OF_ADDR_FLEX) |
                (first2Bits << ADDR_OF_FIRST2_BITS) | (VALUE_RESERVED << ADDR_OF_RESERVED) | ALU_LOAD);
    };

    // Populate the instruction list from the table
    // Example:
    //
    // range = {-15, -13, -11, -9, -7, -5, -3, 0, 252}
    // shift = {2, 0, 2, 4, 2, 3, 1, 0}
    // bias = {1, 10, 1, -1, 1, 0, 1, 0}
    //
    // expectedOutput=* = {-7847934, -6798846, -5749758, -4700670, -3588094, -2539006, -1489918, 83458, 132268034,
    // 1196546, 148482, 1197570, 2310146, 1262082, 1786882, 6, 738818, 278530,
    // 803330, 5522434, 804354, -180222, 868866, 345090, 869890, 409602, 0, 6, 6, 6, 6, 6}
    //
    // first 9 values (first line) are for range, next 8 (second line) should be for shift but 9 + 8 > 15,
    // so we need to add ALU_HALT_OPCODE=6 to the 16 position and to continue with the remaining 2 values
    // for shift. Next 8 values (last line) are for bias, followed by INSTRUCTION_END=0, and after this
    // we need to add ALU_HALT_OPCODE=6 until buffer.size()=32.

    for (int32_t j = 0; j < fullSize; j++) {
        first2Bits = j & MASK_FIRST2_BITS;
        first3Bits = j >> 2;
        if (j < sizeRange) {
            templateTable[j] = generateTableElement(range[j], first2Bits, first3Bits);
        } else if (j < sizeRange + sizeShift) {
            templateTable[j] = generateTableElement(shift[j - sizeRange], first2Bits, first3Bits);
        } else {
            templateTable[j] = generateTableElement(bias[j - sizeRange - sizeShift], first2Bits, first3Bits);
        }
    }

    templateTable.insert(templateTable.begin() + fullSize, INSTRUCTION_END);

    if (noopCount > 0) {
        // insert ALU_HALT_OPCODE at the end of the first chain of 16 bytes
        templateTable.insert(templateTable.begin() + 15, ALU_HALT_OPCODE);
    }

    return templateTable;
}

Shape vpux::VPU::NCESparsity::inferWeightsTableShape(int64_t OC) {
    return Shape{OC, 1, 1, VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};
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
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, weightsTable);

    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataStorageType, Const::ContentAttr::get(dataAttr));

    return dataConstOp.getOutput();
}
