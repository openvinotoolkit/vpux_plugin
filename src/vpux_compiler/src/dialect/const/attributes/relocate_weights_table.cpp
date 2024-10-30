//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// RelocateWeightsTableAttr::print
//

void vpux::Const::RelocateWeightsTableAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer << "weightsPtr=";
    printer.printAttribute(getWeightsPtr());
    printer << ", ";
    printer << "sparsityPtr=";
    printer.printAttribute(getSparsityPtr());
    printer << ", ";
    printer << "offsets=";
    printer.printAttribute(getOffsets());
    printer << ", ";
    printer << "weightsTableSize=";
    printer.printAttribute(getWeightsTableSize());
    if (getWeightsElemBitSize() != nullptr) {
        printer << ", ";
        printer << "weightsElemBitSize=";
        printer.printAttribute(getWeightsElemBitSize());
    }
    if (getWeightsCompression() != nullptr) {
        printer << ", ";
        printer << "weightsCompression=";
        printer.printAttribute(getWeightsCompression());
    }
    if (getChannelOffset() != nullptr) {
        printer << ", ";
        printer << "channelOffset=";
        printer.printAttribute(getChannelOffset());
    }
    printer << ">";
}

//
// RelocateWeightsTableAttr::parse
//

mlir::Attribute vpux::Const::RelocateWeightsTableAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr weightsPtr;
    mlir::IntegerAttr sparsityPtr;
    mlir::ArrayAttr offsets;
    mlir::IntegerAttr weightsElemBitSize;
    VPUIP::SparsityCompressionAttr weightsCompression;
    mlir::IntegerAttr weightsTableSize;
    mlir::IntegerAttr channelOffset;

    if (mlir::failed(parser.parseKeyword("weightsPtr"))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseEqual())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseAttribute(weightsPtr))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseKeyword("sparsityPtr"))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseEqual())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseAttribute(sparsityPtr))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseKeyword("offsets"))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseEqual())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseAttribute(offsets))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseKeyword("weightsTableSize"))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseEqual())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseAttribute(weightsTableSize))) {
        return nullptr;
    }

    if (mlir::succeeded(parser.parseOptionalGreater())) {
        return Const::RelocateWeightsTableAttr::get(weightsPtr, sparsityPtr, offsets, weightsTableSize,
                                                    weightsElemBitSize, weightsCompression, channelOffset);
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("weightsElemBitSize"))) {
        if (mlir::failed(parser.parseEqual())) {
            return nullptr;
        }
        if (mlir::failed(parser.parseAttribute(weightsElemBitSize))) {
            return nullptr;
        }
        if (mlir::succeeded(parser.parseOptionalGreater())) {
            return Const::RelocateWeightsTableAttr::get(weightsPtr, sparsityPtr, offsets, weightsTableSize,
                                                        weightsElemBitSize, weightsCompression, channelOffset);
        }
        if (mlir::failed(parser.parseComma())) {
            return nullptr;
        }
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("weightsCompression"))) {
        if (mlir::failed(parser.parseEqual())) {
            return nullptr;
        }
        if (mlir::failed(parser.parseAttribute(weightsCompression))) {
            return nullptr;
        }
        if (mlir::succeeded(parser.parseOptionalGreater())) {
            return Const::RelocateWeightsTableAttr::get(weightsPtr, sparsityPtr, offsets, weightsTableSize,
                                                        weightsElemBitSize, weightsCompression, channelOffset);
        }
        if (mlir::failed(parser.parseComma())) {
            return nullptr;
        }
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("channelOffset"))) {
        if (mlir::failed(parser.parseEqual())) {
            return nullptr;
        }
        if (mlir::failed(parser.parseAttribute(channelOffset))) {
            return nullptr;
        }
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::RelocateWeightsTableAttr::get(weightsPtr, sparsityPtr, offsets, weightsTableSize, weightsElemBitSize,
                                                weightsCompression, channelOffset);
}

//
// RelocateWeightsTableAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::RelocateWeightsTableAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

bool vpux::Const::RelocateWeightsTableAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

//
// RelocateWeightsTableAttr::transform
//

Const::Content vpux::Const::RelocateWeightsTableAttr::transform(vpux::Const::Content& input) const {
    constexpr auto numElemPerOC = static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);
    auto inputSplat = input.isSplat();
    auto output = inputSplat ? Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                                               input.getType().getElementType(), false)
                             : Const::Content::copyUnownedBuffer(std::move(input));
    // in case input was splat code above allocated new temporary buffer which now has to be filled with
    // splat value.
    if (inputSplat) {
        input.copyTo(output.getTempBuf<char>());
    }
    auto values = output.getTempBuf<int32_t>();

    const auto weightsPtr = parseIntArrayAttr<int32_t>(getWeightsPtr());
    const auto sparsityPtr = static_cast<int32_t>(*getSparsityPtr().getValue().getRawData());
    const auto offsets = parseIntArrayAttr<int64_t>(getOffsets());
    int32_t weightPtrStep = 0;
    int32_t sparsityPtrStep = 0;
    auto numWTEntries = getWeightsTableSize().getInt() / sizeof(int32_t);

    if (numWTEntries >= numElemPerOC * 2) {
        weightPtrStep = values[1 * numElemPerOC + 0] - values[0 * numElemPerOC + 0];
        sparsityPtrStep = values[1 * numElemPerOC + 1] - values[0 * numElemPerOC + 1];
    }

    const auto channelOffset = getChannelOffset() != nullptr ? checked_cast<int32_t>(getChannelOffset().getInt()) : 0;
    const auto weightsPtrOffset = channelOffset * weightPtrStep;
    const auto sparsityPtrOffset = channelOffset * sparsityPtrStep;

    const auto OC = checked_cast<int64_t>(numWTEntries / numElemPerOC);
    const auto numClusters = checked_cast<int64_t>(offsets.size());
    // In case all clusters have the same channel offsets, the weights are not segmented
    const auto areWeightsSegmented =
            std::adjacent_find(offsets.begin(), offsets.end(), std::not_equal_to<>()) != offsets.end();
    const auto isNewCluster = [&](const int64_t oc, const int64_t currentClusterIdx) -> bool {
        return areWeightsSegmented && (currentClusterIdx + 1) < numClusters && oc >= offsets[currentClusterIdx + 1];
    };
    SmallVector<int64_t> weightsPtrSteps(OC);
    if (getWeightsCompression() != nullptr) {
        const auto numElems = to_small_vector(getWeightsCompression().getNumElems().getValues<int64_t>());
        VPUX_THROW_UNLESS(numElems.size() == static_cast<size_t>(OC),
                          "Invalid weights compression with {0} elements for {1} channels", numElems.size(), OC);
        VPUX_THROW_UNLESS(getWeightsElemBitSize() != nullptr, "Missing weights element type attribute");
        const auto weightsElemBitSize = getWeightsElemBitSize().getInt();
        const auto alignment = (getWeightsCompression().getAlignment() != nullptr)
                                       ? getWeightsCompression().getAlignment().getInt()
                                       : VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT;
        int64_t ptrOffset = 0;
        for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
            if (isNewCluster(oc, clusterIdx)) {
                clusterIdx++;
                ptrOffset = 0;
            }
            weightsPtrSteps[oc] = ptrOffset;
            const auto weightSetByteSize =
                    alignMemSize(Bit(numElems[oc] * weightsElemBitSize), Byte(1)).to<Byte>().count();
            ptrOffset += alignValUp<int64_t>(weightSetByteSize, alignment);
        }
    } else {
        for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
            if (isNewCluster(oc, clusterIdx)) {
                clusterIdx++;
            }
            weightsPtrSteps[oc] = weightPtrStep * (oc - offsets[clusterIdx]);
        }
    }
    for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
        if (isNewCluster(oc, clusterIdx)) {
            clusterIdx++;
        }

        const auto wtInd = oc * numElemPerOC;
        values[wtInd + 0] = checked_cast<int32_t>(weightsPtr[clusterIdx] + weightsPtrSteps[oc]) + weightsPtrOffset;
        if (values[wtInd + 1] != VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY) {
            values[wtInd + 1] = checked_cast<int32_t>(sparsityPtr + (oc - offsets[clusterIdx]) * sparsityPtrStep) +
                                sparsityPtrOffset;
        }
    }
    return output;
}

Const::ContentSetup vpux::Const::ContentSetup::relocateWeightsTablePointers(
        ArrayRef<uint32_t> weightsPtr, uint64_t sparsityPtr, ShapeRef offsets, uint64_t weightsTableSize,
        uint64_t weightsElemBitSize, VPUIP::SparsityCompressionAttr weightsCompression, uint64_t channelOffset) {
    return addTransformation(Const::RelocateWeightsTableAttr::get(
            getIntArrayAttr(getContext(), weightsPtr), getIntAttr(getContext(), sparsityPtr),
            getIntArrayAttr(getContext(), offsets), getIntAttr(getContext(), weightsTableSize),
            getIntAttr(getContext(), weightsElemBitSize), weightsCompression, getIntAttr(getContext(), channelOffset)));
}
