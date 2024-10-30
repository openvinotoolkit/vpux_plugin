//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"

using namespace vpux;

NDTypeInterface Const::FuseAttr::inferOutputType(NDTypeInterface input) const {
    (void)input;
    return getFusedType();
}

bool vpux::Const::FuseAttr::inferOutputSplat(bool, vpux::NDTypeInterface) {
    return false;
}

void appendContentToVector(Const::Content& content, MutableArrayRef<char> buffer, size_t& start) {
    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    auto* oldEnd = buffer.data() + start;
    VPUX_THROW_UNLESS(start + bufSizeBytes <= buffer.size(),
                      "Overflow during fusing buffer size {0}, size after copying {1}", buffer.size(),
                      start + bufSizeBytes);
    MutableArrayRef<char> newBufferSlice(reinterpret_cast<char*>(oldEnd), bufSizeBytes);
    content.copyTo(newBufferSlice);
    start += bufSizeBytes;
}

Const::Content Const::FuseAttr::transform(Const::Content& input) const {
    auto outputType = inferOutputType(input.getType());
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(),
                                                  inferOutputSplat(input.isSplat(), input.getType()));

    auto fusedBuffer = output.getRawTempBuf();

    Const::ContentAttr contentVector[] = {getWeightsTable(), getWeights(), getSparsity(), getActivations()};

    size_t index = 0;
    for (auto content : contentVector) {
        if (content == nullptr) {
            continue;
        }
        auto foldedContent = content.fold();
        auto contentType = foldedContent.getType().cast<vpux::NDTypeInterface>();
        auto elemType = contentType.getElementType();

        if (elemType.isInteger(1)) {
            const auto packedNumElems = contentType.getNumElements() / CHAR_BIT;
            const auto packedElemType = getUInt8Type(contentType.getContext());
            const auto packedContentType =
                    contentType.changeShapeElemType(Shape({1, 1, 1, packedNumElems}), packedElemType);
            auto packedContent = Const::Content::fromRawBuffer(packedContentType, foldedContent.getRawStorageBuf(),
                                                               packedElemType, foldedContent.isSplat());
            appendContentToVector(packedContent, fusedBuffer, index);
        } else {
            appendContentToVector(foldedContent, fusedBuffer, index);
        }
    }

    return output;
}

Const::ContentSetup Const::ContentSetup::fuse(mlir::RankedTensorType fusedTensorType, Const::ContentAttr weightsTable,
                                              Const::ContentAttr weights, Const::ContentAttr sparsity,
                                              Const::ContentAttr activations) {
    return addTransformation(
            Const::FuseAttr::get(getContext(), fusedTensorType, weightsTable, weights, sparsity, activations));
}
