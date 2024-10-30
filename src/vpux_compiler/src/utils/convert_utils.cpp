//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/convert_utils.hpp"
#include "vpux/compiler/utils/loop.hpp"

using namespace vpux;

// Unpack subbyte constants.
// Consider this configuration for a bit width of 4:
// 0x89, 0x88
// Will end like:
// 0x9, 0x8, 0x8, 0x8
vpux::Const::Content vpux::subByteConversion(vpux::Const::Content& input, vpux::NDTypeInterface outputType,
                                             bool outputIsSplat, size_t bitWidth) {
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(), outputIsSplat);
    const auto sourceData = input.getRawStorageBuf();
    auto targetData = output.getRawTempBuf();

    const auto elemPerByte = CHAR_BIT / bitWidth;
    VPUX_THROW_UNLESS(vpux::isPowerOfTwo(elemPerByte), "Invalid number of elements per byte '{0}'", elemPerByte);

    const size_t mask = checked_cast<uint8_t>(checked_cast<uint16_t>(std::pow(2, bitWidth)) - 1);

    if (input.isSplat()) {
        // Unpack first element
        auto firstElem = checked_cast<uint8_t>(sourceData.front() & mask);
        std::fill_n(targetData.data(), targetData.size(), firstElem);
    } else {
        const auto numBytes = sourceData.size();
        for (size_t byteIdx = 0; byteIdx < numBytes; byteIdx++) {
            for (size_t elemIdxPerByte = 0; elemIdxPerByte < elemPerByte; elemIdxPerByte++) {
                size_t shift = elemIdxPerByte * bitWidth;
                targetData[byteIdx * elemPerByte + elemIdxPerByte] = (sourceData[byteIdx] >> shift) & mask;
            }
        }
    }

    return output;
}
