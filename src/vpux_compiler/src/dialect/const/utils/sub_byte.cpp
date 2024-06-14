//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"

//
// Const::isSubByte
//

bool vpux::Const::isSubByte(const size_t bitWidth) {
    return bitWidth < CHAR_BIT && bitWidth > 1;
}

//
// Const::getConstBuffer
//
// This function is used to unpack subbyte OV constants but also for reading byte by byte OV constants.
// Consider this configuration for a bit width of 4:
// 0x89, 0x88
// Will end like:
// 0x9, 0x8, 0x8, 0x8

mlir::SmallVector<char> vpux::Const::getConstBuffer(const char* sourceData, const size_t bitWidth,
                                                    const int64_t numElems) {
    if (!isSubByte(bitWidth)) {
        VPUX_THROW_UNLESS(vpux::isPowerOfTwo(bitWidth), "Invalid bitWidth: '{0}'", bitWidth);
        return SmallVector<char>(sourceData, sourceData + numElems);
    }

    auto targetData = SmallVector<char>(numElems);
    // For sub 8 bit we need to unpack the data
    const auto elemPerByte = CHAR_BIT / bitWidth;
    VPUX_THROW_UNLESS(vpux::isPowerOfTwo(elemPerByte), "Invalid number of elements per byte '{0}'", elemPerByte);
    const size_t mask = checked_cast<uint8_t>(checked_cast<uint16_t>(std::pow(2, bitWidth)) - 1);
    for (size_t idx = 0; idx < numElems / elemPerByte; idx++) {
        size_t shift = 0;
        for (size_t elemIdx = 0; elemIdx <= elemPerByte - 1; elemIdx++) {
            targetData[idx * elemPerByte + elemIdx] = (sourceData[idx] >> shift) & mask;
            shift += bitWidth;
        }
    }
    return targetData;
}
