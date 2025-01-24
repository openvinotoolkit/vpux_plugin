//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/content.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

//
// Content::fromRawBuffer
//

Const::Content vpux::Const::Content::fromRawBuffer(vpux::NDTypeInterface type, ArrayRef<char> data,
                                                   mlir::Type storageElemType, bool isSplat) {
    Const::Content content;
    content._type = type;
    content._data = Const::ConstData::fromRawBuffer(static_cast<const void*>(data.data()), data.size());
    content._storageElemType = storageElemType;
    content._isSplat = isSplat;
    return content;
}

//
// Content::allocTempBuffer
//

Const::Content vpux::Const::Content::allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType,
                                                     bool isSplat) {
    const int64_t tempBufSize = isSplat ? 1 : type.getNumElements();
    const Bit tempElemBitSize = vpux::getElemTypeSize(storageElemType);
    const auto tempBufRawBitSize = alignMemSize(tempElemBitSize * tempBufSize, Byte(1));
    auto data = Const::ConstData::allocate<char>(Byte(tempBufRawBitSize).count());
    return Const::Content(type, std::move(data), storageElemType, isSplat);
}

Const::Content vpux::Const::Content::allocTempBuffer(vpux::NDTypeInterface type, mlir::Type storageElemType,
                                                     bool isSplat, size_t tempBufRawSize) {
    // Overloading for sub-byte cases.
    auto data = Const::ConstData::allocate<char>(tempBufRawSize);
    return Const::Content(type, std::move(data), storageElemType, isSplat);
}

//
// Content::moveBuffer
//

Const::Content vpux::Const::Content::moveBuffer(vpux::NDTypeInterface type, Const::Content&& other) {
    Const::Content content(std::move(other));
    content._type = type;
    return content;
}

// The Content object might not own the referred data. This function ensures the returned Content object owns the
// referred data by copying it into a new buffer when needed
Const::Content vpux::Const::Content::copyUnownedBuffer(Const::Content&& origin) {
    if (origin._data.isMutable()) {  // is internal already
        return std::move(origin);
    }

    auto oldData = origin._data.data();
    if (!oldData.empty()) {  // could be the case after --fuse-constants?
        const auto dataSize = oldData.size();
        origin._data = Const::ConstData::allocate<char>(dataSize);
        auto mutableData = origin._data.mutableData<char>();
        std::copy_n(oldData.begin(), dataSize, mutableData.begin());
    }

    return std::move(origin);
}

//
// Content::copyTo
//

namespace {

template <class Range>
void fillBuf(const Range& range, MutableArrayRef<char> buf) {
    using value_type = typename Range::iterator::value_type;
    static const auto VALUE_BYTE_SIZE = sizeof(value_type);

    VPUX_THROW_UNLESS(buf.size() >= range.size() * VALUE_BYTE_SIZE,
                      "Buffer with byte size '{0}' is not enough to hold actual elements with '{1}' byte size",
                      buf.size(), range.size() * VALUE_BYTE_SIZE);

    for (size_t i = 0; i < range.size(); ++i) {
        auto* bufPtr = reinterpret_cast<value_type*>(buf.data() + i * VALUE_BYTE_SIZE);
        *bufPtr = range[i];
    }
}

}  // namespace

void vpux::Const::Content::copySubByteContent(MutableArrayRef<char> targetData, mlir::Type elemType) const {
    if (_isSplat) {
        const Bit elemSize = vpux::getElemTypeSize(elemType);
        const auto numShifts = CHAR_BIT / elemSize.count();
        uint8_t byteValue = 0;
        int8_t subByteValue = 0;
        // Previously, in the case of a splat boolean tensor, the value "true" was interpreted as 0x01
        // Now LLVM interprets the value as 0xff
        // This change helps to preserve the logic of handling constants on our end
        // More info here: https://reviews.llvm.org/D133743
        // The _data can't correctly retrieve data when it is of type Float, but getValues can.
        // However, getValues does not support subbytes storageElementType.
        if (_storageElemType.isInteger(1)) {
            subByteValue = _data.data().front() & 1;
        } else {
            subByteValue = getValues<int8_t>()[0];
        }

        for (int64_t shift = 0; shift < numShifts; shift++) {
            byteValue = (byteValue << elemSize.count()) + subByteValue;
        }
        std::fill(targetData.begin(), targetData.end(), byteValue);
        return;
    }

    // The source data might be stored with each sub-byte element into an individual byte.
    // This can happen when using the `CastElemType<i1>` transformation which does not alter the underlying data.
    // If the target buffer for the copy is smaller than the source buffer, the data will be packed if that is possible.
    // e.g. source data contains bytes with values 1 or 0 representing (i.e. boolean element type) while the target
    // buffer contains 1/8th of elements - the source values will be packed into bytes in the target buffer
    if (targetData.size() < _data.size()) {
        VPUX_THROW_UNLESS(_data.size() % targetData.size() == 0,
                          "Cannot pack sub-byte elements into buffer: source data size '{0}', target data size '{1}'",
                          _data.size(), targetData.size());
        auto sourceValues = getValues<int8_t>();
        const auto elemPerByte = sourceValues.size() / targetData.size();
        VPUX_THROW_UNLESS(elemPerByte <= CHAR_BIT && vpux::isPowerOfTwo(elemPerByte),
                          "Invalid number of elements per byte '{0}'", elemPerByte);

        const auto bits = CHAR_BIT / elemPerByte;
        const char mask = checked_cast<uint8_t>(checked_cast<uint16_t>(std::pow(2, bits)) - 1);
        for (size_t idx = 0; idx < sourceValues.size(); idx += elemPerByte) {
            uint8_t byte = 0;
            uint8_t shift = 0;
            for (uint64_t elemIdx = 0; elemIdx < elemPerByte; ++elemIdx) {
                byte |= (sourceValues[idx + elemIdx] & mask) << shift;
                shift += bits;
            }
            targetData[idx / elemPerByte] = byte;
        }
        return;
    }
    auto srcData = _data.data();
    std::memcpy(targetData.data(), srcData.data(), srcData.size());
}

mlir::Type vpux::Const::Content::getNormalizedQuantStorageType(mlir::quant::QuantizedType qType) {
    return vpux::normalizeQuantStorageType(qType);
}

void vpux::Const::Content::copyTo(MutableArrayRef<char> targetData) const {
    const auto elemType = getType().getElementType();
    const Bit elemSize = vpux::getElemTypeSize(elemType);
    const auto isSubByte = elemSize.count() < CHAR_BIT;
    if (isSubByte) {
        copySubByteContent(targetData, elemType);
        return;
    }

    const bool isTrivialStorage = (elemType == _storageElemType);
    if (!_isSplat && isTrivialStorage) {
        VPUX_THROW_UNLESS(targetData.size() >= _data.size(),
                          "Byte sizes of the target buffer '{0}' is smaller then storage buffer '{1}' ",
                          targetData.size(), _data.size());
        auto srcData = _data.data();
        std::memcpy(targetData.data(), srcData.data(), srcData.size());
        return;
    }

    dispatchByElemType<void>(elemType, [this, targetData](auto dummy) {
        using ElemT = std::decay_t<decltype(dummy)>;
        fillBuf(this->getValues<ElemT>(), targetData);
    });
}

//
// Content::fillWithZero
//

void vpux::Const::Content::fillWithZero() {
    if (auto perAxisQType = getType().getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto outShape = getType().getShape();
        const auto order = getType().getDimsOrder();
        const auto outMemShape = order.toMemoryOrder(outShape);

        VPUX_THROW_UNLESS(outShape.size() == 4, "Unsupported shape size {0}", outShape.size());
        VPUX_THROW_UNLESS(perAxisQType.getQuantizedDimension() == 0, "Only per-channel quantization is supported");

        const auto OC = outShape[Dims4D::Filter::OC];
        const auto IC = outShape[Dims4D::Filter::IC];
        const auto H = outShape[Dims4D::Filter::KY];
        const auto W = outShape[Dims4D::Filter::KX];

        const auto zeroPoints = perAxisQType.getZeroPoints();
        loop_4d(LoopExecPolicy::Parallel, getType().getContext(), OC, IC, H, W,
                [&](int64_t i, int64_t ic, int64_t h, int64_t w) {
                    const auto zp = zeroPoints[i];

                    const auto fillChannel = [&](auto buffer) {
                        using BufferType = std::decay_t<decltype(buffer)>;
                        using ElemType = typename BufferType::value_type;

                        const auto inMemIndND = order.toMemoryOrder(Shape{i, ic, h, w});
                        const auto inMemInd1D = getMemIndex1D(inMemIndND, outMemShape);

                        buffer[inMemInd1D] = checked_cast<ElemType>(zp);
                    };

                    mutate(fillChannel);
                });

    } else if (auto qType = getType().getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
        const auto zp = qType.getZeroPoint();

        const auto fillBuffer = [&](auto buffer) {
            using BufferType = std::decay_t<decltype(buffer)>;
            using ElemType = typename BufferType::value_type;

            std::fill_n(buffer.data(), buffer.size(), checked_cast<ElemType>(zp));
        };

        mutate(fillBuffer);
    } else {
        auto outBuf = getRawTempBuf();
        std::fill_n(outBuf.data(), outBuf.size(), char(0));
    }
}

//
// Content::setStorageElemType
//

void vpux::Const::Content::setStorageElemType(mlir::Type newStorageElemType) {
    _storageElemType = newStorageElemType;
}
