//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <vpux/utils/core/logger.hpp>

using namespace vpux;

//
// BitPackAttr::verify
//

mlir::LogicalResult vpux::Const::BitPackAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::IntegerAttr width) {
    if (width == nullptr) {
        return printTo(emitError(), "Got NULL 'width' in 'BitPackAttr'");
    }

    if (width.getValue() != 4) {
        return printTo(emitError(), "BitPackAttr does not support any bitwidth except for 4 at this point.");
    }

    return mlir::success();
}

//
// BitPackAttr::print
//

void vpux::Const::BitPackAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getWidth());
    printer << ">";
}

//
// BitPackAttr::parse
//

mlir::Attribute vpux::Const::BitPackAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr width;
    if (mlir::failed(parser.parseAttribute(width))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::BitPackAttr::get(width);
}

//
// BitPackAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::BitPackAttr::inferOutputType(vpux::NDTypeInterface input) const {
    // Check that we're not trying to pack any floating point values.
    VPUX_THROW_WHEN(input.getElementType().isa<mlir::FloatType>(), "Bit pack does not support float inputs.");
    const auto bitWidth = checked_cast<unsigned>(getWidth().getInt());
    mlir::Type outElementType;

    if (auto quantInType = mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedType>(input.getElementType())) {
        const auto minVal = quantInType.getStorageTypeMin();
        const auto maxVal = quantInType.getStorageTypeMax();
        const auto signedness = quantInType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        const auto elementIntegerType = mlir::IntegerType::get(getContext(), bitWidth, signedness);
        outElementType = mlir::quant::QuantileQuantizedType::get(
                quantInType.getFlags(), elementIntegerType, quantInType.getQuantileType(),
                quantInType.getExpressedType(), quantInType.getQuantiles(), quantInType.getScale(),
                quantInType.getZeroPoint(), minVal, maxVal);
    } else if (auto quantInType =
                       mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(input.getElementType())) {
        const auto minVal = quantInType.getStorageTypeMin();
        const auto maxVal = quantInType.getStorageTypeMax();
        const auto signedness = quantInType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        const auto elementIntegerType = mlir::IntegerType::get(getContext(), bitWidth, signedness);
        outElementType = mlir::quant::QuantileQuantizedPerAxisType::get(
                quantInType.getFlags(), elementIntegerType, quantInType.getQuantileType(),
                quantInType.getExpressedType(), quantInType.getQuantiles(), quantInType.getScales(),
                quantInType.getZeroPoints(), quantInType.getQuantizedDimension(), minVal, maxVal);
    } else if (auto quantInType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(input.getElementType())) {
        const auto minVal = quantInType.getStorageTypeMin();
        const auto maxVal = quantInType.getStorageTypeMax();
        const auto signedness = quantInType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        const auto elementIntegerType = mlir::IntegerType::get(getContext(), bitWidth, signedness);
        outElementType = mlir::quant::UniformQuantizedType::get(quantInType.getFlags(), elementIntegerType,
                                                                quantInType.getExpressedType(), quantInType.getScale(),
                                                                quantInType.getZeroPoint(), minVal, maxVal);
    } else if (auto quantInType =
                       mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(input.getElementType())) {
        const auto minVal = quantInType.getStorageTypeMin();
        const auto maxVal = quantInType.getStorageTypeMax();
        const auto signedness = quantInType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        const auto elementIntegerType = mlir::IntegerType::get(getContext(), bitWidth, signedness);
        outElementType = mlir::quant::UniformQuantizedPerAxisType::get(
                quantInType.getFlags(), elementIntegerType, quantInType.getExpressedType(), quantInType.getScales(),
                quantInType.getZeroPoints(), quantInType.getQuantizedDimension(), minVal, maxVal);
    } else if (auto intInType = mlir::dyn_cast_or_null<mlir::IntegerType>(input.getElementType())) {
        outElementType = mlir::IntegerType::get(getContext(), bitWidth, intInType.getSignedness());
    } else {
        VPUX_THROW("Got unsupported input element type '{0}' in bitpack", input.getElementType());
    }
    return input.changeElemType(outElementType);
}

bool vpux::Const::BitPackAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    VPUX_THROW_WHEN(inputIsSplat, "Bit pack does not support splat inputs.");  // as per ::transform()
    return false;
}

//
// BitPackAttr::transform
//

Const::Content vpux::Const::BitPackAttr::transform(vpux::Const::Content& input) const {
    VPUX_THROW_WHEN(input.isSplat(), "Bit pack does not support splat inputs.");
    const auto widthParam = getWidth().getInt();
    VPUX_THROW_UNLESS(widthParam == 4, "Bit pack does not support any bitwidth except for 4 at this point.");
    const auto inBuf = input.getValues<uint8_t>();
    VPUX_THROW_UNLESS((inBuf.size() % 2) == 0, "Storage buffer size is odd, which is unexpected for 4 bit packing.");
    const auto outputType = inferOutputType(input.getType());
    const Byte outputByteSize = outputType.getTotalAllocSize();
    const size_t tempBufferSize = outputByteSize.count();
    auto output = Const::Content::allocTempBuffer(outputType, getUInt8Type(getContext()),
                                                  inferOutputSplat(input.isSplat(), input.getType()), tempBufferSize);

    auto outBuf = output.getRawTempBuf();
    auto outBlobPtr = reinterpret_cast<uint8_t*>(outBuf.data());
    for (size_t idx = 0; idx < inBuf.size(); idx += 2) {
        const auto lsn = static_cast<uint8_t>(inBuf[idx + 0] & 0x0f);
        const auto msn = static_cast<uint8_t>(inBuf[idx + 1] & 0x0f);
        const auto byte = static_cast<uint8_t>((msn << 4) + lsn);
        outBlobPtr[idx / 2] = byte;
    }

    return output;
}

//
// BitPackAttr::getPositionRequirement
//

Const::details::PositionRequirement vpux::Const::BitPackAttr::getPositionRequirement() const {
    return Const::details::PositionRequirement::LAST;
}
