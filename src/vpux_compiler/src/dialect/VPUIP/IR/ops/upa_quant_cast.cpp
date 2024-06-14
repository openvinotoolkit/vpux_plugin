//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

std::pair<VPUIP::BlobWriter::Vector<uint16_t>, VPUIP::BlobWriter::Vector<uint16_t>> serializeScalesAndZeroPoints(
        mlir::Value input, mlir::Value output, VPUIP::BlobWriter& writer) {
    const auto inType = input.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outType = output.getType().cast<vpux::NDTypeInterface>().getElementType();

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = vpux::type::float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](const auto& range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    SmallVector<double> scales;
    SmallVector<int64_t> zeroPoints;
    if (qType.isa<mlir::quant::UniformQuantizedType>()) {
        auto quantParams = qType.cast<mlir::quant::UniformQuantizedType>();
        scales = {quantParams.getScale()};
        zeroPoints = {quantParams.getZeroPoint()};
    } else if (qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto quantParams = qType.cast<mlir::quant::UniformQuantizedPerAxisType>();
        scales = {quantParams.getScales().begin(), quantParams.getScales().end()};
        zeroPoints = {quantParams.getZeroPoints().begin(), quantParams.getZeroPoints().end()};
    } else {
        VPUX_THROW("Unsupported quantized type {0}", qType);
    }

    return {getVecFP16(scales), getVecFP16(zeroPoints)};
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::QuantCastUPAOp::verify() {
    const auto op = getOperation();
    const auto inType = getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outType = getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();

    if (!((inType.isF16() && outType.isa<mlir::quant::QuantizedType>()) ||
          (inType.isa<mlir::quant::QuantizedType>() && outType.isF16()))) {
        return errorAt(op, "Unsupported quantize/dequantize conversion '{0}' -> '{1}'", inType, outType);
    }

    const auto qType = inType.isa<mlir::quant::QuantizedType>() ? inType.cast<mlir::quant::QuantizedType>()
                                                                : outType.cast<mlir::quant::QuantizedType>();

    const auto qStorageType = qType.getStorageType();
    if (!qStorageType.isInteger(CHAR_BIT)) {
        return errorAt(op, "Unsupported quantized storage type '{0}'", qStorageType);
    }

    if (!qType.isa<mlir::quant::UniformQuantizedType>() && !qType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return errorAt(op, "Unsupported quantized type '{0}'", qType);
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::QuantCastUPAOp::serialize(BlobWriter& writer) {
    auto scalesAndZeroPoints = serializeScalesAndZeroPoints(getInput(), getOutput(), writer);

    MVCNN::QuantizeParamsBuilder builder(writer);
    builder.add_scale(scalesAndZeroPoints.first);
    builder.add_zero(scalesAndZeroPoints.second);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_QuantizeParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseQuantCast(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                         ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask*) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAQuantCast supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAQuantCast supports only 1 output, got {0}", outputs.size());
    return builder.create<VPUIP::QuantCastUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
}
