//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

mlir::LogicalResult poolSizesCheck(VPUIP::PoolingUPAOp op, int64_t sizeI, int64_t sizeO, int64_t kernel, int64_t stride,
                                   int64_t lPad, int64_t rPad) {
    if (sizeI <= 0 || sizeO <= 0 || kernel <= 0 || stride <= 0) {
        return errorAt(op, "sizeI, sizeO kernel, stride negative values do not make sense");
    }

    if (lPad < 0) {
        return errorAt(op, "Left/top side padding can not be negative '{0}'", lPad);
    }
    if (rPad < 0) {
        return errorAt(op, "Right/bottom side padding can not be negative '{0}'", rPad);
    }

    if (lPad > (kernel - 1)) {
        return errorAt(op, "Left/top padding is too big '{0}'", lPad);
    }
    if ((sizeO - 1) * stride - lPad > sizeI - 1) {
        return errorAt(op, "Output size is too big, the last kernel application is out of real data '{0}'", sizeO);
    }
    if ((sizeO - 1) * stride - lPad + (kernel - 1) > sizeI - 1 + rPad) {
        return errorAt(op, "The last kernel application is out of input size + rPad range");
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::PoolingUPAOp::verify() {
    const auto op = getOperation();
    static const auto N = Dim(0);
    static const auto C = Dim(1);
    static const auto H = Dim(2);
    static const auto W = Dim(3);

    const auto inShape = getShape(getInput());
    const auto outShape = getShape(getOutput());

    if (inShape[N] != outShape[N]) {
        return errorAt(op, "Input batch '{0}' doesn't match with output '{1}'", inShape[N], outShape[N]);
    }
    if (inShape[C] != outShape[C]) {
        return errorAt(op, "Input number of channels '{0}' doesn't match with output '{1}'", inShape[C], outShape[C]);
    }

    const auto kernelVal = parseIntArrayAttr<int64_t>(getKernel());
    const auto stridesVal = parseIntArrayAttr<int64_t>(getStrides());
    const auto padsBeginVal = parseIntArrayAttr<int64_t>(getPadsBegin());
    const auto padsEndVal = parseIntArrayAttr<int64_t>(getPadsEnd());

    if (kernelVal.size() != 2) {
        return errorAt(op, "Got unsupported kernel '{0}', only 2D is supported", kernelVal);
    }
    if (stridesVal.size() != 2) {
        return errorAt(op, "Got unsupported strides '{0}', only 2D is supported", stridesVal);
    }
    if (padsBeginVal.size() != 2) {
        return errorAt(op, "Got unsupported padsBegin '{0}', only 2D is supported", padsBeginVal);
    }
    if (padsEndVal.size() != 2) {
        return errorAt(op, "Got unsupported padsEnd '{0}', only 2D is supported", padsEndVal);
    }

    const auto kernelY = kernelVal[0];
    const auto kernelX = kernelVal[1];

    if (kernelY < 1 || kernelX < 1 || (kernelY > 64 && kernelX > 64)) {
        return errorAt(op, "Got unsupported kernel '{0}', kx, ky up to 64 only supported", kernelVal);
    }
    if (kernelY > 255 || kernelX > 255) {
        return errorAt(op, "Got unsupported kernel '{0}', kx, ky cannot exceed unsigned byte value range", kernelVal);
    }

    const auto strideY = stridesVal[0];
    const auto strideX = stridesVal[1];

    const auto padY = padsBeginVal[0];
    const auto padX = padsBeginVal[1];

    const auto bpadY = padsEndVal[0];
    const auto rpadX = padsEndVal[1];

    if (mlir::failed(poolSizesCheck(*this, inShape[W], outShape[W], kernelX, strideX, padX, rpadX))) {
        return mlir::failure();
    }
    if (mlir::failed(poolSizesCheck(*this, inShape[H], outShape[H], kernelY, strideY, padY, bpadY))) {
        return mlir::failure();
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PoolingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto kernel = VPUIP::createOrder3(this->getKernel());
    const auto strides = VPUIP::createOrder3(this->getStrides());
    const auto padsBegin = VPUIP::createOrder3(this->getPadsBegin());
    const auto padsEnd = VPUIP::createOrder3(this->getPadsEnd());

    VPUIP::BlobWriter::String type;
    switch (this->getTaskType()) {
    case VPUIP::PoolLayerType::MAX:
        type = writer.createString("max");
        break;
    case VPUIP::PoolLayerType::AVG:
        type = writer.createString("avg");
        break;
    default:
        VPUX_THROW("Unsupported PoolLayerType {0}", this->getTaskType());
    }

    const auto excludePad = writer.createString(this->getExcludePad() ? "true" : "false");

    MVCNN::PoolingParamsBuilder builder(writer);
    builder.add_pool_method(type);
    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_exclude_pad(excludePad);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PoolingParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parsePooling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAPooling supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAPooling supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_PoolingParams();
    const auto typeStr = params->pool_method()->str();
    VPUIP::PoolLayerType type;
    if (typeStr == std::string("max")) {
        type = VPUIP::PoolLayerType::MAX;
    } else if (typeStr == std::string("avg")) {
        type = VPUIP::PoolLayerType::AVG;
    } else {
        VPUX_THROW("Unsupported PoolLayerType {0}", typeStr);
    }
    const auto kernel = parseOrder3(params->kernel(), 2);
    const auto strides = parseOrder3(params->strides(), 2);
    const auto padsBegin = parseOrder3(params->pads_begin(), 2);
    const auto padsEnd = parseOrder3(params->pads_end(), 2);
    return builder.create<VPUIP::PoolingUPAOp>(
            mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], VPUIP::PoolLayerTypeAttr::get(_ctx, type), kernel,
            strides, padsBegin, padsEnd,
            params->exclude_pad()->str() == std::string("true") ? mlir::UnitAttr::get(_ctx) : nullptr);
}
