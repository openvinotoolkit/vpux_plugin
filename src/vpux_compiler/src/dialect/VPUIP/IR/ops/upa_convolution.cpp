//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::ConvolutionUPAOp::verify() {
    const auto op = getOperation();
    if (verifySameInOutSpecificDimsOrder(op, {DimsOrder::NCHW}).failed()) {
        return mlir::failure();
    }

    // There are two uPA tasks which perform convolution: ConvUPA and SWConvUPA.
    // SWConvUPA supports parallel execution on multiple uPA units.
    // However, it does not have group support, so group convolutions go to ConvUPA.
    // ConvUPA expects NCHW order, SWConvUPA expects YXOI.
    const auto expectedFilterLayout = (getGroups() > 1) ? DimsOrder::OIYX : DimsOrder::YXOI;
    const auto filterLayout = DimsOrder::fromValue(getFilter());

    if (filterLayout != expectedFilterLayout) {
        return errorAt(op, "filter layout must be {0}, got {1}", expectedFilterLayout, filterLayout);
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ConvolutionUPAOp::serialize(VPUIP::BlobWriter& writer) {
    static const auto dY = Dim(2);
    static const auto dX = Dim(3);

    const auto strides = VPUIP::createOrder3(this->getStrides());
    const auto dilations = VPUIP::createOrder3(this->getDilations());
    const auto padsBegin = VPUIP::createOrder3(this->getPadsBegin());
    const auto padsEnd = VPUIP::createOrder3(this->getPadsEnd());

    const auto filterShape = getShape(getFilter());
    const auto kernel =
            MVCNN::order3(checked_cast<uint8_t>(filterShape[dX]), checked_cast<uint8_t>(filterShape[dY]), 0);

    if (getGroups() > 1) {
        MVCNN::ConvolutionParamsBuilder builder(writer);
        builder.add_kernel(&kernel);
        builder.add_strides(&strides);
        builder.add_dilations(&dilations);
        builder.add_pads_begin(&padsBegin);
        builder.add_pads_end(&padsEnd);
        builder.add_group(checked_cast<int32_t>(getGroups()));
        const auto paramsOff = builder.Finish();
        return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConvolutionParams});
    } else {
        MVCNN::SWConvolutionParamsBuilder builder(writer);
        builder.add_kernel(&kernel);
        builder.add_strides(&strides);
        builder.add_dilations(&dilations);
        builder.add_pads_begin(&padsBegin);
        builder.add_pads_end(&padsEnd);
        builder.add_group(checked_cast<int32_t>(getGroups()));
        const auto paramsOff = builder.Finish();
        return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SWConvolutionParams});
    }
}

mlir::Operation* vpux::VPUIP::BlobReader::parseConvolution(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                           ArrayRef<mlir::Value> outputs,
                                                           const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3, "UPAConvolution supports only 2 or 3 inputs, got {0}",
                      inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAConvolution supports only 1 output, got {0}", outputs.size());

    if (const auto* params = task->softLayerParams_as_ConvolutionParams()) {
        const auto strides = parseOrder3(params->strides());
        const auto dilations = parseOrder3(params->dilations());
        const auto padsBegin = parseOrder3(params->pads_begin());
        const auto padsEnd = parseOrder3(params->pads_end());

        return builder.create<VPUIP::ConvolutionUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1],
                                                       inputs.size() == 3 ? inputs[2] : nullptr, outputs[0], strides,
                                                       dilations, padsBegin, padsEnd, params->group());
    } else if (const auto* params = task->softLayerParams_as_SWConvolutionParams()) {
        const auto strides = parseOrder3(params->strides());
        const auto dilations = parseOrder3(params->dilations());
        const auto padsBegin = parseOrder3(params->pads_begin());
        const auto padsEnd = parseOrder3(params->pads_end());

        return builder.create<VPUIP::ConvolutionUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1],
                                                       inputs.size() == 3 ? inputs[2] : nullptr, outputs[0], strides,
                                                       dilations, padsBegin, padsEnd, params->group());
    } else {
        VPUX_THROW("Unknown SW Convolution '{0}'", task->softLayerParams_type());
    }
}
