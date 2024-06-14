//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExtractImagePatchesUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::ExtractImagePatchesParamsBuilder builder(writer);

    MVCNN::ExtractImagePatchesPadMode vpux_padding;

    if (this->getAutoPad() == IE::PadType::SAME_UPPER) {
        vpux_padding = MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_SAME_UPPER;
    } else if (this->getAutoPad() == IE::PadType::SAME_LOWER) {
        vpux_padding = MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_SAME_LOWER;
    } else if (this->getAutoPad() == IE::PadType::VALID) {
        vpux_padding = MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_VALID;
    } else {
        VPUX_THROW("Unsupported pad type {0}", this->getAutoPad());
    }

    const auto sizes = parseIntArrayAttr<int64_t>(getSizesAttr());
    const auto strides = parseIntArrayAttr<int64_t>(getStridesAttr());
    const auto rates = parseIntArrayAttr<int64_t>(getRatesAttr());

    builder.add_sizeRows(checked_cast<int32_t>(sizes[0]));
    builder.add_sizeCols(checked_cast<int32_t>(sizes[1]));

    builder.add_strideRows(checked_cast<int32_t>(strides[0]));
    builder.add_strideCols(checked_cast<int32_t>(strides[1]));

    builder.add_rateRows(checked_cast<int32_t>(rates[0]));
    builder.add_rateCols(checked_cast<int32_t>(rates[1]));

    builder.add_autoPad(vpux_padding);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ExtractImagePatchesParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseExtractImagePatches(mlir::OpBuilder& builder,
                                                                   ArrayRef<mlir::Value> inputs,
                                                                   ArrayRef<mlir::Value> outputs,
                                                                   const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAExtractImagePatches supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAExtractImagePatches supports only 1 output, got {0}", outputs.size());

    const auto params = task->softLayerParams_as_ExtractImagePatchesParams();

    const auto sizes = getIntArrayAttr(_ctx, SmallVector<int32_t>{params->sizeRows(), params->sizeCols()});
    const auto strides = getIntArrayAttr(_ctx, SmallVector<int32_t>{params->strideRows(), params->strideCols()});
    const auto rates = getIntArrayAttr(_ctx, SmallVector<int32_t>{params->rateRows(), params->rateCols()});

    IE::PadType padding;
    switch (params->autoPad()) {
    case MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_SAME_LOWER:
        padding = IE::PadType::SAME_LOWER;
        break;
    case MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_SAME_UPPER:
        padding = IE::PadType::SAME_UPPER;
        break;
    case MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_VALID:
        padding = IE::PadType::VALID;
        break;
    default:
        VPUX_THROW("Unknown PadType {0}. Only same upper, same lower and valid types are supported", params->autoPad());
    }

    return builder.create<VPUIP::ExtractImagePatchesUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], sizes,
                                                           strides, rates, IE::PadTypeAttr::get(_ctx, padding));
}
