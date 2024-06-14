//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/custom_float.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

bool checkFakeQuantizeParamsShape(ShapeRef shape, int64_t numChannels) {
    if (shape.empty()) {
        return true;
    }

    if (shape.size() == 1) {
        if (shape[Dim(0)] != 1 && shape[Dim(0)] != numChannels) {
            return false;
        }

        return true;
    }

    if (shape.size() != 4) {
        return false;
    }

    if (shape[Dim(0)] != 1 || shape[Dim(2)] != 1 || shape[Dim(3)] != 1) {
        return false;
    }
    if (shape[Dim(1)] != 1 && shape[Dim(1)] != numChannels) {
        return false;
    }

    return true;
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::FakeQuantizeUPAOp::verify() {
    const auto op = getOperation();
    static const auto C = Dim(1);

    const auto inShape = getShape(getInput());
    const auto outShape = getShape(getOutput());
    if (inShape != outShape) {
        return errorAt(op, "Input and output shapes must be equal. Got: {0} != {1}", inShape, outShape);
    }

    const auto inOrder = DimsOrder::fromValue(getInput());
    const auto inStrides = getStrides(getInput());
    const auto memShape = inOrder.toMemoryOrder(inShape);

    const auto strideReqs = StrideReqs::compact(inShape.size());
    if (!strideReqs.checkStrides(getInput())) {
        return errorAt(op, "Only compact strides are supported");
    }

    const auto md0 = memShape[MemDim(0)];
    const auto md1 = memShape[MemDim(1)];

    if (Byte(md0 * md1 * FP16_SIZE) >= Byte(SHAVE_LIB_DATA_SIZE / 2)) {
        return errorAt(op, "Memory buffers doesn't fit to inner CMX buffer");
    }

    const auto numChannels = inShape[C];

    const auto inLowShape = getInputLow().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto inHighShape = getInputHigh().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto outLowShape = getOutputLow().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto outHighShape = getOutputHigh().getType().cast<vpux::NDTypeInterface>().getShape();

    if (!checkFakeQuantizeParamsShape(inLowShape, numChannels)) {
        return errorAt(op, "input_low shape is not per-tensor/per-channel : '{0}'", inLowShape);
    }
    if (!checkFakeQuantizeParamsShape(inHighShape, numChannels)) {
        return errorAt(op, "input_high shape is not per-tensor/per-channel : '{0}'", inHighShape);
    }
    if (!checkFakeQuantizeParamsShape(outLowShape, numChannels)) {
        return errorAt(op, "output_low shape is not per-tensor/per-channel : '{0}'", outLowShape);
    }
    if (!checkFakeQuantizeParamsShape(outHighShape, numChannels)) {
        return errorAt(op, "output_high shape is not per-tensor/per-channel : '{0}'", outHighShape);
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::FakeQuantizeUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto getRawFP16 = [](const vpux::type::float16& val) {
        return val.to_bits();
    };

    const auto getVecFP16 = [&](Const::ContentAttr attr) {
        const auto attrContent = attr.fold();
        return writer.createVector(attrContent.getValues<vpux::type::float16>() | transformed(getRawFP16));
    };

    const auto input_low = getVecFP16(this->getInputLowAttr());
    const auto input_high = getVecFP16(this->getInputHighAttr());
    const auto output_low = getVecFP16(this->getOutputLowAttr());
    const auto output_high = getVecFP16(this->getOutputHighAttr());

    MVCNN::FakeQuantizeParamsBuilder builder(writer);
    builder.add_levels(checked_cast<uint32_t>(getLevels()));
    builder.add_input_low(input_low);
    builder.add_input_high(input_high);
    builder.add_output_low(output_low);
    builder.add_output_high(output_high);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_FakeQuantizeParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseFakeQuantize(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                            ArrayRef<mlir::Value> outputs,
                                                            const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAFakeQuantize supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAFakeQuantize supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_FakeQuantizeParams();
    const auto levels = params->levels();
    const auto createFP16VecFromBits = [](const flatbuffers::Vector<uint16_t>* vec) {
        SmallVector<vpux::type::float16> res;
        for (const auto& elem : *vec) {
            res.push_back(vpux::type::float16::from_bits(elem));
        }
        return res;
    };
    const auto inputLow = createFP16VecFromBits(params->input_low());
    const auto inputHigh = createFP16VecFromBits(params->input_high());
    const auto outputLow = createFP16VecFromBits(params->output_low());
    const auto outputHigh = createFP16VecFromBits(params->output_high());

    const auto inputShapeType =
            mlir::RankedTensorType::get({static_cast<int64_t>(inputLow.size())}, mlir::Float16Type::get(_ctx));
    const auto outputShapeType =
            mlir::RankedTensorType::get({static_cast<int64_t>(outputLow.size())}, mlir::Float16Type::get(_ctx));

    return builder.create<VPUIP::FakeQuantizeUPAOp>(
            mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], levels,
            Const::ContentAttr::get(mlir::DenseElementsAttr::get(inputShapeType, ArrayRef(inputLow))),
            Const::ContentAttr::get(mlir::DenseElementsAttr::get(inputShapeType, ArrayRef(inputHigh))),
            Const::ContentAttr::get(mlir::DenseElementsAttr::get(outputShapeType, ArrayRef(outputLow))),
            Const::ContentAttr::get(mlir::DenseElementsAttr::get(outputShapeType, ArrayRef(outputHigh))));
}
