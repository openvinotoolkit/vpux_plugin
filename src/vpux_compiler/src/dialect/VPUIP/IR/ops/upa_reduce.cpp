//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReduceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String type;

    switch (this->getTaskType()) {
    case VPUIP::ReduceLayerType::MAX:
        type = writer.createString("max");
        break;
    case VPUIP::ReduceLayerType::MEAN:
        type = writer.createString("mean");
        break;
    case VPUIP::ReduceLayerType::LOGICAL_OR:
        type = writer.createString("logicalor");
        break;
    case VPUIP::ReduceLayerType::LOGICAL_AND:
        type = writer.createString("logicaland");
        break;
    case VPUIP::ReduceLayerType::PROD:
        type = writer.createString("prod");
        break;
    case VPUIP::ReduceLayerType::SUM:
        type = writer.createString("sum");
        break;
    case VPUIP::ReduceLayerType::MIN:
        type = writer.createString("min");
        break;
    case VPUIP::ReduceLayerType::L1:
        type = writer.createString("l1");
        break;
    case VPUIP::ReduceLayerType::L2:
        type = writer.createString("l2");
        break;
    default:
        VPUX_THROW("Unsupported ReduceLayerType {0}", this->getTaskType());
    }
    const auto axes = writer.createVector(parseIntArrayAttr<int64_t>(getAxesValue()));
    MVCNN::ReduceParamsBuilder builder(writer);

    builder.add_keep_dims(checked_cast<bool>(getKeepDims()));
    builder.add_operation(type);
    builder.add_axes_value(axes);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseReduce(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                      ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAReduce supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAReduce supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_ReduceParams();
    const auto typeStr = params->operation()->str();

    VPUIP::ReduceLayerType type;
    if (typeStr == std::string("max")) {
        type = VPUIP::ReduceLayerType::MAX;
    } else if (typeStr == std::string("mean")) {
        type = VPUIP::ReduceLayerType::MEAN;
    } else if (typeStr == std::string("logicalor")) {
        type = VPUIP::ReduceLayerType::LOGICAL_OR;
    } else if (typeStr == std::string("logicaland")) {
        type = VPUIP::ReduceLayerType::LOGICAL_AND;
    } else if (typeStr == std::string("prod")) {
        type = VPUIP::ReduceLayerType::PROD;
    } else if (typeStr == std::string("sum")) {
        type = VPUIP::ReduceLayerType::SUM;
    } else if (typeStr == std::string("min")) {
        type = VPUIP::ReduceLayerType::MIN;
    } else if (typeStr == std::string("l1")) {
        type = VPUIP::ReduceLayerType::L1;
    } else if (typeStr == std::string("l2")) {
        type = VPUIP::ReduceLayerType::L2;
    } else {
        VPUX_THROW("Unsupported ReduceLayerType {0}", typeStr);
    }

    const SmallVector<int64_t> axes{params->axes_value()->cbegin(), params->axes_value()->cend()};
    const auto keep_dims = params->keep_dims() ? mlir::UnitAttr::get(_ctx) : nullptr;

    return builder.create<VPUIP::ReduceUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                              getIntArrayAttr(_ctx, axes), keep_dims,
                                              VPUIP::ReduceLayerTypeAttr::get(_ctx, type));
}
