//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;
VPUIP::BlobWriter::SpecificTask vpux::VPUIP::LogicalNotUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String type;

    type = writer.createString("logicalnot");

    MVCNN::EltwiseParamsBuilder builder(writer);
    builder.add_operation(type);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EltwiseParams});
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::EltwiseUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String type;
    switch (this->getTaskType()) {
    case VPU::EltwiseType::ADD:
        type = writer.createString("sum");
        break;
    case VPU::EltwiseType::MULTIPLY:
        type = writer.createString("prod");
        break;
    case VPU::EltwiseType::DIVIDE:
        type = writer.createString("div");
        break;
    case VPU::EltwiseType::SQUARED_DIFF:
        type = writer.createString("sqdiff");
        break;
    case VPU::EltwiseType::POWER:
        type = writer.createString("pow");
        break;
    case VPU::EltwiseType::FLOOR_MOD:
        type = writer.createString("floormod");
        break;
    case VPU::EltwiseType::MOD:
        type = writer.createString("mod");
        break;
    case VPU::EltwiseType::MIN:
        type = writer.createString("min");
        break;
    case VPU::EltwiseType::MAX:
        type = writer.createString("max");
        break;
    case VPU::EltwiseType::AND:
        type = writer.createString("logicaland");
        break;
    case VPU::EltwiseType::EQUAL:
        type = writer.createString("compareeq");
        break;
    case VPU::EltwiseType::LESS:
        type = writer.createString("comparelt");
        break;
    case VPU::EltwiseType::LESS_EQUAL:
        type = writer.createString("comparele");
        break;
    case VPU::EltwiseType::GREATER:
        type = writer.createString("comparegt");
        break;
    case VPU::EltwiseType::GREATER_EQUAL:
        type = writer.createString("comparege");
        break;
    case VPU::EltwiseType::NOT_EQUAL:
        type = writer.createString("comparene");
        break;
    case VPU::EltwiseType::LOGICAL_OR:
        type = writer.createString("logicalor");
        break;
    case VPU::EltwiseType::LOGICAL_XOR:
        type = writer.createString("logicalxor");
        break;
    default:
        VPUX_THROW("Unsupported EltwiseType {0}", this->getTaskType());
    }

    MVCNN::EltwiseParamsBuilder builder(writer);
    builder.add_operation(type);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EltwiseParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseEltwise(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() >= 1 && inputs.size() <= 2, "UPAEltwise supports 1 or 2 inputs, got {0}",
                      inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAEltwise supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_EltwiseParams();
    const auto strType = params->operation()->str();
    VPU::EltwiseType type;
    if (strType == "sum") {
        type = VPU::EltwiseType::ADD;
    } else if (strType == "prod") {
        type = VPU::EltwiseType::MULTIPLY;
    } else if (strType == "div") {
        type = VPU::EltwiseType::DIVIDE;
    } else if (strType == "sqdiff") {
        type = VPU::EltwiseType::SQUARED_DIFF;
    } else if (strType == "pow") {
        type = VPU::EltwiseType::POWER;
    } else if (strType == "floormod") {
        type = VPU::EltwiseType::FLOOR_MOD;
    } else if (strType == "mod") {
        type = VPU::EltwiseType::MOD;
    } else if (strType == "min") {
        type = VPU::EltwiseType::MIN;
    } else if (strType == "max") {
        type = VPU::EltwiseType::MAX;
    } else if (strType == "logicaland") {
        type = VPU::EltwiseType::AND;
    } else if (strType == "compareeq") {
        type = VPU::EltwiseType::EQUAL;
    } else if (strType == "comparelt") {
        type = VPU::EltwiseType::LESS;
    } else if (strType == "comparele") {
        type = VPU::EltwiseType::LESS_EQUAL;
    } else if (strType == "comparene") {
        type = VPU::EltwiseType::NOT_EQUAL;
    } else if (strType == "comparegt") {
        type = VPU::EltwiseType::GREATER;
    } else if (strType == "comparege") {
        type = VPU::EltwiseType::GREATER_EQUAL;
    } else if (strType == "logicalnot") {
        type = VPU::EltwiseType::LOGICAL_NOT;
        return builder.create<VPUIP::LogicalNotUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                                      VPU::EltwiseTypeAttr::get(_ctx, type));
    } else if (strType == "logicalor") {
        type = VPU::EltwiseType::LOGICAL_OR;
    } else if (strType == "logicalxor") {
        type = VPU::EltwiseType::LOGICAL_XOR;
    } else {
        VPUX_THROW("Unsupported EltwiseType {0}", strType);
    }

    return builder.create<VPUIP::EltwiseUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                               VPU::EltwiseTypeAttr::get(_ctx, type));
}
