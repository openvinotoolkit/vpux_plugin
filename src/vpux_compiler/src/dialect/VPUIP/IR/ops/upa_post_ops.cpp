//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// verifyPostOp
//

static mlir::LogicalResult verifyPostOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyPostOp");

    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement RT Layer interface", op->getName());
    }

    for (auto& operand : layer.getOpOperands()) {
        const auto opVal = operand.get();
        // TODO : can we fix that limitation?
        const auto strideReqs =
                StrideReqs::compact(opVal.getType().cast<vpux::NDTypeInterface>().getRank()).remove(MemDim(1));

        if (!strideReqs.checkStrides(opVal)) {
            return errorAt(op, "Value '{0}' strides do not match requirements '{1}'", opVal, strideReqs);
        }
    }

    return mlir::success();
}

//
// ClampUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ClampUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const float min_val = static_cast<float>(getMin().convertToDouble());
    const float max_val = static_cast<float>(getMax().convertToDouble());

    const auto clamp = MVCNN::CreateClampParams(writer, min_val, max_val);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ClampParams);
    builder.add_nested_params(clamp.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::ClampUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// EluUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::EluUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const float x_val = static_cast<float>(getX().convertToDouble());

    const auto elu = MVCNN::CreateEluParams(writer, x_val);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_EluParams);
    builder.add_nested_params(elu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::EluUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// HSwishUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::HSwishUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto hswish = MVCNN::CreateHSwishParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_HSwishParams);
    builder.add_nested_params(hswish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::HSwishUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// FloorUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::FloorUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto floor = MVCNN::CreateFloorParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_FloorParams);
    builder.add_nested_params(floor.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::FloorUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// RoundUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::RoundUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto roundMode = VPUIP::convertVPUXRoundMode2MVCNN(getMode());
    const auto round = MVCNN::CreateRoundParams(writer, roundMode);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_RoundParams);
    builder.add_nested_params(round.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::RoundUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// MishUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::MishUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto mish = MVCNN::CreateMishParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_MishParams);
    builder.add_nested_params(mish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::MishUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// ErfUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ErfUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto erf = MVCNN::CreateErfParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ErfParams);
    builder.add_nested_params(erf.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::ErfUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Tan
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::TanUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto tan = MVCNN::CreateTanParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_TanParams);
    builder.add_nested_params(tan.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::TanUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Tanh
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::TanhUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto tanh = MVCNN::CreateTanhParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_TanhParams);
    builder.add_nested_params(tanh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::TanhUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Sin
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SinUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto sine = MVCNN::CreateSinParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SinParams);
    builder.add_nested_params(sine.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::SinUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Cos
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CosUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto cosine = MVCNN::CreateCosParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_CosParams);
    builder.add_nested_params(cosine.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::CosUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Sqrt
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SqrtUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto sqrt = MVCNN::CreateSqrtParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SqrtParams);
    builder.add_nested_params(sqrt.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::SqrtUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Sinh
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SinhUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto sinh = MVCNN::CreateSinhParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SinhParams);
    builder.add_nested_params(sinh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::SinhUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Cosh
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CoshUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto cosh = MVCNN::CreateCoshParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_CoshParams);
    builder.add_nested_params(cosh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::CoshUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Asinh
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AsinhUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto asinh = MVCNN::CreateAsinhParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_AsinhParams);
    builder.add_nested_params(asinh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::AsinhUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Acosh
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AcoshUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto acosh = MVCNN::CreateAcoshParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_AcoshParams);
    builder.add_nested_params(acosh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::AcoshUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// AbsUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AbsUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto abs = MVCNN::CreateAbsParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_AbsParams);
    builder.add_nested_params(abs.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::AbsUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// HSigmoidUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::HSigmoidUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto hsigmoid = MVCNN::CreateHSigmoidParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_HSigmoidParams);
    builder.add_nested_params(hsigmoid.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::HSigmoidUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// AtanUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AtanUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto atan = MVCNN::CreateAtanParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_AtanParams);
    builder.add_nested_params(atan.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::AtanUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// AsinUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AsinUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto asin = MVCNN::CreateAsinParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_AsinParams);
    builder.add_nested_params(asin.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::AsinUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// AcosUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AcosUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto acos = MVCNN::CreateAcosParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_AcosParams);
    builder.add_nested_params(acos.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::AcosUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Atanh
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AtanhUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto atanh = MVCNN::CreateAtanhParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_AtanhParams);
    builder.add_nested_params(atanh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::AtanhUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// LogUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::LogUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto log = MVCNN::CreateLogParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_LogParams);
    builder.add_nested_params(log.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::LogUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Selu
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SeluUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto alpha = getAlphaValueAttr().getValueAsDouble();
    const auto lambda = getLambdaValueAttr().getValueAsDouble();

    const auto selu = MVCNN::CreateSeluParams(writer, checked_cast<float>(alpha), checked_cast<float>(lambda));

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SeluParams);
    builder.add_nested_params(selu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// GeluUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::GeluUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto gelu = MVCNN::CreateGeluParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_GeluParams);
    builder.add_nested_params(gelu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::GeluUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// Exp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExpUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto exp = MVCNN::CreateExpParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ExpParams);
    builder.add_nested_params(exp.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::ExpUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// ReLUUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReLUUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto relu = MVCNN::CreateReluParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_ReluParams);
    builder.add_nested_params(relu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::ReLUUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// SigmoidUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SigmoidUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto sigmoid = MVCNN::CreateSigmoidParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SigmoidParams);
    builder.add_nested_params(sigmoid.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::SigmoidUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// SignUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SignUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto Sign = MVCNN::CreateSignParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SignParams);
    builder.add_nested_params(Sign.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// PRelu
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PReluUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto prelu = MVCNN::CreatePReluParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_PReluParams);
    builder.add_nested_params(prelu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// LeakyRelu
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::LeakyReluUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const float negative_slope_val = static_cast<float>(getNegativeSlope().convertToDouble());

    const auto leaky_relu = MVCNN::CreateLeakyReluParams(writer, negative_slope_val);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_LeakyReluParams);
    builder.add_nested_params(leaky_relu.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// Swish
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SwishUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto beta = getBetaValueAttr().getValueAsDouble();

    const auto swish = MVCNN::CreateSwishParams(writer, checked_cast<float>(beta));

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SwishParams);
    builder.add_nested_params(swish.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// ScaleShiftUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ScaleShiftUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto scaleShift = MVCNN::CreateScaleShiftParams(writer);

    MVCNN::PostOpsNestedParams opType{};
    if (getWeights() != nullptr && getBiases() != nullptr) {
        opType = MVCNN::PostOpsNestedParams_ScaleShiftParams;
    } else if (getWeights() != nullptr) {
        opType = MVCNN::PostOpsNestedParams_ScaleParams;
    } else if (getBiases() != nullptr) {
        opType = MVCNN::PostOpsNestedParams_BiasParams;
    } else {
        VPUX_THROW("ScaleShift must have weights or biases");
    }

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(opType);
    builder.add_nested_params(scaleShift.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::ScaleShiftUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// CeilingUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CeilingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto ceiling = MVCNN::CreateCeilingParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_CeilingParams);
    builder.add_nested_params(ceiling.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::CeilingUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// SoftPlusUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SoftPlusUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto softPlus = MVCNN::CreateSoftPlusParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_SoftPlusParams);
    builder.add_nested_params(softPlus.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::LogicalResult vpux::VPUIP::SoftPlusUPAOp::verify() {
    return verifyPostOp(getOperation());
}

//
// HardSigmoidUPAOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::HardSigmoidUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto alpha = getAlphaValueAttr().getValueAsDouble();
    const auto beta = getBetaValueAttr().getValueAsDouble();
    const auto hardSigmoid =
            MVCNN::CreateHardSigmoidParams(writer, checked_cast<float>(alpha), checked_cast<float>(beta));

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_HardSigmoidParams);
    builder.add_nested_params(hardSigmoid.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

//
// parsePostOps
//

mlir::Operation* vpux::VPUIP::BlobReader::parsePostOps(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() >= 1 && inputs.size() <= 3, "UPAPostOps supports 1, 2 or 3 inputs, got {0}",
                      inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAPostOps supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_PostOpsParams();

    mlir::Operation* op;
    switch (params->nested_params_type()) {
    case MVCNN::PostOpsNestedParams_ClampParams: {
        const auto clampParams = params->nested_params_as_ClampParams();
        op = builder.create<VPUIP::ClampUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                               getFPAttr(_ctx, clampParams->min()),
                                               getFPAttr(_ctx, clampParams->max()));
        break;
    }
    case MVCNN::PostOpsNestedParams_EluParams: {
        const auto eluParams = params->nested_params_as_EluParams();
        op = builder.create<VPUIP::EluUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                             getFPAttr(_ctx, eluParams->x()));
        break;
    }
    case MVCNN::PostOpsNestedParams_HSwishParams:
        op = builder.create<VPUIP::HSwishUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_FloorParams:
        op = builder.create<VPUIP::FloorUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_RoundParams: {
        const auto roundParams = params->nested_params_as_RoundParams();
        IE::RoundMode roundMode;
        switch (roundParams->mode()) {
        case MVCNN::RoundMode::RoundMode_HALF_TO_EVEN:
            roundMode = IE::RoundMode::HALF_TO_EVEN;
            break;
        case MVCNN::RoundMode::RoundMode_HALF_AWAY_FROM_ZERO:
            roundMode = IE::RoundMode::HALF_AWAY_FROM_ZERO;
            break;
        default:
            VPUX_THROW("Unsupported RoundMode {0}", roundParams->mode());
        }
        op = builder.create<VPUIP::RoundUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                               IE::RoundModeAttr::get(_ctx, roundMode));
        break;
    }
    case MVCNN::PostOpsNestedParams_MishParams:
        op = builder.create<VPUIP::MishUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_ErfParams:
        op = builder.create<VPUIP::ErfUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_TanParams:
        op = builder.create<VPUIP::TanUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_TanhParams:
        op = builder.create<VPUIP::TanhUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_SinParams:
        op = builder.create<VPUIP::SinUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_CosParams:
        op = builder.create<VPUIP::CosUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_SqrtParams:
        op = builder.create<VPUIP::SqrtUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_SinhParams:
        op = builder.create<VPUIP::SinhUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_CoshParams:
        op = builder.create<VPUIP::CoshUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_AsinhParams:
        op = builder.create<VPUIP::AsinhUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_AcoshParams:
        op = builder.create<VPUIP::AcoshUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_AtanhParams:
        op = builder.create<VPUIP::AtanhUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_LogParams:
        op = builder.create<VPUIP::LogUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_SeluParams: {
        const auto seluParams = params->nested_params_as_SeluParams();
        op = builder.create<VPUIP::SeluUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                              getFPAttr(_ctx, seluParams->alpha()),
                                              getFPAttr(_ctx, seluParams->lambda()));
        break;
    }
    case MVCNN::PostOpsNestedParams_GeluParams:
        op = builder.create<VPUIP::GeluUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_ReluParams:
        op = builder.create<VPUIP::ReLUUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_SigmoidParams:
        op = builder.create<VPUIP::SigmoidUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_SignParams:
        op = builder.create<VPUIP::SignUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_PReluParams:
        op = builder.create<VPUIP::PReluUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_LeakyReluParams: {
        const auto leakyReluParams = params->nested_params_as_LeakyReluParams();
        op = builder.create<VPUIP::LeakyReluUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                                   getFPAttr(_ctx, leakyReluParams->negative_slope()));
        break;
    }
    case MVCNN::PostOpsNestedParams_SwishParams: {
        const auto swishParams = params->nested_params_as_SwishParams();
        op = builder.create<VPUIP::SwishUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                               getFPAttr(_ctx, swishParams->beta()));
        break;
    }
    case MVCNN::PostOpsNestedParams_BiasParams:
        op = builder.create<VPUIP::ScaleShiftUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], nullptr, inputs[1],
                                                    outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_ScaleParams:
        op = builder.create<VPUIP::ScaleShiftUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], nullptr,
                                                    outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_ScaleShiftParams:
        op = builder.create<VPUIP::ScaleShiftUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2],
                                                    outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_CeilingParams:
        op = builder.create<VPUIP::CeilingUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_SoftPlusParams:
        op = builder.create<VPUIP::SoftPlusUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_AbsParams:
        op = builder.create<VPUIP::AbsUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_HSigmoidParams:
        op = builder.create<VPUIP::HSigmoidUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_AtanParams:
        op = builder.create<VPUIP::AtanUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_AsinParams:
        op = builder.create<VPUIP::AsinUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_AcosParams:
        op = builder.create<VPUIP::AcosUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
        break;
    case MVCNN::PostOpsNestedParams_HardSigmoidParams: {
        const auto hardSigmoidParams = params->nested_params_as_HardSigmoidParams();
        op = builder.create<VPUIP::HardSigmoidUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                                     getFPAttr(_ctx, hardSigmoidParams->alpha()),
                                                     getFPAttr(_ctx, hardSigmoidParams->beta()));
        break;
    }
    default:
        VPUX_THROW("Unsupported PostOps operation type {0}", params->nested_params_type());
    }
    return op;
}
