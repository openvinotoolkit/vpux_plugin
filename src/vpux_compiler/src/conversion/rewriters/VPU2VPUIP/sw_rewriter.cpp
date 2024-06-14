//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPU2VPUIP/sw_rewriter.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux {

mlir::Operation* createRTLayer(VPU::QuantizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::QuantCastUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::QuantCastUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::DequantizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::QuantCastUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::QuantCastUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::ConvertOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ConvertUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ConvertUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::ReLUOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReLUUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReLUUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::SigmoidOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SigmoidUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SigmoidUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::SignOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SignUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SignUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::HSwishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::HSwishUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::HSwishUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::FloorOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::FloorUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::FloorUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::RoundOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::RoundUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::RoundUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getMode());
}

mlir::Operation* createRTLayer(VPU::MishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::MishUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::MishUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::ErfOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ErfUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ErfUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::TanOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::TanUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::TanUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::TanhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::TanhUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::TanhUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::SinOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SinUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SinUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::CosOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CosUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CosUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::SqrtOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SqrtUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SqrtUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::SinhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SinhUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SinhUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::CoshOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CoshUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CoshUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::AsinhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AsinhUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AsinhUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::AcoshOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AcoshUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AcoshUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::AtanhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AtanhUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AtanhUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::LogOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::LogUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::LogUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::GeluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GeluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GeluUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::NegativeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::NegativeUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::NegativeUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::PReluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PReluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PReluUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getNegativeSlope(),
                                       newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::GatherOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GatherUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GatherUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getIndices(), newOp.getOutputBuff(),
                                        origOp.getAxisValueAttr(), origOp.getBatchDimsAttr());
}

mlir::Operation* createRTLayer(VPU::GatherNDOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GatherNDUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GatherNDUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getIndices(), newOp.getOutputBuff(),
                                          origOp.getBatchDimsAttr());
}

mlir::Operation* createRTLayer(VPU::YuvToRgbOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    const auto newInp2 = origOp.getInput2() != nullptr ? allBufs[2 - 1] : nullptr;
    const auto newInp3 = origOp.getInput3() != nullptr ? allBufs[3 - 1] : nullptr;
    return b.create<VPUIP::YuvToRgbUPAOp>(origOp.getLoc(), allBufs[0], newInp2, newInp3, allBufs.back(),
                                          origOp.getInFmtAttr(), origOp.getOutFmtAttr());
}

mlir::Operation* createRTLayer(VPU::GatherElementsOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GatherElementsUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GatherElementsUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getIndices(),
                                                newOp.getOutputBuff(), origOp.getAxisAttr());
}

mlir::Operation* createRTLayer(VPU::ScatterNDUpdateOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ScatterNDUpdateUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ScatterNDUpdateUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getIndices(),
                                                 newOp.getUpdates(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::ScatterUpdateOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ScatterUpdateUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ScatterUpdateUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getIndices(),
                                               newOp.getUpdates(), newOp.getOutputBuff(), origOp.getAxisValueAttr());
}

mlir::Operation* createRTLayer(VPU::AddOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::ADD));
}

mlir::Operation* createRTLayer(VPU::MultiplyOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::MULTIPLY));
}

mlir::Operation* createRTLayer(VPU::AndOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::AND));
}

mlir::Operation* createRTLayer(VPU::DivideOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::DIVIDE));
}

mlir::Operation* createRTLayer(VPU::SquaredDifferenceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(
            origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
            VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::SQUARED_DIFF));
}

mlir::Operation* createRTLayer(VPU::PowerOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::POWER));
}

mlir::Operation* createRTLayer(VPU::FloorModOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::FLOOR_MOD));
}

mlir::Operation* createRTLayer(VPU::ModOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::MOD));
}

mlir::Operation* createRTLayer(VPU::MinimumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::MIN));
}

mlir::Operation* createRTLayer(VPU::MaximumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::MAX));
}

mlir::Operation* createRTLayer(VPU::SoftMaxOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SoftMaxUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SoftMaxUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                         origOp.getAxisIndAttr());
}

mlir::Operation* createRTLayer(VPU::AvgPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PoolingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PoolingUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                         VPUIP::PoolLayerTypeAttr::get(origOp.getContext(), VPUIP::PoolLayerType::AVG),
                                         origOp.getKernelSizeAttr(), origOp.getStridesAttr(), origOp.getPadsBeginAttr(),
                                         origOp.getPadsEndAttr(), origOp.getExcludePadsAttr());
}

mlir::Operation* createRTLayer(VPU::MaxPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PoolingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PoolingUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                         VPUIP::PoolLayerTypeAttr::get(origOp.getContext(), VPUIP::PoolLayerType::MAX),
                                         origOp.getKernelSizeAttr(), origOp.getStridesAttr(), origOp.getPadsBeginAttr(),
                                         origOp.getPadsEndAttr(), nullptr);
}

mlir::Operation* createRTLayer(VPU::ClampOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ClampUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ClampUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getMinAttr(),
                                       origOp.getMaxAttr());
}

mlir::Operation* createRTLayer(VPU::EluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EluUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getXAttr());
}

mlir::Operation* createRTLayer(VPU::LeakyReluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::LeakyReluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::LeakyReluUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                           origOp.getNegativeSlopeAttr());
}

mlir::Operation* createRTLayer(VPU::GRNOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GRNUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GRNUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getBiasAttr());
}

mlir::Operation* createRTLayer(VPU::LRN_IEOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::NormUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::NormUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAlphaAttr(),
                                      origOp.getBetaAttr(), origOp.getBiasAttr(), origOp.getSizeAttr(),
                                      origOp.getRegionAttr());
}

mlir::Operation* createRTLayer(VPU::BroadcastOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::BroadcastUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::BroadcastUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getTargetShape(),
                                           newOp.getAxesMapping(), newOp.getOutputBuff(), origOp.getModeAttr());
}

mlir::Operation* createRTLayer(VPU::ReduceMaxOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAxesValueAttr(),
            origOp.getKeepDimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::MAX));
}

mlir::Operation* createRTLayer(VPU::ReduceMeanOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAxesValueAttr(),
            origOp.getKeepDimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::MEAN));
}

mlir::Operation* createRTLayer(VPU::ReduceProdOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAxesValueAttr(),
            origOp.getKeepDimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::PROD));
}

mlir::Operation* createRTLayer(VPU::ReduceSumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAxesValueAttr(),
            origOp.getKeepDimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::SUM));
}

mlir::Operation* createRTLayer(VPU::ReduceMinOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAxesValueAttr(),
            origOp.getKeepDimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::MIN));
}

mlir::Operation* createRTLayer(VPU::ReduceL1Op origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAxesValueAttr(),
            origOp.getKeepDimsAttr(), VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::L1));
}

mlir::Operation* createRTLayer(VPU::ReduceL2Op origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAxesValueAttr(),
            origOp.getKeepDimsAttr(), VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::L2));
}

mlir::Operation* createRTLayer(VPU::ReduceLogicalOrOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAxesValueAttr(),
            origOp.getKeepDimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::LOGICAL_OR));
}

mlir::Operation* createRTLayer(VPU::ReduceLogicalAndOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAxesValueAttr(),
            origOp.getKeepDimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::LOGICAL_AND));
}

mlir::Operation* createRTLayer(VPU::PerAxisTileOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PerAxisTileUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PerAxisTileUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                             origOp.getAxisAttr(), origOp.getTilesAttr());
}

mlir::Operation* createRTLayer(VPU::ROIPoolingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ROIPoolingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ROIPoolingUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getCoords(), newOp.getOutputBuff(),
                                            origOp.getOutputSizeAttr(), origOp.getSpatialScaleAttr(),
                                            origOp.getMethodAttr());
}

mlir::Operation* createRTLayer(VPU::PSROIPoolingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PSROIPoolingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PSROIPoolingUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getCoords(), newOp.getOutputBuff(), origOp.getOutputDimAttr(),
            origOp.getSpatialScaleAttr(), origOp.getGroupSizeAttr(), origOp.getSpatialBinsXAttr(),
            origOp.getSpatialBinsYAttr(), origOp.getModeAttr());
}

mlir::Operation* createRTLayer(VPU::ROIAlignOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ROIAlignUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ROIAlignUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getCoords(), newOp.getRoisIdx(),
                                          newOp.getOutputBuff(), origOp.getPooledHAttr(), origOp.getPooledWAttr(),
                                          origOp.getSamplingRatioAttr(), origOp.getSpatialScaleAttr(),
                                          origOp.getPoolingModeAttr());
}

mlir::Operation* createRTLayer(VPU::GroupConvolutionOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ConvolutionUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ConvolutionUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getFilter(), newOp.getBias(),
                                             newOp.getOutputBuff(), origOp.getStridesAttr(), origOp.getDilationsAttr(),
                                             origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(),
                                             origOp.getGroupsAttr());
}

mlir::Operation* createRTLayer(VPU::SwishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SwishUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SwishUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                       origOp.getBetaValueAttr());
}

mlir::Operation* createRTLayer(VPU::DetectionOutputOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    const auto newPreds = origOp.getInAdditionalPreds() != nullptr ? allBufs[3] : nullptr;
    const auto newProposals = origOp.getInAdditionalProposals() != nullptr ? allBufs[4] : nullptr;
    return b.create<VPUIP::DetectionOutputUPAOp>(origOp->getLoc(), allBufs[0], allBufs[1], allBufs[2], newPreds,
                                                 newProposals, allBufs.back(), origOp.getAttr());
}

mlir::Operation* createRTLayer(VPU::ScaleShiftOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    mlir::Value newWeights;
    mlir::Value newBiases;
    if (origOp.getWeights() != nullptr && origOp.getBiases() != nullptr) {
        newWeights = allBufs[1];
        newBiases = allBufs[2];
    } else if (origOp.getWeights() != nullptr) {
        newWeights = allBufs[1];
    } else if (origOp.getBiases() != nullptr) {
        newBiases = allBufs[1];
    } else {
        VPUX_THROW("ScaleShift must have weights or biases");
    }
    return b.create<VPUIP::ScaleShiftUPAOp>(origOp->getLoc(), allBufs[0], newWeights, newBiases, allBufs.back());
}

mlir::Operation* createRTLayer(VPU::CTCGreedyDecoderOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CTCGreedyDecoderUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CTCGreedyDecoderUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getSequenceLengths(),
                                                  newOp.getOutputBuff(), origOp.getMergeRepeatedAttr());
}

mlir::Operation* createRTLayer(VPU::CTCGreedyDecoderSeqLenOp origOp, ArrayRef<mlir::Value> allBufs,
                               mlir::OpBuilder& b) {
    VPUIP::CTCGreedyDecoderSeqLenUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CTCGreedyDecoderSeqLenUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getSequenceLength(),
                                                        newOp.getBlankIndex(), newOp.getOutputBuff(),
                                                        newOp.getOutputLengthBuff(), origOp.getMergeRepeatedAttr());
}

mlir::Operation* createRTLayer(VPU::ProposalOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ProposalUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ProposalUPAOp>(origOp.getLoc(), newOp.getClassProbs(), newOp.getBboxDeltas(),
                                          newOp.getImageShape(), newOp.getOutputBuff(), newOp.getProbsBuff(),
                                          origOp.getProposalAttrs());
}

mlir::Operation* createRTLayer(VPU::RollOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::RollUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::RollUPAOp>(origOp.getLoc(), newOp.getData(), newOp.getShift(), newOp.getAxes(),
                                      newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::PadOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUX_THROW_UNLESS(origOp.getPadsBeginAttr().has_value() && origOp.getPadsEndAttr().has_value(),
                      "PadOp must use attributes for `pads_begin` and `pads_end` params");

    VPUIP::PadUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PadUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                     origOp.getPadsBeginAttrAttr(), origOp.getPadsEndAttrAttr(),
                                     origOp.getPadValueAttrAttr(), origOp.getModeAttr());
}

mlir::Operation* createRTLayer(VPU::ExpOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ExpUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ExpUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::InterpolateOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUX_THROW_UNLESS(origOp.getSizesAttr().has_value() && origOp.getScalesAttr().has_value(),
                      "Interpolate must have constant sizes or scales");
    VPUX_THROW_UNLESS(origOp.getAxesAttr().has_value(), "Interpolate must have constant axes");

    VPUIP::InterpolateUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::InterpolateUPAOp>(
            origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getAttr().getMode().getValue(),
            origOp.getAttr().getCoordMode().getValue(), origOp.getAttr().getNearestMode().getValue(),
            origOp.getAttr().getAntialias().getValue());
}

mlir::Operation* createRTLayer(VPU::StridedSliceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUX_THROW_UNLESS(allBufs.size() == 2, "Constant inputs should have been converted to attributes");
    VPUX_THROW_UNLESS(origOp.getBeginsAttrAttr(), "begins_attr is null");
    VPUX_THROW_UNLESS(origOp.getEndsAttrAttr(), "ends_attr is null");
    VPUX_THROW_UNLESS(origOp.getStridesAttrAttr(), "strides_attr is null");

    return b.create<VPUIP::StridedSliceUPAOp>(origOp.getLoc(), allBufs[0], allBufs.back(), origOp.getBeginsAttrAttr(),
                                              origOp.getEndsAttrAttr(), origOp.getStridesAttrAttr());
}

mlir::Operation* createRTLayer(VPU::RegionYoloOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::RegionYoloUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::RegionYoloUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                            origOp.getCoords(), origOp.getClasses(), origOp.getNumRegions(),
                                            origOp.getDoSoftmaxAttr(), origOp.getMask());
}

mlir::Operation* createRTLayer(VPU::ReorgYoloOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReorgYoloUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReorgYoloUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                           origOp.getStrideAttr());
}

mlir::Operation* createRTLayer(VPU::MVNOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::MVNUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::MVNUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                     origOp.getAcrossChannelsAttr(), origOp.getNormalizeVarianceAttr(),
                                     origOp.getEpsAttr());
}

mlir::Operation* createRTLayer(VPU::DepthToSpaceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::DepthToSpaceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::DepthToSpaceUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                              origOp.getBlockSizeAttr(), origOp.getModeAttr());
}

mlir::Operation* createRTLayer(VPU::MemPermuteOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PermuteUPAOp::Adaptor newOp(allBufs);

    return b.create<VPUIP::PermuteUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(), origOp.getMemPerm());
}

mlir::Operation* createRTLayer(VPU::SoftPlusOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SoftPlusUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SoftPlusUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::CeilingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CeilingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CeilingUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::NormalizeIEOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::NormalizeIEUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::NormalizeIEUPAOp>(origOp.getLoc(), newOp.getData(), newOp.getWeights(),
                                             newOp.getOutputBuff(), origOp.getEps(), origOp.getAcrossSpatial(),
                                             origOp.getChannelShared());
}

mlir::Operation* createRTLayer(VPU::CumSumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CumSumUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CumSumUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                        origOp.getAxisValueAttr(), origOp.getExclusiveAttr(), origOp.getReverseAttr());
}

mlir::Operation* createRTLayer(VPU::EqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::EQUAL));
}

mlir::Operation* createRTLayer(VPU::SelectOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SelectUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SelectUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getInput3(),
                                        newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::UpsamplingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::UpsamplingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::UpsamplingUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                            origOp.getUpsamplingFactorAttr(), origOp.getPadAttr());
}

mlir::Operation* createRTLayer(VPU::LessOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LESS));
}

mlir::Operation* createRTLayer(VPU::LessEqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LESS_EQUAL));
}

mlir::Operation* createRTLayer(VPU::NotEqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::NOT_EQUAL));
}

mlir::Operation* createRTLayer(VPU::GreaterOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::GREATER));
}

mlir::Operation* createRTLayer(VPU::GreaterEqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(
            origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
            VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::GREATER_EQUAL));
}

mlir::Operation* createRTLayer(VPU::LogicalNotOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::LogicalNotUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::LogicalNotUPAOp>(
            origOp.getLoc(), newOp.getInput1(), newOp.getOutputBuff(),
            VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LOGICAL_NOT));
}

mlir::Operation* createRTLayer(VPU::LogicalOrOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LOGICAL_OR));
}

mlir::Operation* createRTLayer(VPU::LogicalXorOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.getInput1(), newOp.getInput2(), newOp.getOutputBuff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LOGICAL_XOR));
}

mlir::Operation* createRTLayer(VPU::SpaceToDepthOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SpaceToDepthUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SpaceToDepthUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                              origOp.getBlockSize(), origOp.getMode());
}

mlir::Operation* createRTLayer(VPU::AbsOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AbsUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AbsUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::AtanOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AtanUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AtanUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::AsinOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AsinUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AsinUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::AcosOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AcosUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AcosUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::HSigmoidOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::HSigmoidUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::HSigmoidUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::HardSigmoidOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::HardSigmoidUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::HardSigmoidUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getOutputBuff(),
                                             origOp.getAlphaValueAttr(), origOp.getBetaValueAttr());
}

mlir::Operation* createRTLayer(VPU::EmbeddingSegmentsSumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EmbeddingSegmentsSumUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EmbeddingSegmentsSumUPAOp>(
            origOp.getLoc(), newOp.getEmbTable(), newOp.getOutputBuff(), origOp.getIndicesValueAttr(),
            origOp.getSegmentIdsValueAttr(), origOp.getNumSegmentsValueAttr(), origOp.getDefaultIndexValueAttr(),
            origOp.getPerSampleWeightsValueAttr());
}

mlir::Operation* createRTLayer(VPU::BucketizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::BucketizeUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::BucketizeUPAOp>(origOp.getLoc(), newOp.getData(), newOp.getBuckets(), newOp.getOutputBuff(),
                                           origOp.getOutputTypeAttr(), origOp.getWithRightBoundAttr());
}

mlir::Operation* createRTLayer(VPU::ExtractImagePatchesOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ExtractImagePatchesUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ExtractImagePatchesUPAOp>(origOp.getLoc(), newOp.getData(), newOp.getOutputBuff(),
                                                     origOp.getSizesAttr(), origOp.getStridesAttr(),
                                                     origOp.getRatesAttr(), origOp.getAutoPadAttr());
}

mlir::Operation* createRTLayer(VPU::AdaptiveAvgPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AdaptiveAvgPoolUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AdaptiveAvgPoolUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getPooledSpatialShape(),
                                                 newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::AdaptiveMaxPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AdaptiveMaxPoolUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AdaptiveMaxPoolUPAOp>(origOp.getLoc(), newOp.getInput(), newOp.getPooledSpatialShape(),
                                                 newOp.getOutputBuff(), newOp.getOutputIndexBuff(),
                                                 origOp.getIndexElementTypeAttr());
}

mlir::Operation* createRTLayer(VPU::SeluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SeluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SeluUPAOp>(origOp.getLoc(), newOp.getData(), newOp.getOutputBuff(),
                                      origOp.getAlphaValueAttr(), origOp.getLambdaValueAttr());
}

mlir::Operation* createRTLayer(VPU::EmbeddingBagOffsetsSumOp origOp, ArrayRef<mlir::Value> allBufs,
                               mlir::OpBuilder& b) {
    VPUIP::EmbeddingBagOffsetsSumUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EmbeddingBagOffsetsSumUPAOp>(origOp.getLoc(), newOp.getInput(), origOp.getIndicesValueAttr(),
                                                        origOp.getOffsetsValueAttr(), origOp.getDefaultIndexValueAttr(),
                                                        origOp.getPerSampleWeightsValueAttr(), newOp.getOutputBuff());
}

mlir::Operation* createRTLayer(VPU::DeformablePSROIPoolingOp origOp, ArrayRef<mlir::Value> allBufs,
                               mlir::OpBuilder& b) {
    VPUIP::DeformablePSROIPoolingUPAOp::Adaptor newOp(allBufs);
    const auto newInp3 = origOp.getInputTransformations() != nullptr ? allBufs[2] : nullptr;
    return b.create<VPUIP::DeformablePSROIPoolingUPAOp>(
            origOp.getLoc(), newOp.getInputScoreMaps(), newOp.getInputRois(), newInp3, newOp.getOutputBuff(),
            origOp.getOutputDimAttr(), origOp.getSpatialScaleAttr(), origOp.getGroupSizeAttr(),
            origOp.getSpatialBinsXAttr(), origOp.getSpatialBinsYAttr(), origOp.getTransStdAttr(),
            origOp.getPartSizeAttr(), origOp.getModeAttr());
}

template <class InLayerOp>
mlir::Operation* LayerRewrite::dispatch(mlir::Operation* origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    return createRTLayer(mlir::cast<InLayerOp>(origOp), allBufs, b);
}

mlir::LogicalResult LayerRewrite::matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    using CreateFunc =
            mlir::Operation* (*)(mlir::Operation * origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder & b);

#define CASE(_OP_)                                   \
    .Case<_OP_>([](mlir::Operation*) -> CreateFunc { \
        return dispatch<_OP_>;                       \
    })

    const CreateFunc createFunc = llvm::TypeSwitch<mlir::Operation*, CreateFunc>(origOp)  //
            CASE(VPU::QuantizeOp)
    CASE(VPU::DequantizeOp)
    CASE(VPU::ConvertOp)
    CASE(VPU::SoftMaxOp)
    CASE(VPU::AvgPoolOp)
    CASE(VPU::MaxPoolOp)
    CASE(VPU::GroupConvolutionOp)
    CASE(VPU::ReLUOp)
    CASE(VPU::SigmoidOp)
    CASE(VPU::SignOp)
    CASE(VPU::ClampOp)
    CASE(VPU::EluOp)
    CASE(VPU::HSwishOp)
    CASE(VPU::FloorOp)
    CASE(VPU::RoundOp)
    CASE(VPU::MishOp)
    CASE(VPU::ErfOp)
    CASE(VPU::BroadcastOp)
    CASE(VPU::ReduceMaxOp)
    CASE(VPU::ReduceMeanOp)
    CASE(VPU::ReduceProdOp)
    CASE(VPU::ReduceSumOp)
    CASE(VPU::ReduceMinOp)
    CASE(VPU::ReduceL1Op)
    CASE(VPU::ReduceL2Op)
    CASE(VPU::ReduceLogicalOrOp)
    CASE(VPU::ReduceLogicalAndOp)
    CASE(VPU::TanOp)
    CASE(VPU::TanhOp)
    CASE(VPU::SinOp)
    CASE(VPU::CosOp)
    CASE(VPU::SqrtOp)
    CASE(VPU::SinhOp)
    CASE(VPU::CoshOp)
    CASE(VPU::AsinhOp)
    CASE(VPU::AcoshOp)
    CASE(VPU::AtanhOp)
    CASE(VPU::LogOp)
    CASE(VPU::GeluOp)
    CASE(VPU::PReluOp)
    CASE(VPU::GatherOp)
    CASE(VPU::GatherNDOp)
    CASE(VPU::YuvToRgbOp)
    CASE(VPU::GatherElementsOp)
    CASE(VPU::ScatterNDUpdateOp)
    CASE(VPU::ScatterUpdateOp)
    CASE(VPU::LeakyReluOp)
    CASE(VPU::AddOp)
    CASE(VPU::MultiplyOp)
    CASE(VPU::AndOp)
    CASE(VPU::DivideOp)
    CASE(VPU::SquaredDifferenceOp)
    CASE(VPU::PowerOp)
    CASE(VPU::FloorModOp)
    CASE(VPU::ModOp)
    CASE(VPU::MinimumOp)
    CASE(VPU::MaximumOp)
    CASE(VPU::SwishOp)
    CASE(VPU::GRNOp)
    CASE(VPU::LRN_IEOp)
    CASE(VPU::PerAxisTileOp)
    CASE(VPU::NegativeOp)
    CASE(VPU::ROIPoolingOp)
    CASE(VPU::PSROIPoolingOp)
    CASE(VPU::ROIAlignOp)
    CASE(VPU::DetectionOutputOp)
    CASE(VPU::ScaleShiftOp)
    CASE(VPU::CTCGreedyDecoderOp)
    CASE(VPU::CTCGreedyDecoderSeqLenOp)
    CASE(VPU::ProposalOp)
    CASE(VPU::RollOp)
    CASE(VPU::PadOp)
    CASE(VPU::ExpOp)
    CASE(VPU::InterpolateOp)
    CASE(VPU::StridedSliceOp)
    CASE(VPU::RegionYoloOp)
    CASE(VPU::ReorgYoloOp)
    CASE(VPU::MVNOp)
    CASE(VPU::DepthToSpaceOp)
    CASE(VPU::MemPermuteOp)
    CASE(VPU::SoftPlusOp)
    CASE(VPU::CeilingOp)
    CASE(VPU::NormalizeIEOp)
    CASE(VPU::CumSumOp)
    CASE(VPU::EqualOp)
    CASE(VPU::SelectOp)
    CASE(VPU::UpsamplingOp)
    CASE(VPU::LessOp)
    CASE(VPU::LessEqualOp)
    CASE(VPU::NotEqualOp)
    CASE(VPU::GreaterOp)
    CASE(VPU::GreaterEqualOp)
    CASE(VPU::SpaceToDepthOp)
    CASE(VPU::LogicalNotOp)
    CASE(VPU::LogicalOrOp)
    CASE(VPU::LogicalXorOp)
    CASE(VPU::AbsOp)
    CASE(VPU::AtanOp)
    CASE(VPU::AsinOp)
    CASE(VPU::AcosOp)
    CASE(VPU::HSigmoidOp)
    CASE(VPU::HardSigmoidOp)
    CASE(VPU::BucketizeOp)
    CASE(VPU::ExtractImagePatchesOp)
    CASE(VPU::AdaptiveAvgPoolOp)
    CASE(VPU::AdaptiveMaxPoolOp)
    CASE(VPU::SeluOp)
    CASE(VPU::EmbeddingBagOffsetsSumOp)
    CASE(VPU::EmbeddingSegmentsSumOp)
    CASE(VPU::DeformablePSROIPoolingOp)
    .Default([](mlir::Operation*) {
        return nullptr;
    });

#undef CASE

    if (createFunc == nullptr) {
        return mlir::failure();
    }

    _log.trace("Found Layer Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(), "Got wrong newOperands size : '{0}'",
                      newOperands.size());

    const auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, origOp->getOpResults());

    SmallVector<mlir::Value> allBufs;
    allBufs.reserve(newOperands.size() + resultBufs.size());
    allBufs.append(newOperands.begin(), newOperands.end());
    allBufs.append(resultBufs.begin(), resultBufs.end());

    const auto newOp = createFunc(origOp, allBufs, rewriter);
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}
}  // namespace vpux
