//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux;

namespace {

//
// ConvolutionDimsOrderOpModelForSW
//

class ConvolutionDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<ConvolutionDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        auto expectedFilterLayout = mlir::isa<IE::GroupConvolutionOp>(op) ? DimsOrder::OIYX : DimsOrder::YXOI;
        info.setInput(0, DimsOrder::NCHW);
        info.setInput(1, expectedFilterLayout);
        info.setOutput(0, DimsOrder::NCHW);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
        if (layer == nullptr) {
            return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
        }
        auto expectedFilterLayout = mlir::isa<VPU::GroupConvolutionOp>(op) ? DimsOrder::OIYX : DimsOrder::YXOI;
        const auto input = layer.getInputs()[0];
        const auto filter = layer.getInputs()[1];
        const auto output = layer.getOutputs()[0];

        const auto inOrder = DimsOrder::fromValue(input);
        const auto filterOrder = DimsOrder::fromValue(filter);
        const auto outOrder = DimsOrder::fromValue(output);

        if (inOrder != DimsOrder::NCHW) {
            return errorAt(op->getLoc(), "Operation input order is not as expected. inL={0}, expectedInL=NCHW",
                           inOrder);
        }
        if (outOrder != DimsOrder::NCHW) {
            return errorAt(op->getLoc(), "Operation output order is not as expected. outL={0}, expectedOutL=NCHW",
                           outOrder);
        }
        if (filterOrder != expectedFilterLayout) {
            return errorAt(op->getLoc(), "Operation filter order is not as expected. filterL={0}, expectedFilterL={1}",
                           filterOrder, expectedFilterLayout);
        }
        return mlir::success();
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// NCEConvolutionWithCMajorDimsOrderOpModelForHW
//

class NCEConvolutionWithCMajorDimsOrderOpModelForHW final :
        public IE::LayoutInfoOpInterface::FallbackModel<NCEConvolutionWithCMajorDimsOrderOpModelForHW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        const auto arch = VPU::getArch(op);

        const auto canUseCMajor = vpux::VPU::NCEInvariant::isChannelMajorCompatible(
                arch, op->getOperand(0).getType().cast<vpux::NDTypeInterface>());

        if (info.getInput(0) == DimsOrder::NCHW && canUseCMajor) {
            info.setInput(0, DimsOrder::NCHW);
        } else {
            info.setInput(0, DimsOrder::NHWC);
        }

        info.setInput(1, DimsOrder::OYXI);

        info.setOutput(0, DimsOrder::NHWC);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
        if (layer == nullptr) {
            return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
        }

        const auto canUseCMajor = vpux::VPU::NCEInvariant::isChannelMajorCompatible(
                VPU::ArchKind::NPU30XX, op->getOperand(0).getType().cast<vpux::NDTypeInterface>());

        const auto input = layer.getInputs()[0];
        const auto filter = layer.getInputs()[1];
        const auto output = layer.getOutputs()[0];

        const auto inOrder = DimsOrder::fromValue(input);
        const auto filterOrder = DimsOrder::fromValue(filter);
        const auto outOrder = DimsOrder::fromValue(output);

        if (inOrder != DimsOrder::NHWC && !(inOrder == DimsOrder::NCHW && canUseCMajor)) {
            return errorAt(op->getLoc(), "Operation input order is not as expected. inL={0}, expectedInL=NHWC",
                           inOrder);
        }
        if (outOrder != DimsOrder::NHWC) {
            return errorAt(op->getLoc(), "Operation output order is not as expected. outL={0}, expectedOutL=NHWC",
                           outOrder);
        }
        if (filterOrder != DimsOrder::OYXI) {
            return errorAt(op->getLoc(), "Operation filter order is not as expected. filterL={0}, expectedFilterL=OYXI",
                           filterOrder);
        }
        return mlir::success();
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// redirectLayoutOpInterfacesForVPU
//

void redirectLayoutOpInterfacesForVPU(mlir::DialectRegistry& registry) {
    // Tracking number [E#84955]
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::ConvertOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SoftMaxOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SwishOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::PReluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::AddOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::MultiplyOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);

        VPU::AvgPoolOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::AdaptiveAvgPoolOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::AdaptiveMaxPoolOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        // TODO: E#115645 support HW/SW option for SEP VPU ops
        VPU::RollOp::attachInterface<vpux::VPU::RollDimsOrderOpModelForSW>(*ctx);
        VPU::TanOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::SinOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::CosOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LogOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::SeluOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::SelectOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LeakyReluOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::HardSigmoidOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GridSampleOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::BucketizeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::TileOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::PerAxisTileOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::NegativeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::PadOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LSTMCellOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::NormalizeL2Op::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::NormalizeIEOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::CumSumOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ReverseSequenceOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::SoftPlusOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ScatterNDUpdateOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::DynamicQuantizeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::DivideOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::SquaredDifferenceOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::PowerOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::FloorModOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ModOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::MinimumOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::MaximumOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LogicalOrOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LogicalXorOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::BitwiseAndOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::BitwiseOrOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::BitwiseXorOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::BitwiseNotOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LessOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LessEqualOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::NotEqualOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GreaterOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GreaterEqualOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::EqualOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LogicalNotOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ScatterUpdateOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ScaleShiftOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GatherTreeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::FullyConnectedOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::SubtractOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::AndOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);

        VPU::OneHotOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::DFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::RDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::IDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::IRDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::RDFTUncutOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::IRDFTLastAxisOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GatherNDOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::CTCGreedyDecoderOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::NonMaxSuppressionOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GatherOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::RandomUniformOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::EyeOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::CTCGreedyDecoderSeqLenOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::NonZeroOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);

        VPU::SigmoidOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::ClampOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::ReLUOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::EluOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::HSwishOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::ErfOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::MishOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::FloorOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::RoundOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::TanhOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::SqrtOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::SinhOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::CoshOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::AsinhOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::AcoshOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::AbsOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::AtanOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::AsinOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::AcosOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::AtanhOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::GeluOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::ExpOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::LRN_IEOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::UpsamplingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::ROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::PSROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::CeilingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::DeformablePSROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(
                *ctx);
        VPU::StridedSliceOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::MaxPoolOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);

        VPU::DetectionOutputOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::EmbeddingBagOffsetsSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::EmbeddingSegmentsSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::EmbeddingBagPackedSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::BroadcastOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::ProposalOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::ReorgYoloOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::YuvToRgbOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::LSTMSequenceOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::GRUSequenceOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::MemPermuteOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::ScatterElementsUpdateOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);

        VPU::FakeQuantizeOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NHWC>(*ctx);
        VPU::GRNOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NHWC>(*ctx);

        VPU::HSigmoidOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NC_CHW_HWC_NCHW_NHWC>(*ctx);

        VPU::ExtractImagePatchesOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW>(*ctx);

        VPU::MVNOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NCWH_NHWC_NWHC>(*ctx);
        VPU::MVN6Op::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NCWH_NHWC_NWHC>(*ctx);

        VPU::DepthToSpaceOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NHWC>(*ctx);

        VPU::NCEMaxPoolOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NHWC>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NHWC>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NHWC>(*ctx);

        VPU::NCEEltwiseOp::attachInterface<vpux::VPU::SameMultipleInOutDimsOrderOpModelForHW_NHWC>(*ctx);

        VPU::ReduceMaxOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceMeanOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceLogicalOrOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceLogicalAndOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceProdOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceSumOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceMinOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceL1Op::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceL2Op::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);

        VPU::AffineReshapeOp::attachInterface<vpux::VPU::AffineReshapeDimsOrderOpModelForSW>(*ctx);

        VPU::QuantizeOp::attachInterface<vpux::VPU::QuantizeDimsOrderOpModelForSW>(*ctx);
        VPU::DequantizeOp::attachInterface<vpux::VPU::DequantizeDimsOrderOpModelForSW>(*ctx);

        VPU::SqueezeOp::attachInterface<vpux::VPU::SqueezeUnsqueezeDimsOrderOpModelForSW>(*ctx);
        VPU::UnsqueezeOp::attachInterface<vpux::VPU::SqueezeUnsqueezeDimsOrderOpModelForSW>(*ctx);

        VPU::RegionYoloOp::attachInterface<vpux::VPU::RegionYoloDimsOrderOpModelForSW>(*ctx);

        VPU::TopKOp::attachInterface<vpux::VPU::TopKSameInOutDimsOrderOpModelForSW>(*ctx);

        VPU::ConvolutionOp::attachInterface<ConvolutionDimsOrderOpModelForSW>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<ConvolutionDimsOrderOpModelForSW>(*ctx);

        VPU::NCEConvolutionOp::attachInterface<NCEConvolutionWithCMajorDimsOrderOpModelForHW>(*ctx);

        VPU::NCEDepthConvolutionOp::attachInterface<vpux::VPU::NCEConvolutionDimsOrderOpModelForHW>(*ctx);

        VPU::InterpolateOp::attachInterface<vpux::VPU::InterpolateDimsOrderOpModelForSW>(*ctx);
    });
}

//
// redirectLayoutOpInterfacesForIE
//

void redirectLayoutOpInterfacesForIE(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvertOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SoftMaxOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SpaceToDepthOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SwishOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::PReluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);

        IE::AdaptiveAvgPoolOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::AdaptiveMaxPoolOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::TanOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::SinOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::CosOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LogOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::SeluOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::SelectOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LeakyReluOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::HardSigmoidOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::GridSampleOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::BucketizeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::TileOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::PerAxisTileOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::NegativeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LSTMCellOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::NormalizeL2Op::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::NormalizeIEOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::CumSumOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::ReverseSequenceOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::SoftPlusOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::ScatterNDUpdateOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::DynamicQuantizeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::DivideOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::SquaredDifferenceOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::PowerOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::FloorModOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::ModOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::MinimumOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::MaximumOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LogicalOrOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LogicalXorOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::BitwiseAndOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::BitwiseOrOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::BitwiseXorOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::BitwiseNotOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LessOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LessEqualOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::NotEqualOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::GreaterOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::GreaterEqualOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::EqualOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LogicalNotOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::ScatterUpdateOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::GatherTreeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::FullyConnectedOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);

        IE::OneHotOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::DFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::RDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::IDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::IRDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::GatherNDOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::CTCGreedyDecoderOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::NonMaxSuppressionOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::GatherOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::RandomUniformOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::EyeOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::CTCGreedyDecoderSeqLenOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::NonZeroOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);

        IE::SigmoidOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::ClampOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::ReLUOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::EluOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::HSwishOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::ErfOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::MishOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::FloorOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::RoundOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::TanhOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::SqrtOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::SinhOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::CoshOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::AsinhOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::AcoshOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::AbsOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::AtanOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::AsinOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::AcosOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::AtanhOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::GeluOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::ExpOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::LRN_IEOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::UpsamplingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::ROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::PSROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::CeilingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::DeformablePSROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(
                *ctx);
        IE::StridedSliceOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::ScaleShiftOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);

        IE::DetectionOutputOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::EmbeddingBagOffsetsSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::EmbeddingSegmentsSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::EmbeddingBagPackedSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::BroadcastOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::ProposalOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::ReorgYoloOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::YuvToRgbOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::LSTMSequenceOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::GRUSequenceOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::ReorderOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::MemPermuteOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::ScatterElementsUpdateOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);

        IE::FakeQuantizeOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NHWC>(*ctx);
        IE::GRNOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NHWC>(*ctx);

        IE::HSigmoidOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NC_CHW_HWC_NCHW_NHWC>(*ctx);

        IE::ExtractImagePatchesOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW>(*ctx);

        IE::MVNOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NCWH_NHWC_NWHC>(*ctx);
        IE::MVN6Op::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NCWH_NHWC_NWHC>(*ctx);

        IE::DepthToSpaceOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NHWC>(*ctx);

        IE::ReduceMaxOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceMeanOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceLogicalOrOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceLogicalAndOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceProdOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceSumOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceMinOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceL1Op::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceL2Op::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);

        IE::AffineReshapeOp::attachInterface<vpux::VPU::AffineReshapeDimsOrderOpModelForSW>(*ctx);

        IE::QuantizeOp::attachInterface<vpux::VPU::QuantizeDimsOrderOpModelForSW>(*ctx);
        IE::DequantizeOp::attachInterface<vpux::VPU::DequantizeDimsOrderOpModelForSW>(*ctx);

        IE::SqueezeOp::attachInterface<vpux::VPU::SqueezeUnsqueezeDimsOrderOpModelForSW>(*ctx);
        IE::UnsqueezeOp::attachInterface<vpux::VPU::SqueezeUnsqueezeDimsOrderOpModelForSW>(*ctx);

        IE::RegionYoloOp::attachInterface<vpux::VPU::RegionYoloDimsOrderOpModelForSW>(*ctx);

        IE::TopKOp::attachInterface<vpux::VPU::TopKSameInOutDimsOrderOpModelForSW>(*ctx);

        // clang-format off
        IE::ConvolutionOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::ConvolutionOp,
                /*FallbackSWImplOpType=*/ConvolutionDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/NCEConvolutionWithCMajorDimsOrderOpModelForHW>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::GroupConvolutionOp,
                /*FallbackSWImplOpType=*/ConvolutionDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::NCEConvolutionDimsOrderOpModelForHW>>(*ctx);
        IE::MaxPoolOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::MaxPoolOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::AvgPoolOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::AvgPoolOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::AddOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::AddOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameAnyDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameMultipleInOutDimsOrderOpModelForHW_NHWC>>(*ctx);
        IE::MultiplyOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::MultiplyOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameAnyDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameMultipleInOutDimsOrderOpModelForHW_NHWC>>(*ctx);
        IE::SubtractOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::SubtractOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameMultipleInOutDimsOrderOpModelForHW_NHWC>>(*ctx);
        IE::AndOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::AndOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameMultipleInOutDimsOrderOpModelForHW_NHWC>>(*ctx);
        IE::InterpolateOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::InterpolateOp,
                /*FallbackSWImplOpType=*/vpux::VPU::InterpolateDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::TransposedConvolutionOp,
                /*FallbackSWImplOpType=*/ConvolutionDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::NCEConvolutionDimsOrderOpModelForHW>>(*ctx);
        IE::PadOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::PadOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::RollOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::RollOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::RollDimsOrderOpModelForHW>>(*ctx);
        // clang-format on
    });
}

}  // namespace

//
// registerLayoutInfoOpInterfaces
//

void vpux::VPU::arch30xx::registerLayoutInfoOpInterfaces(mlir::DialectRegistry& registry) {
    redirectLayoutOpInterfacesForVPU(registry);
    redirectLayoutOpInterfacesForIE(registry);
}
