//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/Timing.h>

// Opset versions supported
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset12.hpp>
#include <openvino/opsets/opset2.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/opsets/opset9.hpp>

#include <ov_ops/nms_ie_internal.hpp>

// Utils
#include "vpux/utils/IE/hash.hpp"

namespace vpux {
namespace IE {

// TODO Get rid of this function (importNetwork), move logic to compiler.cpp
mlir::OwningOpRef<mlir::ModuleOp> importNetwork(mlir::MLIRContext* ctx, const std::shared_ptr<ov::Model>& model,
                                                bool sharedConstants, mlir::TimingScope& rootTiming,
                                                bool enableProfiling, bool stubLayers, bool dynamicShapeToStatic,
                                                vpux::VPU::ArchKind arch, Logger log = Logger::global());

// TODO Move to separate file NGraphPasses
class NGraphPasses final {
public:
    static void runNGraphPasses(const std::shared_ptr<ov::Model>& netGraph, mlir::TimingScope& rootTiming,
                                const vpux::VPU::ArchKind arch);
};

class NGraphImporter final {
public:
    NGraphImporter(mlir::MLIRContext* ctx, std::shared_ptr<const ov::Model> netGraph, bool sharedConstants, Logger log)
            : _ctx(ctx), _netGraph(std::move(netGraph)), _sharedConstants(sharedConstants), _log(log) {
    }

    mlir::func::FuncOp buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName, mlir::TimingScope& rootTiming,
                                     bool stubLayers, bool dynamicShapeToStatic);
    void buildBlockFromRegion(mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* block);
    void buildBlockFromBody(mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* block);
    SmallVector<mlir::Type> getRegionResults();
    SmallVector<mlir::Type> getTensorIteratorRegionResults(int64_t numIter,
                                                           ArrayRef<mlir::Attribute> concatOutputVector,
                                                           ArrayRef<mlir::Attribute> invariantOutputVector);
    static bool isOpSupported(const std::shared_ptr<ov::Node>& op);

private:
    using OrigNode = ov::Node;
    using OrigNodePtr = std::shared_ptr<OrigNode>;
    using NodeOutputMap = std::unordered_map<ov::Output<OrigNode>, mlir::Value>;
    using Callback = void (NGraphImporter::*)(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    static Callback getParser(const std::shared_ptr<ov::Node>& op);

    template <class NodeType>
    void parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    void parseEmpty(mlir::OpBuilder&, const OrigNodePtr&) {
    }

    void parseNodeAsStub(mlir::OpBuilder& builder, const OrigNodePtr& origNode);

    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Constant>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Convert>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ConvertLike>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Softmax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::LogSoftmax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Tile>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Relu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Split>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Power>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Multiply>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Convolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::GroupConvolution>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ConvolutionBackpropData>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::GroupConvolutionBackpropData>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::AvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::MaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveAvgPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::AdaptiveMaxPool>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ShuffleChannels>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::Gather>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::GatherND>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::GatherTree>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toRGB>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::NV12toBGR>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toRGB>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::I420toBGR>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::RandomUniform>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::OneHot>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::BatchNormInference>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::GatherElements>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ScatterNDUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ScatterUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ScatterElementsUpdate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Clamp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Elu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Reshape>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Squeeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::LRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReduceMax>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReduceMean>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReduceLogicalOr>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReduceLogicalAnd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReduceProd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReduceSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReduceMin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReduceL1>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReduceL2>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Unsqueeze>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Minimum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Maximum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Add>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Divide>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::SquaredDifference>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::FloorMod>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Mod>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Proposal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::FakeQuantize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::FakeConvert>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::MatMul>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Tan>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Tanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Cos>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sqrt>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sinh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Cosh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Asinh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Acosh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Atanh>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Log>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Selu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset2::Gelu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Exp>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::HSwish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Floor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Round>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Mish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Erf>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Broadcast>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Bucketize>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Transpose>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Interpolate>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::TopK>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::TopK>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::RegionYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReorgYolo>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::DetectionOutput>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::NormalizeL2>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::CumSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::Eye>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset4::MVN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::MVN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Concat>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::PSROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v9::ROIAlign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::StridedSlice>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::PRelu>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Swish>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::GRN>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Negative>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Sign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::CTCGreedyDecoder>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::CTCGreedyDecoderSeqLen>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Pad>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::LSTMCell>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Subtract>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::LogicalAnd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::LSTMSequence>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Ceiling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Equal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Select>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::NonMaxSuppression>& origNode);
    void parseNode(mlir::OpBuilder& builder,
                   const std::shared_ptr<ov::op::internal::NonMaxSuppressionIEInternal>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::DepthToSpace>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ReverseSequence>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Less>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::LessEqual>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::NotEqual>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::SoftPlus>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Greater>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::GreaterEqual>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseAnd>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseOr>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseXor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::op::v13::BitwiseNot>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::LogicalNot>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::LogicalOr>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::LogicalXor>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::SpaceToDepth>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::BatchToSpace>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::SpaceToBatch>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::ExtractImagePatches>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Abs>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Atan>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Asin>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Acos>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::Roll>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::HSigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::HardSigmoid>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::GridSample>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::EmbeddingBagOffsetsSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::EmbeddingSegmentsSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::EmbeddingBagPackedSum>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::Assign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset3::ReadValue>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::Assign>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset6::ReadValue>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::GRUCell>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::GRUSequence>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::DeformablePSROIPooling>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::TensorIterator>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::DFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::RDFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::IDFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset9::IRDFT>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset8::If>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset1::ShapeOf>& origNode);
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::NonZero>& origNode);

    SmallVector<mlir::Value> getInputs(const OrigNodePtr& node);
    void addOutputs(const OrigNodePtr& node, mlir::Operation* op);
    mlir::Location createLocation(const OrigNodePtr& node);

    static SmallVector<int64_t> importShape(const ov::PartialShape& shape);
    mlir::RankedTensorType importTensor(const ov::PartialShape& shape, const ov::element::Type& elemType);
    mlir::RankedTensorType importConstantTensor(const ov::PartialShape& shape, const ov::element::Type& elemType);
    IE::AutoBroadcastTypeAttr importBroadcastType(ov::op::AutoBroadcastType bType);
    IE::BroadcastTypeAttr importBroadcastMode(ov::op::BroadcastType bType);
    IE::RoundingTypeAttr importRoundingType(ov::op::RoundingType roundingType);
    IE::EpsModeAttr importEpsMode(ov::op::EpsMode val);
    IE::MvnEpsModeAttr importMvnEpsMode(ov::op::MVNEpsMode val);
    IE::TopKModeAttr importTopKMode(ov::op::TopKMode val);
    IE::TopKSortTypeAttr importTopKSortType(ov::op::TopKSortType val);
    IE::GridSampleModeAttr importGridSampleMode(const ov::op::v9::GridSample::InterpolationMode& val);
    IE::GridSamplePaddingModeAttr importGridSamplePaddingMode(const ov::op::v9::GridSample::PaddingMode& val);
    IE::ProposalAttr importProposalAttrs(const ov::op::v0::Proposal::Attributes& val);
    IE::InterpolateAttr importInterpolateAttrs(const ov::opset7::Interpolate::InterpolateAttrs& val);
    IE::DetectionOutputAttr importDetectionOutputAttrs(const ov::op::v0::DetectionOutput::Attributes& val);
    IE::ROIPoolingMethodAttr importROIPoolingMethod(const std::string& method);
    IE::PSROIPoolingModeAttr importPSROIPoolingMode(const std::string& mode);
    IE::ROIAlignMethodAttr importROIAlignMethod(const ov::op::v9::ROIAlign::PoolingMode& mode);
    IE::ROIAlignAlignedMethodAttr importROIAlignAlignedMethod(const ov::op::v9::ROIAlign::AlignedMode& mode);
    IE::PadModeAttr importPadMode(const ov::op::PadMode val);
    IE::RoundModeAttr importRoundMode(const ov::op::v5::Round::RoundMode val);
    IE::RNNSequenceDirectionAttr importRNNSequenceDirection(const ov::op::RecurrentSequenceDirection val);
    IE::BoxEncodingTypeAttr importBoxEncodingType(const int val);
    IE::DepthToSpaceModeAttr importDepthToSpaceMode(const ov::op::v0::DepthToSpace::DepthToSpaceMode val);
    IE::SpaceToDepthModeAttr importSpaceToDepthMode(const ov::op::v0::SpaceToDepth::SpaceToDepthMode val);
    IE::PadTypeAttr importPadType(ov::op::PadType autoPads);
    IE::DeformablePSROIPoolingModeAttr importDeformablePSROIPoolingMode(const std::string& mode);
    IE::DetectionOutputCodeTypeAttr importDetectionOutputCodeType(const std::string& codeType);
    IE::SliceInputPortMapAttr importSliceInputPortMapAttr(
            mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::SliceInputDescription>& desc,
            const mlir::Value input, const int64_t numIter);
    IE::InvariantInputPortMapAttr importInvariantInputPortMapAttr(
            mlir::MLIRContext* ctx,
            const std::shared_ptr<ov::op::util::MultiSubGraphOp::InvariantInputDescription>& desc);
    IE::MergedInputPortMapAttr importMergedInputPortMapAttr(
            mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>& desc);
    IE::ConcatOutputPortMapAttr importConcatOutputPortMapAttr(
            mlir::MLIRContext* ctx,
            const std::shared_ptr<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>& desc);
    IE::InvariantOutputPortMapAttr importBodyOutputPortMapAttr(
            mlir::MLIRContext* ctx, const std::shared_ptr<ov::op::util::MultiSubGraphOp::BodyOutputDescription>& desc);
    mlir::MLIRContext* _ctx = nullptr;
    std::shared_ptr<const ov::Model> _netGraph;
    bool _sharedConstants = false;
    Logger _log;

    NodeOutputMap _importedVals;
};

template <class NodeType>
void NGraphImporter::parseDispatch(mlir::OpBuilder& builder, const OrigNodePtr& origNode) {
    auto targetPtr = std::dynamic_pointer_cast<NodeType>(origNode);
    OPENVINO_ASSERT(targetPtr != nullptr);
    parseNode(builder, targetPtr);
}

}  // namespace IE
}  // namespace vpux
