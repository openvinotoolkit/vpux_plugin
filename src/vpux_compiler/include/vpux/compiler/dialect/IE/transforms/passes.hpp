//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>
#include <type_traits>

namespace vpux {
namespace IE {

//
// AdjustPrecision
//

struct AdjustPrecisionOptions : mlir::PassPipelineOptions<AdjustPrecisionOptions> {
    BoolOption enableConvertPrecisionToFP16{*this, "convert-precision-to-fp16",
                                            llvm::cl::desc("Enable convert-precision-to-fp16 pass"),
                                            llvm::cl::init(true)};

    StrOption computeLayersWithHigherPrecision{*this, "compute-layers-with-higher-precision",
                                               llvm::cl::desc("Enable compute layers with higher precision"),
                                               llvm::cl::init("")};

    AdjustPrecisionOptions() = default;

    template <class OtherOptions>
    explicit AdjustPrecisionOptions(const OtherOptions& options) {
        enableConvertPrecisionToFP16 = options.enableConvertPrecisionToFP16;
        computeLayersWithHigherPrecision = options.computeLayersWithHigherPrecision;
    }
};

void buildAdjustPrecisionPipeline(mlir::OpPassManager& pm, const AdjustPrecisionOptions& options,
                                  Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertPrecisionToFP16Pass(Logger log = Logger::global(),
                                                             StringRef computeLayersWithHigherPrecision = "");
std::unique_ptr<mlir::Pass> createConvertPrecisionToI32Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUseUserPrecisionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustSoftwareOpsPrecisionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustNCEOpsWithI32InputsPass(Logger log = Logger::global());

//
// AdjustLayout
//

struct AdjustLayoutOptions : mlir::PassPipelineOptions<AdjustLayoutOptions> {
    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(true)};

    BoolOption enableForceZMajorConcat{*this, "force-z-major-concat",
                                       llvm::cl::desc("Enable transpose-reorder-concat pass"), llvm::cl::init(true)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    AdjustLayoutOptions() = default;

    template <class OtherOptions>
    explicit AdjustLayoutOptions(const OtherOptions& options) {
        enableOptimizeReorders = options.enableOptimizeReorders;
        enableForceZMajorConcat = options.enableForceZMajorConcat;
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableExperimentalSEPtrsOperations = options.enableExperimentalSEPtrsOperations;
    }
};

std::unique_ptr<mlir::Pass> createAdjustLayoutsPass(const bool seOpsEnabled = false,
                                                    const bool seExperimentalOpsEnabled = false,
                                                    Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseReshapeMvnPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeReordersPass(const bool seOpsEnabled = false,
                                                       const bool seExperimentalOpsEnabled = false,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeReordersAcrossFunctionCallsPass(const bool seOpsEnabled = false,
                                                                          const bool seExperimentalOpsEnabled = false,
                                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUniquifyOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeIdentityPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToMemPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMemPermuteAndExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeNDMemPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTransposeToPermuteCastPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdaptShapesForScaleShiftPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSplitConcatToTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOutlinerPass(const std::string& mode = "naive", Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDebatcherPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAndInitDebatcherPass(StringRef extraArgs, Logger log);
std::unique_ptr<mlir::Pass> createDeDebatcherPass(Logger log = Logger::global());

//
// AdjustForVPU
//

struct AdjustForVPUOptions : mlir::PassPipelineOptions<AdjustForVPUOptions> {
    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", llvm::cl::desc("Enable fuse clamp operations"),
                                         llvm::cl::init(false)};

    AdjustForVPUOptions() = default;

    template <class OtherOptions>
    explicit AdjustForVPUOptions(const OtherOptions& options) {
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableExperimentalSEPtrsOperations = options.enableExperimentalSEPtrsOperations;
        enableFuseClampOperations = options.enableFuseClampOperations;
    }
};

void buildAdjustForVPUPipeline(mlir::OpPassManager& pm, const AdjustForVPUOptions& options,
                               Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertAssignReadValueToReturnsAndInputs(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScalarToTensorPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMinMaxToClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertShapeTo4DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapOperationsPass(const bool seOpsEnabled = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapViewOpAndClampPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapTransposeConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapPadLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNceOpsTo4DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGroupConvToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollConv3dToConv2dPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollTensorIteratorPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPaddingsToFloorModePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertShuffleChannelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLegalizeDilatedConvolutionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveStridedSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertStridedSlice2ConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSliceWithStridePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseActivationOpsPass(const bool enableFuseClamp = false,
                                                        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFusePadOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPadToConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMaxPoolWithActivation(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertTransposedConv2DToConv2DPass(const bool enableSEPTransposedConv = false,
                                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGroupTransposedConvToGroupConvPass(const bool enableSEPTransposedConv = false,
                                                                            Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGroupTransposedConvToTransposedConvPass(
        const bool enableSEPTransposedConv = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertUpsamplingToStridedConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDepth2SpaceToTransposedConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapD2SAndScaleShiftPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDepth2SpaceLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSpace2DepthLayerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertReorderBetweenLayerAndConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateAffineReshapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateShapeCastPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapTransposeWithFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateFqThroughConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapConvertWithTransposeReshapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPerAxisFQConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGatherToSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToScaleShiftPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeLSTMCellPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDilatedConvConvertPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSubtractToAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeOpSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBroadcastToTilePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertGRNToNormalizeL2Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUniquifyBranchesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMVNWithTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMemPermuteAroundOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandMatMulSoftMaxMatMulPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateMemPermuteThroughAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateMemPermuteBeforeOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateMemPermuteThroughSoftMaxPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateOpThroughBatchConcatPass(Logger log = Logger::global());

//
// OptimizeActivations
//

struct OptimizeActivationsOptions : mlir::PassPipelineOptions<OptimizeActivationsOptions> {
    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", ::llvm::cl::desc("Enable FuseClamp operations"),
                                         ::llvm::cl::init(false)};

    OptimizeActivationsOptions() = default;

    template <class OtherOptions>
    explicit OptimizeActivationsOptions(const OtherOptions& options) {
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableFuseClampOperations = options.enableFuseClampOperations;
    }
};

void buildOptimizeActivationsPipeline(mlir::OpPassManager& pm, const OptimizeActivationsOptions& options,
                                      Logger log = Logger::global());

//
// LowPrecision
//

struct LowPrecisionOptions : mlir::PassPipelineOptions<LowPrecisionOptions> {
    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(false)};

    BoolOption enableFuseOutstandingDequant{*this, "fuse-outstanding-dequant",
                                            llvm::cl::desc("Fuse outstanding dequantize after NCE task"),
                                            llvm::cl::init(false)};

    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(true)};

    BoolOption enablePropagateQuantDequant{*this, "propagate-quant-dequant",
                                           llvm::cl::desc("Enable Propagate Quantize Dequantize pass"),
                                           llvm::cl::init(true)};

    BoolOption enableFP16ToU8MixedMode{
            *this, "enable-fp16-to-u8-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with FP16 input and quantized output"),
            llvm::cl::init(false)};

    BoolOption enableFloatInQuantWeightsMixedMode{
            *this, "enable-float-in-quant-weights-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with float input and quantized weights"),
            llvm::cl::init(true)};

    BoolOption enableAlignScales{*this, "enable-align-scales", llvm::cl::desc("Enable align scales"),
                                 llvm::cl::init(true)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableAdjustNonZeroFakeQuant{*this, "adjust-non-zero-fake-quant",
                                            llvm::cl::desc("Enable adjust non zero fake quant"), llvm::cl::init(true)};

    BoolOption enableConvolutionMixedPrecisionDecomposition{
            *this, "enable-convolution-mixed-precision-decomposition",
            llvm::cl::desc("Enable mixed precision decomposition for convolution"), llvm::cl::init(false)};

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(false)};

    LowPrecisionOptions() = default;

    template <class OtherOptions>
    explicit LowPrecisionOptions(const OtherOptions& options) {
        enableQuantDequantRemoval = options.enableQuantDequantRemoval;
        enableFuseOutstandingDequant = options.enableFuseOutstandingDequant;
        enableSwapTransposeWithFQ = options.enableSwapTransposeWithFQ;
        enablePropagateQuantDequant = options.enablePropagateQuantDequant;
        enableFP16ToU8MixedMode = options.enableFP16ToU8MixedMode;
        enableFloatInQuantWeightsMixedMode = options.enableFloatInQuantWeightsMixedMode;
        enableAlignScales = options.enableAlignScales;
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableExperimentalSEPtrsOperations = options.enableExperimentalSEPtrsOperations;
        enableAdjustNonZeroFakeQuant = options.enableAdjustNonZeroFakeQuant;
        enableConvolutionMixedPrecisionDecomposition = options.enableConvolutionMixedPrecisionDecomposition;
        enableWDBlockArgumentInput = options.enableWDBlockArgumentInput;
    }
};

struct TransformOptions : mlir::PassPipelineOptions<TransformOptions> {
    TransformOptions() = default;

    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(false)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    template <
            class OtherOptions,
            typename = std::enable_if_t<std::is_base_of<mlir::PassPipelineOptions<OtherOptions>, OtherOptions>::value>>
    explicit TransformOptions(const OtherOptions& options) {
        enableConvertFCToConv = options.enableConvertFCToConv;
        enableWDBlockArgumentInput = options.enableWDBlockArgumentInput;
        enableGroupedMatMul = options.enableGroupedMatMul;
    }
};

struct LowPrecisionTransformOptions : mlir::PassPipelineOptions<LowPrecisionTransformOptions> {
    LowPrecisionTransformOptions() = default;

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(false)};

    template <class OtherOptions>
    explicit LowPrecisionTransformOptions(const OtherOptions& options) {
        enableWDBlockArgumentInput = options.enableWDBlockArgumentInput;
    }
};

struct ExpandActivationChannelsOptions : mlir::PassPipelineOptions<ExpandActivationChannelsOptions> {
    ExpandActivationChannelsOptions() = default;

    BoolOption enableExpandActivationChannels{*this, "expand-activation-channels",
                                              llvm::cl::desc("Enable expand-activation-channels pass"),
                                              llvm::cl::init(true)};
    BoolOption enableAdjustConvShapePass{*this, "adjust-convolution-shape",
                                         llvm::cl::desc("Enable adjust-convolution-shape pass"), llvm::cl::init(true)};

    BoolOption enableOptimizeSliceExpand{*this, "optimize-slice-expand",
                                         llvm::cl::desc("Enable optimize-slice-expand pass"), llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantizeExpand{*this, "fuse-permute-quantize-expand",
                                               llvm::cl::desc("Enable fuse-permute-quantize-expand pass"),
                                               llvm::cl::init(true)};
    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    template <class OtherOptions>
    explicit ExpandActivationChannelsOptions(const OtherOptions& options) {
        enableExpandActivationChannels = options.enableExpandActivationChannels;
        enableAdjustConvShapePass = options.enableAdjustConvShapePass;
        enableOptimizeSliceExpand = options.enableOptimizeSliceExpand;
        enableFusePermuteQuantizeExpand = options.enableFusePermuteQuantizeExpand;
        enableSEPtrsOperations = options.enableSEPtrsOperations;
        enableExperimentalSEPtrsOperations = options.enableExperimentalSEPtrsOperations;
    }
};

void buildScaleShiftProcessingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildOperationConversionPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertMVN6ToMVN1Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleU16FakeQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseFQAndMulPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEltwiseFakeQuantizeFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWeightsDequantizeToFakeQuantizePass();
std::unique_ptr<mlir::Pass> createWeightsDequantizeToFakeQuantizePass(const IE::LowPrecisionTransformOptions& options,
                                                                      Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFoldReLUBeforeFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapFakeQuantWithReshapeAndStridedSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveScatterUpdateByTransposePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleFakeQuantHasNegativeScalesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAlignScalesPass(const bool seOpsEnabled = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateQuantizeDequantizePass(const bool seOpsEnabled = false,
                                                                  Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDequantizeConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseQuantizedOpsPass(const bool seOpsEnabled = false,
                                                       const bool enableExperimentalSEPtrsOperations = false,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveQuantDequantSeqPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeUnalignedQDQSeqPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsToU8Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsToI4Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseConvertWithQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToDequantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToDequantizePass(const IE::LowPrecisionOptions& options,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertQuantizeOpsToNceOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollFakeQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollFullyConnectedPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseScalesToAccumulatePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMultiplyWithMatmulPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReshapeMatMulInputsPass(const bool enableGroupedMatMul = false,
                                                          Logger log = Logger::global());

//
// Legalization for NCE
//

std::unique_ptr<mlir::Pass> createAdjustGroupConvShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustConvolutionShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustConvolutionWeightsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBatchedLayerTo1NPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustConvolutionInputShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMaxPoolInputShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeAvgPoolWithUnalignedChannelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMatMulInputsTo2dPass(const bool enableGroupedMatMul = false,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDivideToMultiplyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMatMulToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertConvBackpropDataToTransposedConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertFCToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAvgPoolToDWConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustScaleShiftForDWConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScaleShiftToDWPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToSpatialOpPass(const bool m2iEnabled = false,
                                                         const bool seExperimentalOpsEnabled = false,
                                                         Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNearestToBroadCastOrStridedConcatPass(const bool interpolateAsSEOp = false,
                                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitBilinerIntoHAndWPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertBilinearToStridedConcatAndConvPass(const bool interpolateAsSEOp = false,
                                                                            Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScatterNDUpdateToStridedConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitConvWithMultipleFQPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargeStridesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleAsymmetricStridesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandActivationChannelsPass(const bool seOpsEnabled = false,
                                                               const bool seExperimentalOpsEnabled = false,
                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargeKernelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReduceSumToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReduceToPoolingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollReduceMinAllAxesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleExcludePadForAvgPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSquaredDiffToSubAndPowerPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPowerToMultPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandActivationWidthPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFusePermuteQuantizePass(const bool dpuOnly = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustInputShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMovePermutePostEltwisePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertExtractImagePatchesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBroadcastInputForAddPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReorderToPermuteQuantizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseMemPermutePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveViewLikeOpsChainPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleLargePadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createNormalizeL2FusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createM2IBatchNormFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertMemPermuteToPoolPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLogOpOptimizationsPass();
std::unique_ptr<mlir::Pass> createAdjustNonZeroFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseConvWithSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMVNFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeGroupConvConcatPass(Logger log = Logger::global());

//
// Generic Optimizations
//

std::unique_ptr<mlir::Pass> createUpstreamSlicePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertReflectPadToSliceAndConcatPass(const bool enableSEPPad = false,
                                                                        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertExpandToConvPass(Logger log = Logger::global());

//
// DefaultHWOptions(for all devices)
//

struct DefaultHWOptionsDialectBase : public virtual vpux::DefaultHWOptionsBase {
    BoolOption enableConvertAvgPoolToDWConv{*this, "convert-avg-pool-to-dw-conv",
                                            llvm::cl::desc("Enable convert-avg-pool-to-dw-conv pass"),
                                            llvm::cl::init(true)};

    BoolOption enableOptimizeScaleShiftToDWConv{*this, "optimize-scale-shift-to-depthwise",
                                                llvm::cl::desc("Enable optimize-scale-shift-to-depthwise pass"),
                                                llvm::cl::init(true)};

    BoolOption enableSplitConvWithMultipleFQ{*this, "split-conv-with-multiple-fq",
                                             llvm::cl::desc("Enable split-conv-with-multiple-fq pass"),
                                             llvm::cl::init(true)};

    BoolOption enableHandleLargeKernel{*this, "handle-large-kernel", llvm::cl::desc("Enable handle-large-kernel pass"),
                                       llvm::cl::init(true)};

    BoolOption enableHandleLargeStrides{*this, "handle-large-strides",
                                        llvm::cl::desc("Enable handle-large-strides pass"), llvm::cl::init(true)};

    BoolOption enableHandleLargePads{*this, "handle-large-pads", llvm::cl::desc("Enable handle-large-pads pass"),
                                     llvm::cl::init(true)};

    BoolOption enableHandleAsymmetricStrides{*this, "handle-asymmetric-strides",
                                             llvm::cl::desc("Enable handle-asymmetric-strides pass"),
                                             llvm::cl::init(true)};

    BoolOption enableBilinearInterpolateOnDPU{*this, "map-interpolate-on-dpu",
                                              llvm::cl::desc("Enable map-interpolate-on-dpu pass"),
                                              llvm::cl::init(true)};

    BoolOption enableSplitBilinerIntoHAndW{*this, "split-bilinear-into-H-and-W",
                                           llvm::cl::desc("Enable split-bilinear-into-H-and-W pass"),
                                           llvm::cl::init(false)};

    BoolOption skipUnrollBatch{*this, "skip-unroll-batch", llvm::cl::desc("Skip unroll on batch dimension"),
                               llvm::cl::init(false)};

    BoolOption enableUpstreamSlice{*this, "upstream-slice", llvm::cl::desc("Enable upstream-slice pipeline building"),
                                   llvm::cl::init(true)};

    BoolOption enableExpandActivationChannels{*this, "expand-activation-channels",
                                              llvm::cl::desc("Enable expand-activation-channels pass"),
                                              llvm::cl::init(true)};

    BoolOption enableAdjustConvShapePass{*this, "adjust-convolution-shape",
                                         llvm::cl::desc("Enable adjust-convolution-shape pass"), llvm::cl::init(true)};

    BoolOption enableOptimizeSliceExpand{*this, "optimize-slice-expand",
                                         llvm::cl::desc("Enable optimize-slice-expand pass"), llvm::cl::init(true)};

    BoolOption enableOptimizeSliceWithStride{*this, "optimize-slice-with-stride",
                                             llvm::cl::desc("Enable optimize-slice-with-stride pass"),
                                             llvm::cl::init(true)};

    BoolOption enableConvertExpandToConvPass{*this, "convert-expand-to-conv",
                                             llvm::cl::desc("Enable convert-expand-to-conv pass"),
                                             llvm::cl::init(true)};

    BoolOption logOpOptimizations{*this, "log-op-optimizations",
                                  llvm::cl::desc("Log potential operation optimizations that can be done"),
                                  llvm::cl::init(false)};

    // AdjustPrecisionOptions
    BoolOption enableConvertPrecisionToFP16{*this, "convert-precision-to-fp16",
                                            llvm::cl::desc("Enable convert-precision-to-fp16 pass"),
                                            llvm::cl::init(true)};

    // TransformOptions
    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(false)};

    // AdjustLayoutOptions

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(true)};

    BoolOption enableForceZMajorConcat{*this, "force-z-major-concat",
                                       llvm::cl::desc("Enable transpose-reorder-concat pass"), llvm::cl::init(true)};

    // LowPrecisionOptions
    BoolOption enableLowPrecision{*this, "low-precision", llvm::cl::desc("Enable low-precision pipeline building"),
                                  llvm::cl::init(true)};

    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(false)};

    BoolOption enableFuseOutstandingDequant{*this, "fuse-outstanding-dequant",
                                            llvm::cl::desc("Fuse outstanding dequantize after NCE task"),
                                            llvm::cl::init(false)};

    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(true)};

    BoolOption enablePropagateQuantDequant{*this, "propagate-quant-dequant",
                                           llvm::cl::desc("Enable Propagate Quantize Dequantize pass"),
                                           llvm::cl::init(true)};

    BoolOption enableAlignScales{*this, "enable-align-scales", llvm::cl::desc("Enable align scales"),
                                 llvm::cl::init(true)};

    BoolOption enableAdjustNonZeroFakeQuant{*this, "adjust-non-zero-fake-quant",
                                            llvm::cl::desc("Enable adjust non zero fake quant"), llvm::cl::init(true)};

    // LowPrecisionOptions(only for 37XX)
    BoolOption enableFP16ToU8MixedMode{
            *this, "enable-fp16-to-u8-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with FP16 input and quantized output"),
            llvm::cl::init(false)};

    BoolOption enableFloatInQuantWeightsMixedMode{
            *this, "enable-float-in-quant-weights-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with float input and quantized weights"),
            llvm::cl::init(true)};

    // LowPrecisionOptions(37XX+)
    BoolOption enableConvolutionMixedPrecisionDecomposition{
            *this, "enable-convolution-mixed-precision-decomposition",
            llvm::cl::desc("Enable mixed precision decomposition for convolution"), llvm::cl::init(false)};

    // Common
    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", llvm::cl::desc("Enable fuse clamp operations"),
                                         llvm::cl::init(false)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};
};

//
// Registration
//

void registerIEPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/IE/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/IE/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace IE
}  // namespace vpux
