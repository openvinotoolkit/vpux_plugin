//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/developer_build_utils.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <type_traits>

namespace vpux {

//
// ReferenceSWMode
//

template <typename T>
struct ReferenceSWOptions : mlir::PassPipelineOptions<T> {
    BoolOption enableVerifiers{*this, "enable-verifiers", llvm::cl::desc("Enable verifiers execution after each pass"),
                               llvm::cl::init(isDeveloperBuild())};

    BoolOption enableMemoryUsageCollector{*this, "enable-memory-usage-collector",
                                          llvm::cl::desc("Enable peak memory usage instrumentation after each pass"),
                                          llvm::cl::init(isDeveloperBuild())};

    BoolOption enableFunctionStatisticsInstrumentation{
            *this, "enable-function-statistics-instrumentation",
            llvm::cl::desc("Enable printing statistics for functions after each pass"), llvm::cl::init(false)};

    // InitCompiler
    IntOption revisionID{*this, "revision-id", ::llvm::cl::desc("[Optional] Revision ID of the platform")};
    IntOption numberOfDPUGroups{*this, "num-of-dpu-groups",
                                ::llvm::cl::desc("[Optional] Number of available DPU groups")};
    IntOption numberOfDMAPorts{*this, "num-of-dma-ports", ::llvm::cl::desc("[Optional] Number of available DMA ports")};
    IntOption availableCMXMemory{*this, "available-cmx-memory", ::llvm::cl::desc("[Optional] Available CMX memory")};
    BoolOption allowCustomValues{*this, "allow-custom-values",
                                 ::llvm::cl::desc("[Optional] Allows keep predefined values in IR")};

    BoolOption enableDummyOpReplacement{*this, "dummy-op-replacement",
                                        llvm::cl::desc("Replace unsupported SW Kernel ops with Dummy ones"),
                                        llvm::cl::init(false)};

    BoolOption constantFoldingInBackground{*this, "constant-folding-in-background",
                                           llvm::cl::desc("Fold constants in background threads"),
                                           llvm::cl::init(false)};

    IntOption constantFoldingInBackgroundNumThreads{
            *this, "constant-folding-in-background-num-threads",
            llvm::cl::desc("Number of background threads to use for constant folding in background. Ignored if "
                           "`constant-folding-in-background` is disabled."),
            llvm::cl::init(1)};

    BoolOption constantFoldingInBackgroundCollectStatistics{
            *this, "constant-folding-in-background-collect-statistics",
            llvm::cl::desc("Toggle for the collection of statistics when folding constants in background. Ignored if "
                           "`constant-folding-in-background` is disabled."),
            llvm::cl::init(false)};

    IntOption constantFoldingInBackgroundMemoryUsageLimit{
            *this, "constant-folding-in-background-memory-usage-limit",
            llvm::cl::desc("Fold constants in background memory usage limit (in MB)"), llvm::cl::init(3 * 1024)};

    DoubleOption constantFoldingInBackgroundCacheCleanThreshold{
            *this, "constant-folding-in-background-cache-clean-threshold",
            llvm::cl::desc("Cache will be cleaned to this threshold when reach the memory usage limit"),
            llvm::cl::init(0.8)};

    BoolOption wlmRollback{
            *this, "wlm-rollback",
            llvm::cl::desc("When compilation with WLM fails, automatically switches to WLM-disabled pipeline"),
            llvm::cl::init(true)};

    IntOption optimizationLevel{*this, "optimization-level",
                                llvm::cl::desc("Set compilation optimization level, enabled starting from NPU4."
                                               "Possible values: 0 - optimization for compilation time,"
                                               "1 - optimization for execution time (default),"
                                               "2 - high optimization for execution time"),
                                llvm::cl::init(1)};

    StrOption performanceHintOverride{*this, "performance-hint-override",
                                      llvm::cl::desc("Set performance hint for compiler to set up number of tiles."
                                                     "Possible values: latency, efficiency (default)"),
                                      llvm::cl::init("efficiency")};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};
    BoolOption enableSWProfiling{*this, "sw-profiling", llvm::cl::desc("Enable SW task profiling"),
                                 llvm::cl::init(true)};

    BoolOption enableMergeFakeQuant{*this, "merge-fake-quant", llvm::cl::desc("Enable merge-fake-quant pass"),
                                    llvm::cl::init(true)};

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(false)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingODU{*this, "enable-auto-padding-odu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingIDU{*this, "enable-auto-padding-idu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    // TODO: find a better way to expose enableSEPtrsOperations to the common AdjustLayouts pipeline
    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", llvm::cl::desc("Enable fuse clamp operations"),
                                         llvm::cl::init(false)};

    BoolOption enableConvertPrecisionToFP16{*this, "convert-precision-to-fp16",
                                            llvm::cl::desc("Enable convert-precision-to-fp16 pass"),
                                            llvm::cl::init(true)};

    BoolOption enableControlGraphSplit{*this, "enable-control-graph-split",
                                       llvm::cl::desc("Enable split of control graph to simplify barrier scheduling"),
                                       llvm::cl::init(true)};
    IntOption controlGraphSplitBlockSize{
            *this, "control-graph-split-block-size",
            llvm::cl::desc("Maximal number of tasks in each block that control graph will be split into. Used to "
                           "reduce memory consumption of barrier legalization pipeline for big models. Memory usage is "
                           "roughly (control-graph-split-block-size)^2/8"),
            llvm::cl::init(CONTROL_GRAPH_SPLIT_BLOCK_SIZE)};

    StrOption computeLayersWithHigherPrecision{
            *this, "compute-layers-with-higher-precision",
            llvm::cl::desc("Enable compute layers with higher precision for the specified layer types"),
            llvm::cl::init("")};

    BoolOption enableSimpleSchedule{*this, "simple-schedule", llvm::cl::desc("Enable schedule simplification"),
                                    llvm::cl::init(true)};

    BoolOption shareWaitAndUpdateBarriers{*this, "share-wait-and-update-barriers",
                                          llvm::cl::desc("Share wait and update barriers"), llvm::cl::init(true)};

    BoolOption reduceParallelControlFlows{*this, "reduce-parallel-control-flows",
                                          llvm::cl::desc("Reduce parallel overlapping control flows where possible"),
                                          llvm::cl::init(true)};
    BoolOption enableColorBinPhysicalBarrierAssignment{
            *this, "enable-color-bin-physical-barrier-assignment",
            llvm::cl::desc("Enable physical barrier assignment optimization"), llvm::cl::init(false)};

    BoolOption enableSWKernelPrefetchingReserveMem{
            *this, "enable-sw-kernel-prefetching-reserve-mem",
            ::llvm::cl::desc("Reserve memory at the end of CMX for SW Kernel data prefetching"),
            ::llvm::cl::init(true)};

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(false)};

    BoolOption enableU16FQToScaleShiftConversion{*this, "enable-u16-fake-quantize-to-scale-shift-conversion",
                                                 llvm::cl::desc("Enable u16 fake quantize to scale shift conversion"),
                                                 llvm::cl::init(false)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    BoolOption accumulateMatmulWithDPU{*this, "accumulate-matmul-with-dpu",
                                       llvm::cl::desc("Accumulate unrolled Matmul results with DPU"),
                                       llvm::cl::init(false)};

    BoolOption fuseScalesToAccumulate{
            *this, "fuse-scales-to-accumulate",
            llvm::cl::desc("Enable scales fusing to following Accumulate op from GPTQ Matmul unrolling"),
            llvm::cl::init(false)};

    BoolOption enableFP16CompressedConvolution{*this, "enable-fp16-compressed-convolution",
                                               llvm::cl::desc("Enable FP16 Compressed convolution op"),
                                               llvm::cl::init(false)};

    BoolOption enableWeightsDynamicDequantization{*this, "enable-weights-dynamic-dequantization",
                                                  llvm::cl::desc("Enable weights dequantization for weights as input"),
                                                  llvm::cl::init(false)};

    StrOption modelHash{*this, "model-hash", llvm::cl::desc("Hash of model XML architecture"), llvm::cl::init("")};

    BoolOption enableRuntimeDequant{*this, "enable-runtime-dequant",
                                    llvm::cl::desc("Enable runtime dequantization of asymmetricly quantized weight"),
                                    llvm::cl::init(false)};
    Int64Option runtimeDequantizationLimit{
            *this, "runtime-dequantization-limit",
            llvm::cl::desc("Lower limit on weight size for runtime dequantization"
                           "Weights smaller than the limit will be statically dequantized"),
            llvm::cl::init(524'288)};  // 512kb

    BoolOption enableInPlaceBufferization{
            *this, "enable-in-place-bufferization",
            llvm::cl::desc("Enable in-place bufferization. Might eliminate some redundant buffer allocations at the "
                           "cost of longer compile time"),
            llvm::cl::init(false)};

    bool enableForceZMajorConcat = false;
    bool enableSwapTransposeWithFQ = false;
    bool enableAlignScales = false;
    bool fuseMvn6ScaleBias = false;
    // TODO: remove option after E#83187
    bool enableFuseClamp = false;
    bool enableConvertFCToConv = false;
    bool enableAdjustNonZeroFakeQuant = false;
    bool enableAdaptiveStripping = false;
    bool enableExtraShapeBoundOps = false;
};

//
// ReferenceHWMode
//

struct ReferenceHWOptions40XX;

template <typename T>
struct ReferenceHWOptions : mlir::PassPipelineOptions<T> {
    BoolOption enableVerifiers{*this, "enable-verifiers", llvm::cl::desc("Enable verifiers execution after each pass"),
                               llvm::cl::init(isDeveloperBuild())};

    BoolOption enableMemoryUsageCollector{*this, "enable-memory-usage-collector",
                                          llvm::cl::desc("Enable peak memory usage instrumentation after each pass"),
                                          llvm::cl::init(isDeveloperBuild())};

    BoolOption enableFunctionStatisticsInstrumentation{
            *this, "enable-function-statistics-instrumentation",
            llvm::cl::desc("Enable printing statistics for functions after each pass"), llvm::cl::init(false)};

    // InitCompiler
    IntOption revisionID{*this, "revision-id", ::llvm::cl::desc("[Optional] Revision ID of the platform")};
    IntOption numberOfDPUGroups{*this, "num-of-dpu-groups",
                                ::llvm::cl::desc("[Optional] Number of available DPU groups")};
    IntOption numberOfDMAPorts{*this, "num-of-dma-ports", ::llvm::cl::desc("[Optional] Number of available DMA ports")};
    IntOption availableCMXMemory{*this, "available-cmx-memory", ::llvm::cl::desc("[Optional] Available CMX memory")};
    BoolOption allowCustomValues{*this, "allow-custom-values",
                                 ::llvm::cl::desc("[Optional] Allows keep predefined values in IR")};

    BoolOption enableConvertFCToConv{*this, "convert-fc-to-conv", llvm::cl::desc("Enable convert-fc-to-conv pass"),
                                     llvm::cl::init(true)};

    BoolOption enableHandleLargeKernel{*this, "handle-large-kernel", llvm::cl::desc("Enable handle-large-kernel pass"),
                                       llvm::cl::init(true)};

    BoolOption enableConvertAvgPoolToDWConv{*this, "convert-avg-pool-to-dw-conv",
                                            llvm::cl::desc("Enable convert-avg-pool-to-dw-conv pass"),
                                            llvm::cl::init(true)};

    BoolOption enableSwapTransposeWithFQ{*this, "swap-transpose-with-fq",
                                         ::llvm::cl::desc("Enable SwapTransposeWithFQ pass"), ::llvm::cl::init(true)};

    BoolOption enableOptimizeScaleShiftToDWConv{*this, "optimize-scale-shift-to-depthwise",
                                                llvm::cl::desc("Enable optimize-scale-shift-to-depthwise pass"),
                                                llvm::cl::init(true)};

    BoolOption enableSplitConvWithMultipleFQ{*this, "split-conv-with-multiple-fq",
                                             llvm::cl::desc("Enable split-conv-with-multiple-fq pass"),
                                             llvm::cl::init(true)};

    BoolOption enableU16FQToScaleShiftConversion{*this, "enable-u16-fake-quantize-to-scale-shift-conversion",
                                                 llvm::cl::desc("Enable u16 fake quantize to scale shift conversion"),
                                                 llvm::cl::init(false)};

    BoolOption enableHandleLargeStrides{*this, "handle-large-strides",
                                        llvm::cl::desc("Enable handle-large-strides pass"), llvm::cl::init(true)};

    BoolOption enableHandleLargePads{*this, "handle-large-pads", llvm::cl::desc("Enable handle-large-pads pass"),
                                     llvm::cl::init(true)};

    BoolOption enableHandleAsymmetricStrides{*this, "handle-asymmetric-strides",
                                             llvm::cl::desc("Enable handle-asymmetric-strides pass"),
                                             llvm::cl::init(true)};

    BoolOption enableBilinearInterpolateOnDPU{*this, "map-interpolate-on-dpu",
                                              llvm::cl::desc("Enable map-interpolate-on-dpu pass"),
                                              llvm::cl::init(false)};

    BoolOption enableSplitBilinerIntoHAndW{*this, "split-bilinear-into-H-and-W",
                                           llvm::cl::desc("Enable split-bilinear-into-H-and-W pass"),
                                           llvm::cl::init(false)};

    BoolOption enableLowPrecision{*this, "low-precision", llvm::cl::desc("Enable low-precision pipeline building"),
                                  llvm::cl::init(true)};

    BoolOption enableUpstreamSlice{*this, "upstream-slice", llvm::cl::desc("Enable upstream-slice pipeline building"),
                                   llvm::cl::init(true)};

    BoolOption enableExpandActivationChannels{*this, "expand-activation-channels",
                                              llvm::cl::desc("Enable expand-activation-channels pass"),
                                              llvm::cl::init(true)};

    BoolOption enableAdjustConvShapePass{*this, "adjust-convolution-shape",
                                         llvm::cl::desc("Enable adjust-convolution-shape pass"), llvm::cl::init(true)};

    BoolOption enableOptimizeSliceExpand{*this, "optimize-slice-expand",
                                         llvm::cl::desc("Enable optimize-slice-expand pass"), llvm::cl::init(true)};

    StrOption weightsSparsityHeuristic{*this, "weights-sparsity-heuristic",
                                       llvm::cl::desc("Weights sparsity heuristic (RATIO or CMX)"),
                                       llvm::cl::init("RATIO")};
    DoubleOption weightsSparsityThreshold{*this, "weights-sparsity-threshold",
                                          llvm::cl::desc("Threshold for ratio of sparse weights values"),
                                          llvm::cl::init(-1.0)};
    Int64Option weightsSparsityLargeConstThreshold{
            *this, "weights-sparsity-large-const-threshold",
            llvm::cl::desc(
                    "Sparsify weights using a single thread if the constant's size is larger than this threshold."),
            llvm::cl::init((200_MB).to<vpux::Byte>().count())};

    BoolOption enablePrefetching{*this, "prefetching",
                                 llvm::cl::desc("Enable prefetch tiling pass and prefetch scheduling"),
                                 llvm::cl::init(false)};

    BoolOption enablePipelining{*this, "pipelining",
                                llvm::cl::desc("Enable vertical fusion pipelining pass and schedule pipelining"),
                                llvm::cl::init(false)};

    IntOption concatRepeatingBlockOutliningSeqLength{
            *this, "concat-repeating-block-outlining-min-seq-length",
            llvm::cl::desc("Threshold for length of concat input sequence for repeating blocks outlining"),
            llvm::cl::init(5)};

    BoolOption enableConcatRepeatingBlockOutlining{*this, "concat-repeating-block-outlining",
                                                   llvm::cl::desc("Enable concat input as repeating blocks outlining"),
                                                   llvm::cl::init(true)};
    IntOption opTilingCacheThreshold{
            *this, "op-tiling-cache-threshold",
            llvm::cl::desc("threshold for number of clustered ops for tiling cache optimization"),
            llvm::cl::init(CLUSTERED_OP_THRESHOLD_FOR_TILING_CACHE)};

    IntOption vfOutliningInstanceThreshold{
            *this, "vf-outlining-instance-threshold",
            llvm::cl::desc("Threshold for number of instances (slices of the graph) to perform outlining"),
            llvm::cl::init(5)};

    IntOption vfOutliningTileThreshold{
            *this, "vf-outlining-tile-threshold",
            llvm::cl::desc("Threshold for outlining vertical fusion regions with accumulated number of tiles"),
            llvm::cl::init(10)};

    BoolOption enableVerticalFusionOutlining{*this, "vf-outlining", llvm::cl::desc("Enable vertical fusion outlining"),
                                             llvm::cl::init(false)};

    BoolOption enableVerticalFusion{*this, "vertical-fusion", llvm::cl::desc("Enable vertical fusion feature"),
                                    llvm::cl::init(false)};

    BoolOption enableOptimizeCopies{*this, "optimize-copies", llvm::cl::desc("Enable optimize-copies pass"),
                                    llvm::cl::init(true)};

    BoolOption enableOptimizeConstCopies{*this, "optimize-const-copies", llvm::cl::desc("Enable optimize-const-copies"),
                                         llvm::cl::init(true)};

    BoolOption enableConstantFusion{*this, "constant-fusion", llvm::cl::desc("Enable constant fusion"),
                                    llvm::cl::init(false)};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};

    // Enable for 40XX once RT will be ready, follow up #E95864
    StrOption enableDMAProfiling{*this, "dma-profiling",
                                 llvm::cl::desc("Enable DMA task profiling: (true, false, static)"),
                                 llvm::cl::init(std::is_same<T, ReferenceHWOptions40XX>::value ? "false" : "true")};

    BoolOption enableDPUProfiling{*this, "dpu-profiling", llvm::cl::desc("Enable DPU task profiling"),
                                  llvm::cl::init(true)};

    BoolOption enableSWProfiling{*this, "sw-profiling", llvm::cl::desc("Enable SW task profiling"),
                                 llvm::cl::init(true)};
    BoolOption enableM2IProfiling{*this, "m2i-profiling", llvm::cl::desc("Enable M2I task profiling"),
                                  llvm::cl::init(true)};

    BoolOption enableGroupAsyncExecuteOps{*this, "group-async-execute-ops",
                                          llvm::cl::desc("Enable group-async-execute-ops pass"), llvm::cl::init(false)};

    BoolOption enableCompressWeightsBTC{*this, "compress-weights-btc", ::llvm::cl::desc("Enable compress-weights pass"),
                                        ::llvm::cl::init(false)};

    BoolOption enableDumpTaskStats{*this, "dump-task-stats",
                                   ::llvm::cl::desc("Enable dumping statistics of Task operations"),
                                   ::llvm::cl::init(vpux::isDeveloperBuild())};

    BoolOption enableDummyOpReplacement{*this, "dummy-op-replacement",
                                        llvm::cl::desc("Replace unsupported SW Kernel ops with Dummy ones"),
                                        llvm::cl::init(false)};

    BoolOption constantFoldingInBackground{*this, "constant-folding-in-background",
                                           llvm::cl::desc("Fold constants in background threads"),
                                           llvm::cl::init(false)};

    IntOption constantFoldingInBackgroundNumThreads{
            *this, "constant-folding-in-background-num-threads",
            llvm::cl::desc("Number of background threads to use for constant folding in background. Ignored if "
                           "`constant-folding-in-background` is disabled."),
            llvm::cl::init(1)};

    BoolOption constantFoldingInBackgroundCollectStatistics{
            *this, "constant-folding-in-background-collect-statistics",
            llvm::cl::desc("Toggle for the collection of statistics when folding constants in background. Ignored if "
                           "`constant-folding-in-background` is disabled."),
            llvm::cl::init(false)};

    IntOption constantFoldingInBackgroundMemoryUsageLimit{
            *this, "constant-folding-in-background-memory-usage-limit",
            llvm::cl::desc("Fold constants in background memory usage limit (in MB)"), llvm::cl::init(3 * 1024)};

    DoubleOption constantFoldingInBackgroundCacheCleanThreshold{
            *this, "constant-folding-in-background-cache-clean-threshold",
            llvm::cl::desc("Cache will be cleaned to this threshold when reach the memory usage limit"),
            llvm::cl::init(0.8)};

    BoolOption wlmRollback{
            *this, "wlm-rollback",
            llvm::cl::desc("When compilation with WLM fails, automatically switches to WLM-disabled pipeline"),
            llvm::cl::init(true)};

    IntOption optimizationLevel{*this, "optimization-level",
                                llvm::cl::desc("Set compilation optimization level, enabled starting from NPU4."
                                               "Possible values: 0 - optimization for compilation time,"
                                               "1 - optimization for execution time (default),"
                                               "2 - high optimization for execution time"),
                                llvm::cl::init(1)};

    StrOption performanceHintOverride{*this, "performance-hint-override",
                                      llvm::cl::desc("Set performance hint for compiler to set up number of tiles."
                                                     "Possible values: latency, efficiency (default)"),
                                      llvm::cl::init("efficiency")};

    BoolOption enableOptimizeReorders{*this, "optimize-reorders", llvm::cl::desc("Enable optimize-reorders pass"),
                                      llvm::cl::init(true)};

    BoolOption enableOptimizeSliceWithStride{*this, "optimize-slice-with-stride",
                                             llvm::cl::desc("Enable optimize-slice-with-stride pass"),
                                             llvm::cl::init(false)};

    BoolOption enableConvertExpandToConvPass{*this, "convert-expand-to-conv",
                                             llvm::cl::desc("Enable convert-expand-to-conv pass"),
                                             llvm::cl::init(false)};

    BoolOption enableQuantDequantRemoval{*this, "quant-dequant-removal",
                                         llvm::cl::desc("Enable quantize->dequantize sequence removal"),
                                         llvm::cl::init(false)};

    BoolOption enableFuseOutstandingDequant{*this, "fuse-outstanding-dequant",
                                            llvm::cl::desc("Fuse outstanding dequantize after NCE task"),
                                            llvm::cl::init(false)};

    BoolOption enableFuseOutstandingQuant{*this, "fuse-outstanding-quant",
                                          llvm::cl::desc("Fuse outstanding quantize before two-input Eltwise task"),
                                          llvm::cl::init(false)};

    BoolOption enableForceZMajorConcat{*this, "force-z-major-concat",
                                       llvm::cl::desc("Enable transpose-reorder-concat pass"), llvm::cl::init(true)};

    BoolOption enablePropagateQuantDequant{*this, "propagate-quant-dequant",
                                           llvm::cl::desc("Enable Propagate Quantize Dequantize pass"),
                                           llvm::cl::init(true)};

    BoolOption enableAlignScales{*this, "enable-align-scales", llvm::cl::desc("Enable align scales"),
                                 llvm::cl::init(true)};

    BoolOption enableAdaptiveStripping{*this, "enable-adaptive-stripping", llvm::cl::desc("Enable adaptive stripping"),
                                       llvm::cl::init(false)};

    BoolOption enableFP16ToU8MixedMode{
            *this, "enable-fp16-to-u8-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with FP16 input and quantized output"),
            llvm::cl::init(false)};

    BoolOption enableFloatInQuantWeightsMixedMode{
            *this, "enable-float-in-quant-weights-mixed-mode",
            llvm::cl::desc("Enable mixed mode for NCE tasks with float input and quantized weights"),
            llvm::cl::init(true)};

    BoolOption enableInPlaceEltwise{*this, "enable-in-place-eltwise",
                                    llvm::cl::desc("Enable inplace eltwise op execution"), llvm::cl::init(false)};

    BoolOption readStrategyFromJson{*this, "read-strategy-from-json",
                                    llvm::cl::desc("Read the multiclustering and tiling strategy from a JSON file"),
                                    llvm::cl::init(false)};

    BoolOption writeStrategyToJson{*this, "write-strategy-to-json",
                                   llvm::cl::desc("Write the multiclustering and tiling strategy to a JSON file"),
                                   llvm::cl::init(false)};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option (true, false or auto)"), llvm::cl::init("true")};

    StrOption modelHash{*this, "model-hash", llvm::cl::desc("Hash of model XML architecture"), llvm::cl::init("")};

    BoolOption enableOpsAsDMA{*this, "enable-ops-as-dma",
                              llvm::cl::desc("Force using DMA transformations instead of SW ops"),
                              llvm::cl::init(false)};

    BoolOption enableControlGraphSplit{*this, "enable-control-graph-split",
                                       llvm::cl::desc("Enable split of control graph to simplify barrier scheduling"),
                                       llvm::cl::init(true)};
    IntOption controlGraphSplitBlockSize{
            *this, "control-graph-split-block-size",
            llvm::cl::desc("Maximal number of tasks in each block that control graph will be split into. Used to "
                           "reduce memory consumption of barrier legalization pipeline for big models. Memory usage is "
                           "roughly (control-graph-split-block-size)^2/8"),
            llvm::cl::init(CONTROL_GRAPH_SPLIT_BLOCK_SIZE)};

    BoolOption enableSimpleSchedule{*this, "simple-schedule", llvm::cl::desc("Enable schedule simplification"),
                                    llvm::cl::init(true)};

    BoolOption shareWaitAndUpdateBarriers{*this, "share-wait-and-update-barriers",
                                          llvm::cl::desc("Share wait and update barriers"), llvm::cl::init(true)};

    BoolOption reduceParallelControlFlows{*this, "reduce-parallel-control-flows",
                                          llvm::cl::desc("Reduce parallel overlapping control flows where possible"),
                                          llvm::cl::init(true)};

    BoolOption enableScheduleTrace{*this, "enable-schedule-trace",
                                   llvm::cl::desc("Enable compile time schedule analysis and trace"),
                                   llvm::cl::init(false)};

    BoolOption enableIntermediateBufferOutput{
            *this, "enable-intermediate-buffer-output",
            llvm::cl::desc("Enable intermediate output of defined operation buffer at specified insertion place"),
            llvm::cl::init(false)};

    BoolOption enableActivityFactor{*this, "enable-activity-factor",
                                    llvm::cl::desc("Enable activity factor and inference time estimation"),
                                    llvm::cl::init(true)};

    StrOption scheduleTraceFile{*this, "schedule-trace-file-name",
                                llvm::cl::desc("Compile time schedule JSON trace file name"),
                                llvm::cl::init("compileTimeScheduleTrace.json")};

    BoolOption logOpOptimizations{*this, "log-op-optimizations",
                                  llvm::cl::desc("Log potential operation optimizations that can be done"),
                                  llvm::cl::init(false)};

    BoolOption enableSMPipeline{*this, "enable-SM-Pipeline", llvm::cl::desc("Enable Strategy Manager pipeline"),
                                llvm::cl::init(false)};

    BoolOption enableAdjustNonZeroFakeQuant{*this, "adjust-non-zero-fake-quant",
                                            llvm::cl::desc("Enable adjust non zero fake quant"), llvm::cl::init(true)};

    BoolOption optimizeFragmentation{*this, "optimize-fragmentation",
                                     ::llvm::cl::desc("Enables compiler to optimize CMX fragmentation"),
                                     ::llvm::cl::init(true)};

    BoolOption optimizeDynamicSpilling{*this, "optimize-dynamic-spilling",
                                       ::llvm::cl::desc("Enables compiler to optimize dynamic spilling DMAs"),
                                       ::llvm::cl::init(true)};

    BoolOption linearizeSchedule{*this, "linearize-schedule", llvm::cl::desc("Linearize tasks on all engines"),
                                 llvm::cl::init(false)};

    BoolOption enableConvertPrecisionToFP16{*this, "convert-precision-to-fp16",
                                            llvm::cl::desc("Enable convert-precision-to-fp16 pass"),
                                            llvm::cl::init(true)};

    StrOption computeLayersWithHigherPrecision{
            *this, "compute-layers-with-higher-precision",
            llvm::cl::desc("Enable compute layers with higher precision for the specified layer types"),
            llvm::cl::init("")};

    BoolOption enableSWKernelPrefetchingReserveMem{
            *this, "enable-sw-kernel-prefetching-reserve-mem",
            ::llvm::cl::desc("Reserve memory at the end of CMX for SW Kernel data prefetching"),
            ::llvm::cl::init(true)};

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(false)};

    BoolOption enableDmaOutOfOrder{*this, "dma-ooo", llvm::cl::desc("Enable out-of-order DMA"), llvm::cl::init(true)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingODU{*this, "enable-auto-padding-odu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingIDU{*this, "enable-auto-padding-idu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    BoolOption enableConvolutionMixedPrecisionDecomposition{
            *this, "enable-convolution-mixed-precision-decomposition",
            llvm::cl::desc("Enable mixed precision decomposition for convolution"), llvm::cl::init(false)};

    BoolOption accumulateMatmulWithDPU{*this, "accumulate-matmul-with-dpu",
                                       llvm::cl::desc("Accumulate unrolled Matmul results with DPU"),
                                       llvm::cl::init(false)};

    BoolOption fuseScalesToAccumulate{
            *this, "fuse-scales-to-accumulate",
            llvm::cl::desc("Enable scales fusing to following Accumulate op from GPTQ Matmul unrolling"),
            llvm::cl::init(false)};

    BoolOption enableDynamicQuant{*this, "enable-dynamic-quant",
                                  llvm::cl::desc("Enable dynamic quant weights signal pass."), llvm::cl::init(false)};

    BoolOption enableColorBinPhysicalBarrierAssignment{
            *this, "enable-color-bin-physical-barrier-assignment",
            llvm::cl::desc("Enable physical barrier assignment optimization"), llvm::cl::init(false)};

    BoolOption enablePopulateWeightTableWithShave{*this, "enable-populate-weight-table-with-shave",
                                                  llvm::cl::desc("Enable populating weights table with Shave"),
                                                  llvm::cl::init(false)};

    BoolOption enableFP16CompressedConvolution{*this, "enable-fp16-compressed-convolution",
                                               llvm::cl::desc("Enable FP16 Compressed convolution op"),
                                               llvm::cl::init(false)};

    BoolOption enableMCSideLoadDump{*this, "enable-mc-side-loading-dump",
                                    llvm::cl::desc("Dump multi-cluster strategies in side-loading format"),
                                    llvm::cl::init(false)};

    BoolOption fuseMvn6ScaleBias{*this, "fuse-mvn6-scale-bias", llvm::cl::desc("Enable fuse-mvn6-scale-bias pass"),
                                 llvm::cl::init(false)};

    BoolOption enableWeightsDynamicDequantization{*this, "enable-weights-dynamic-dequantization",
                                                  llvm::cl::desc("Enable weights dequantization for weights as input"),
                                                  llvm::cl::init(false)};
    BoolOption enableInPlaceBufferization{
            *this, "enable-in-place-bufferization",
            llvm::cl::desc("Enable in-place bufferization. Might eliminate some redundant buffer allocations at the "
                           "cost of longer compile time"),
            llvm::cl::init(false)};

    BoolOption enableDPUF16ToF32Convert{*this, "enable-dpu-f16-to-f32-convert",
                                        llvm::cl::desc("Enable running F16 -> F32 converts on DPU."),
                                        llvm::cl::init(false)};

    bool enableExtraShapeBoundOps = false;
};

//
// DefaultHWOptionsBase
// This class must be inherited by all dialect-base options
// to avoid confusion when we have the same option for IE and the VPU dialect, but with a different value
//

struct DefaultHWOptionsBase : mlir::PassPipelineOptions<DefaultHWOptionsBase> {
    BoolOption enableAdaptiveStripping{*this, "enable-adaptive-stripping", llvm::cl::desc("Enable adaptive stripping"),
                                       llvm::cl::init(false)};

    BoolOption enableDPUF16ToF32Convert{*this, "enable-dpu-f16-to-f32-convert",
                                        llvm::cl::desc("Enable running F16 -> F32 converts on DPU."),
                                        llvm::cl::init(false)};

    BoolOption enableVerifiers{*this, "enable-verifiers", llvm::cl::desc("Enable verifiers execution after each pass"),
                               llvm::cl::init(isDeveloperBuild())};

    BoolOption enableMemoryUsageCollector{*this, "enable-memory-usage-collector",
                                          llvm::cl::desc("Enable peak memory usage instrumentation after each pass"),
                                          llvm::cl::init(isDeveloperBuild())};

    BoolOption enableFunctionStatisticsInstrumentation{
            *this, "enable-function-statistics-instrumentation",
            llvm::cl::desc("Enable printing statistics for functions after each pass"), llvm::cl::init(false)};

    StrOption functionOutlining{*this, "function-outlining",
                                llvm::cl::desc("Define a list of outlining modes and their parameters where the next "
                                               "outlining mode is the fallback mode of the previous one."
                                               "Example: function-outlining=' repeating-blocks=max-num-iterations=30 "
                                               "min-ops-in-block=16, naive=num-parts=2'")};

    BoolOption enableDebatcher{*this, "debatching",
                               llvm::cl::desc("Apply debatching operation for batched tensors, which are arguments of "
                                              "'main', facilitating further function-outlining enabling"),
                               llvm::cl::init(false)};

    StrOption debatcherExtraArgs{*this, "debatching-extra-args", llvm::cl::desc("Extra arguments for debatching "),
                                 llvm::cl::init("")};

    StrOption debatcherInliningMethod{*this, "debatching-inlining-method",
                                      llvm::cl::desc("Method for inlinging of debatching-function. Supported methods: "
                                                     "\"naive\", \"reordering\". Default is \"reordering\""),
                                      llvm::cl::init("reordering")};

    BoolOption enableLoopOutliner{*this, "loop-outlining", llvm::cl::desc("Apply outlining for body of Loop op"),
                                  llvm::cl::init(false)};

    BoolOption enableDummyOpReplacement{*this, "dummy-op-replacement",
                                        llvm::cl::desc("Replace unsupported SW Kernel ops with Dummy ones"),
                                        llvm::cl::init(false)};

    BoolOption constantFoldingInBackground{*this, "constant-folding-in-background",
                                           llvm::cl::desc("Fold constants in background threads"),
                                           llvm::cl::init(false)};

    IntOption constantFoldingInBackgroundNumThreads{
            *this, "constant-folding-in-background-num-threads",
            llvm::cl::desc("Number of background threads to use for constant folding in background. Ignored if "
                           "`constant-folding-in-background` is disabled."),
            llvm::cl::init(1)};

    BoolOption constantFoldingInBackgroundCollectStatistics{
            *this, "constant-folding-in-background-collect-statistics",
            llvm::cl::desc("Toggle for the collection of statistics when folding constants in background. Ignored if "
                           "`constant-folding-in-background` is disabled."),
            llvm::cl::init(false)};

    IntOption constantFoldingInBackgroundMemoryUsageLimit{
            *this, "constant-folding-in-background-memory-usage-limit",
            llvm::cl::desc("Fold constants in background memory usage limit (in MB)"), llvm::cl::init(3 * 1024)};

    DoubleOption constantFoldingInBackgroundCacheCleanThreshold{
            *this, "constant-folding-in-background-cache-clean-threshold",
            llvm::cl::desc("Cache will be cleaned to this threshold when reach the memory usage limit"),
            llvm::cl::init(0.8)};

    BoolOption wlmRollback{
            *this, "wlm-rollback",
            llvm::cl::desc("When compilation with WLM fails, automatically switches to WLM-disabled pipeline"),
            llvm::cl::init(true)};

    IntOption optimizationLevel{*this, "optimization-level",
                                llvm::cl::desc("Set compilation optimization level, enabled starting from NPU4."
                                               "Possible values: 0 - optimization for compilation time,"
                                               "1 - optimization for execution time (default),"
                                               "2 - high optimization for execution time"),
                                llvm::cl::init(1)};

    StrOption performanceHintOverride{*this, "performance-hint-override",
                                      llvm::cl::desc("Set performance hint for compiler to set up number of tiles."
                                                     "Possible values: latency, efficiency (default)"),
                                      llvm::cl::init("efficiency")};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};

    BoolOption enableScheduleTrace{*this, "enable-schedule-trace",
                                   llvm::cl::desc("Enable compile time schedule analysis and trace"),
                                   llvm::cl::init(false)};

    BoolOption enableIntermediateBufferOutput{
            *this, "enable-intermediate-buffer-output",
            llvm::cl::desc("Enable intermediate output of defined operation buffer at specified insertion place"),
            llvm::cl::init(false)};

    BoolOption enableActivityFactor{*this, "enable-activity-factor",
                                    llvm::cl::desc("Enable activity factor and inference time estimation"),
                                    llvm::cl::init(true)};

    StrOption scheduleTraceFile{*this, "schedule-trace-file-name",
                                llvm::cl::desc("Compile time schedule JSON trace file name"),
                                llvm::cl::init("compileTimeScheduleTrace.json")};

    BoolOption enablePrefetching{*this, "prefetching",
                                 llvm::cl::desc("Enable prefetch tiling pass and prefetch scheduling"),
                                 llvm::cl::init(true)};

    BoolOption enablePipelining{*this, "pipelining",
                                llvm::cl::desc("Enable vertical fusion pipelining pass and schedule pipelining"),
                                llvm::cl::init(true)};

    IntOption concatRepeatingBlockOutliningSeqLength{
            *this, "concat-repeating-block-outlining-min-seq-length",
            llvm::cl::desc("Threshold for length of concat input sequence for repeating blocks outlining"),
            llvm::cl::init(5)};

    BoolOption enableConcatRepeatingBlockOutlining{*this, "concat-repeating-block-outlining",
                                                   llvm::cl::desc("Enable concat input as repeating blocks outlining"),
                                                   llvm::cl::init(true)};
    IntOption opTilingCacheThreshold{
            *this, "op-tiling-cache-threshold",
            llvm::cl::desc("threshold for number of clustered ops for tiling cache optimization"),
            llvm::cl::init(CLUSTERED_OP_THRESHOLD_FOR_TILING_CACHE)};

    IntOption vfOutliningInstanceThreshold{
            *this, "vf-outlining-instance-threshold",
            llvm::cl::desc("Threshold for number of instances (slices of the graph) to perform outlining"),
            llvm::cl::init(5)};

    IntOption vfOutliningTileThreshold{
            *this, "vf-outlining-tile-threshold",
            llvm::cl::desc("Threshold for outlining vertical fusion regions with accumulated number of tiles"),
            llvm::cl::init(10)};

    BoolOption enableVerticalFusionOutlining{*this, "vf-outlining", llvm::cl::desc("Enable vertical fusion outlining"),
                                             llvm::cl::init(true)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingODU{*this, "enable-auto-padding-odu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    // SetupChannelsAutoPadding pass options
    BoolOption enableAutoPaddingIDU{*this, "enable-auto-padding-idu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(false)};

    StrOption computeLayersWithHigherPrecision{
            *this, "compute-layers-with-higher-precision",
            llvm::cl::desc("Enable compute layers with higher precision for the specified layer types"),
            llvm::cl::init("")};

    BoolOption accumulateMatmulWithDPU{*this, "accumulate-matmul-with-dpu",
                                       llvm::cl::desc("Accumulate unrolled Matmul results with DPU"),
                                       llvm::cl::init(false)};

    BoolOption fuseScalesToAccumulate{
            *this, "fuse-scales-to-accumulate",
            llvm::cl::desc("Enable scales fusing to following Accumulate op from GPTQ Matmul unrolling"),
            llvm::cl::init(false)};

    // InitCompiler
    IntOption revisionID{*this, "revision-id", ::llvm::cl::desc("[Optional] Revision ID of the platform")};
    IntOption numberOfDPUGroups{*this, "num-of-dpu-groups",
                                ::llvm::cl::desc("[Optional] Number of available DPU groups")};
    IntOption numberOfDMAPorts{*this, "num-of-dma-ports", ::llvm::cl::desc("[Optional] Number of available DMA ports")};
    IntOption availableCMXMemory{*this, "available-cmx-memory", ::llvm::cl::desc("[Optional] Available CMX memory")};
    BoolOption allowCustomValues{*this, "allow-custom-values",
                                 ::llvm::cl::desc("[Optional] Allows keep predefined values in IR")};

    // VPURT
    BoolOption enableControlGraphSplit{*this, "enable-control-graph-split",
                                       llvm::cl::desc("Enable split of control graph to simplify barrier scheduling"),
                                       llvm::cl::init(true)};
    IntOption controlGraphSplitBlockSize{
            *this, "control-graph-split-block-size",
            llvm::cl::desc("Maximal number of tasks in each block that control graph will be split into. Used to "
                           "reduce memory consumption of barrier legalization pipeline for big models. Memory usage is "
                           "roughly (control-graph-split-block-size)^2/8"),
            llvm::cl::init(CONTROL_GRAPH_SPLIT_BLOCK_SIZE)};

    BoolOption enableSimpleSchedule{*this, "simple-schedule", llvm::cl::desc("Enable schedule simplification"),
                                    llvm::cl::init(true)};

    BoolOption shareWaitAndUpdateBarriers{*this, "share-wait-and-update-barriers",
                                          llvm::cl::desc("Share wait and update barriers"), llvm::cl::init(true)};

    BoolOption reduceParallelControlFlows{*this, "reduce-parallel-control-flows",
                                          llvm::cl::desc("Reduce parallel overlapping control flows where possible"),
                                          llvm::cl::init(true)};

    Int64Option constantFoldingSizeThreshold{
            *this, "constant-folding-threshold",
            llvm::cl::desc("Fold constants in single threading if the size is larger than the threshold."),
            llvm::cl::init(300 * 1024 * 1024)};  // 300MB

    // Option to control locations verification. Possible options are:
    // off - no verification
    // fast - verifies locations only after last* pass by thorough algorithm
    // full - verifies locations after each pass using fast algorithm and thorough after last* pass
    // thorough - does thorough verification after each** pass
    // This feature is in process of development, details: #E81319
    // *last in sequence of fixed passes. Some passes is not fixed yet, so may break compilation because of locations
    // reuse
    // **each of fixed passes
    StrOption locationsVerificationMode{
            *this, "verify-locations",
            llvm::cl::desc("Selects location verification mode. Possible options are off/fast/full/thorough"),
            llvm::cl::init(vpux::isDeveloperBuild() ? "fast" : "off")};

    BoolOption enableDmaOutOfOrder{*this, "dma-ooo", llvm::cl::desc("Enable out-of-order DMA"), llvm::cl::init(true)};

    BoolOption enableColorBinPhysicalBarrierAssignment{
            *this, "enable-color-bin-physical-barrier-assignment",
            llvm::cl::desc("Enable physical barrier assignment optimization"), llvm::cl::init(false)};

    BoolOption enablePopulateWeightTableWithShave{*this, "enable-populate-weight-table-with-shave",
                                                  llvm::cl::desc("Enable populating weights table with Shave"),
                                                  llvm::cl::init(false)};

    BoolOption enableFP16CompressedConvolution{*this, "enable-fp16-compressed-convolution",
                                               llvm::cl::desc("Enable FP16 Compressed convolution op"),
                                               llvm::cl::init(false)};

    StrOption modelHash{*this, "model-hash", llvm::cl::desc("Hash of model XML architecture"), llvm::cl::init("")};

    BoolOption enableMCSideLoadDump{*this, "enable-mc-side-loading-dump",
                                    llvm::cl::desc("Dump multi-cluster strategies in side-loading format"),
                                    llvm::cl::init(false)};

    BoolOption enableWeightsDynamicDequantization{*this, "enable-weights-dynamic-dequantization",
                                                  llvm::cl::desc("Enable weights dequantization for weights as input"),
                                                  llvm::cl::init(false)};
    BoolOption enableInPlaceBufferization{
            *this, "enable-in-place-bufferization",
            llvm::cl::desc("Enable in-place bufferization. Might eliminate some redundant buffer allocations at the "
                           "cost of longer compile time"),
            llvm::cl::init(false)};

    BoolOption enableExtraShapeBoundOps{
            *this, "enable-extra-shape-bound-ops",
            llvm::cl::desc("Attach ShapeBoundOp trait to operations that perform faster when their shapes are static."),
            llvm::cl::init(false)};
};

//
// MCAndTilingOptionsBase options
//

struct MCAndTilingOptionsBase : mlir::PassPipelineOptions<MCAndTilingOptionsBase> {
    BoolOption enablePrefetching{*this, "prefetching", llvm::cl::desc("Enable prefetch mode"), llvm::cl::init(true)};

    BoolOption enableVerticalFusion{*this, "vertical-fusion", llvm::cl::desc("Enable vertical fusion feature"),
                                    llvm::cl::init(false)};

    BoolOption enablePipelining{*this, "pipelining", llvm::cl::desc("Enable vertical fusion pipelining"),
                                llvm::cl::init(false)};

    IntOption opTilingCacheThreshold{
            *this, "op-tiling-cache-threshold",
            llvm::cl::desc("threshold for number of clustered ops for tiling cache optimization"),
            llvm::cl::init(CLUSTERED_OP_THRESHOLD_FOR_TILING_CACHE)};

    IntOption vfOutliningInstanceThreshold{
            *this, "vf-outlining-instance-threshold",
            llvm::cl::desc("Threshold for number of instances (slices of the graph) to perform outlining"),
            llvm::cl::init(5)};

    IntOption vfOutliningTileThreshold{
            *this, "vf-outlining-tile-threshold",
            llvm::cl::desc("Threshold for outlining vertical fusion regions with accumulated number of tiles"),
            llvm::cl::init(10)};

    BoolOption enableVerticalFusionOutlining{*this, "vf-outlining", llvm::cl::desc("Enable vertical fusion outlining"),
                                             llvm::cl::init(false)};

    BoolOption enableProfiling{*this, "profiling", llvm::cl::desc("Enable profiling"), llvm::cl::init(false)};

    // Extended Tiling options - Incremental Pipeline
    BoolOption readStrategyFromJson{*this, "read-strategy-from-json",
                                    llvm::cl::desc("Read the multiclustering and tiling strategy from a JSON file"),
                                    llvm::cl::init(false)};

    BoolOption writeStrategyToJson{*this, "write-strategy-to-json",
                                   llvm::cl::desc("Write the multiclustering and tiling strategy to a JSON file"),
                                   llvm::cl::init(false)};

    BoolOption enableVPUNNCostForTiling{*this, "enable-vpunn-cost-for-tiling",
                                        llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                                        llvm::cl::init(false)};

    BoolOption enableOutputPipelining{*this, "output-pipelining", llvm::cl::desc("Enable output pipelining"),
                                      llvm::cl::init(false)};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option. (true, false or auto)"), llvm::cl::init("true")};

    BoolOption enableExplicitDistributionInfoAttr{
            *this, "enable-explicit-distributed-attr",
            llvm::cl::desc("Enable DistributionInfoAttr with explicit per cluster memory/compute shapes & offsets"),
            llvm::cl::init(false)};

    BoolOption enableMCSideLoadDump{*this, "enable-mc-side-loading-dump",
                                    llvm::cl::desc("Dump multi-cluster strategies in side-loading format"),
                                    llvm::cl::init(false)};

    StrOption modelHash{*this, "model-hash", llvm::cl::desc("Hash of model architecture XML"), llvm::cl::init("")};

    MCAndTilingOptionsBase() = default;

    template <class OtherOptions>
    explicit MCAndTilingOptionsBase(const OtherOptions& options) {
        enablePrefetching = options.enablePrefetching;
        enableVerticalFusion = options.enableVerticalFusion;
        enablePipelining = options.enablePipelining;
        enableVPUNNCostForTiling = options.enableVPUNNCostForTiling;
        opTilingCacheThreshold = options.opTilingCacheThreshold;
        vfOutliningInstanceThreshold = options.vfOutliningInstanceThreshold;
        vfOutliningTileThreshold = options.vfOutliningTileThreshold;
        enableVerticalFusionOutlining = options.enableVerticalFusionOutlining;
        enableProfiling = options.enableProfiling;
        enableOutputPipelining = options.enableOutputPipelining;
        enableShaveDDRAccessOptimization = options.enableShaveDDRAccessOptimization;
        readStrategyFromJson = options.readStrategyFromJson;
        writeStrategyToJson = options.writeStrategyToJson;
        enableExplicitDistributionInfoAttr = options.enableExplicitDistributionInfoAttr;
        modelHash = options.modelHash;
        enableMCSideLoadDump = options.enableMCSideLoadDump;
    }
};

template <typename T>
struct BackendCompilationOptionsBase : mlir::PassPipelineOptions<T> {
    BoolOption enableMemorySideCache{*this, "enable-memory-side-cache", llvm::cl::desc("Enable memory side cache"),
                                     llvm::cl::init(false)};
    BoolOption enablePartialWorkloadManagement{*this, "enable-partial-workload-management",
                                               llvm::cl::desc("Enable partial workload management"),
                                               llvm::cl::init(true)};
    StrOption enableDMAProfiling{*this, "dma-profiling",
                                 llvm::cl::desc("Enable DMA task profiling (true|static|false)"),
                                 llvm::cl::init("false")};

    IntOption wlmOptimizationThreshold{*this, "wlm-barriers-threshold",
                                       llvm::cl::desc("Threshold for WLM optimization"),
                                       llvm::cl::init(VIRTUAL_BARRIER_THRESHOLD_WLM)};

    mlir::detail::PassOptions::Option<WlmVpurtEnqueueMode> wlmVpurtEnqueue{
            *this, "wlm-vpurt-enqueue",
            ::llvm::cl::desc("Option for enabling WLM enqueue barriers search algorithm at VPURT. To be used only for "
                             "experiments."),
            ::llvm::cl::init(WlmVpurtEnqueueMode::DISABLED),
            ::llvm::cl::values(clEnumValN(WlmVpurtEnqueueMode::ENABLED, "ENABLED",
                                          "WLM enqueue barriers search algorithm at VPURT ENABLED"),
                               clEnumValN(WlmVpurtEnqueueMode::DISABLED, "DISABLED",
                                          "WLM enqueue barriers search algorithm at VPURT DISABLED. Use LCA based "
                                          "enqueue algorithm at VPUMI"))};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option (true, false or auto)"), llvm::cl::init("true")};

    BoolOption enableDumpStatisticsOfWlmOps{*this, "enable-dump-wlm-ops-stats",
                                            llvm::cl::desc("Enable dump of WLM ops statistics"), llvm::cl::init(false)};

    mlir::detail::PassOptions::Option<AllocateShaveStackFrames> allocateShaveStackFrames{
            *this, "allocate-shave-stack-frames",
            ::llvm::cl::desc("Enable the computation and allocation of a new section which "
                             "will be used as stack frame form shaves."),
            ::llvm::cl::init(AllocateShaveStackFrames::DISABLED),
            ::llvm::cl::values(
                    clEnumValN(AllocateShaveStackFrames::ENABLED, "ENABLED",
                               "Allocate DDR buffer to be used as shave stack frames."),
                    clEnumValN(AllocateShaveStackFrames::DISABLED, "DISABLED", "Stack frames allocated by FW."))};
};

template <typename T>
struct ShaveCodeGenOptionsBase : mlir::PassPipelineOptions<T> {};

}  // namespace vpux
