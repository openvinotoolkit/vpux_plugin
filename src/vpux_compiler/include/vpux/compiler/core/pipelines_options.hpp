//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/compiler/core/developer_build_utils.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <type_traits>

namespace vpux {

//
// ReferenceSWMode
//

template <typename T>
struct ReferenceSWOptions : mlir::PassPipelineOptions<T> {
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
                                llvm::cl::desc("Set compilation optimization level, enabled starting from NPU40XX."
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
            llvm::cl::init(100000)};

    StrOption computeLayersWithHigherPrecision{
            *this, "compute-layers-with-higher-precision",
            llvm::cl::desc("Enable compute layers with higher precision for the specified layer types"),
            llvm::cl::init("")};

    BoolOption enableSimpleSchedule{*this, "simple-schedule", llvm::cl::desc("Enable schedule simplification"),
                                    llvm::cl::init(false)};

    BoolOption shareWaitAndUpdateBarriers{*this, "share-wait-and-update-barriers",
                                          llvm::cl::desc("Share wait and update barriers"), llvm::cl::init(true)};

    BoolOption reduceParallelControlFlows{*this, "reduce-parallel-control-flows",
                                          llvm::cl::desc("Reduce parallel overlapping control flows where possible"),
                                          llvm::cl::init(true)};

    BoolOption enableSWKernelPrefetchingReserveMem{
            *this, "enable-sw-kernel-prefetching-reserve-mem",
            ::llvm::cl::desc("Reserve memory at the end of CMX for SW Kernel data prefetching"),
            ::llvm::cl::init(true)};

    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"),
            llvm::cl::init(false)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    bool enableForceZMajorConcat = false;
    bool enableSwapTransposeWithFQ = false;
    bool enableAlignScales = false;
    // TODO: remove option after E-83187
    bool enableFuseClamp = false;
    bool enableConvertFCToConv = false;
    bool enableAdjustNonZeroFakeQuant = false;
};

//
// ReferenceHWMode
//

struct ReferenceHWOptions40XX;

template <typename T>
struct ReferenceHWOptions : mlir::PassPipelineOptions<T> {
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

    BoolOption enablePrefetching{*this, "prefetching",
                                 llvm::cl::desc("Enable prefetch tiling pass and prefetch scheduling"),
                                 llvm::cl::init(false)};

    BoolOption enablePipelining{*this, "pipelining",
                                llvm::cl::desc("Enable vertical fusion pipelining pass and schedule pipelining"),
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
                                llvm::cl::desc("Set compilation optimization level, enabled starting from NPU40XX0XX."
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

    BoolOption enableForceZMajorConcat{*this, "force-z-major-concat",
                                       llvm::cl::desc("Enable transpose-reorder-concat pass"), llvm::cl::init(true)};

    BoolOption enablePropagateQuantDequant{*this, "propagate-quant-dequant",
                                           llvm::cl::desc("Enable Propagate Quantize Dequantize pass"),
                                           llvm::cl::init(true)};

    BoolOption enableAlignScales{*this, "enable-align-scales", llvm::cl::desc("Enable align scales"),
                                 llvm::cl::init(true)};

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
            llvm::cl::init(100000)};

    BoolOption enableSimpleSchedule{*this, "simple-schedule", llvm::cl::desc("Enable schedule simplification"),
                                    llvm::cl::init(false)};

    BoolOption shareWaitAndUpdateBarriers{*this, "share-wait-and-update-barriers",
                                          llvm::cl::desc("Share wait and update barriers"), llvm::cl::init(true)};

    BoolOption reduceParallelControlFlows{*this, "reduce-parallel-control-flows",
                                          llvm::cl::desc("Reduce parallel overlapping control flows where possible"),
                                          llvm::cl::init(true)};

    BoolOption enableScheduleTrace{*this, "enable-schedule-trace",
                                   llvm::cl::desc("Enable compile time schedule analysis and trace"),
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

    BoolOption enableConvolutionMixedPrecisionDecomposition{
            *this, "enable-convolution-mixed-precision-decomposition",
            llvm::cl::desc("Enable mixed precision decomposition for convolution"), llvm::cl::init(false)};
};

//
// DefaultHWOptionsBase
// This class must be inherited by all dialect-base options
// to avoid confusion when we have the same option for IE and the VPU dialect, but with a different value
//

struct DefaultHWOptionsBase : mlir::PassPipelineOptions<DefaultHWOptionsBase> {
    BoolOption enableFunctionOutlining{*this, "function-outlining",
                                       llvm::cl::desc("Divide the IR into multiple parts to compile them in parallel"),
                                       llvm::cl::init(false)};

    StrOption functionOutliningMode{*this, "function-outlining-mode",
                                    llvm::cl::desc("Selects the outlining mode: `naive` or `repeating-blocks`"),
                                    llvm::cl::init("naive")};

    BoolOption enableDebatcher{*this, "debatching",
                               llvm::cl::desc("Apply debatching operation for batched tensors, which are arguments of "
                                              "'main', facilitating further function-outlining enabling"),
                               llvm::cl::init(false)};

    StrOption debatcherExtraArgs{*this, "debatching-extra-args", llvm::cl::desc("Extra arguments for debatching "),
                                 llvm::cl::init("")};

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
                                llvm::cl::desc("Set compilation optimization level, enabled starting from NPU40XX."
                                               "Possible values: 0 - optimization for compilation time,"
                                               "1 - optimization for execution time (default),"
                                               "2 - high optimization for execution time"),
                                llvm::cl::init(1)};

    StrOption performanceHintOverride{*this, "performance-hint-override",
                                      llvm::cl::desc("Set performance hint for compiler to set up number of tiles."
                                                     "Possible values: latency, efficiency (default)"),
                                      llvm::cl::init("efficiency")};

    BoolOption enableScheduleTrace{*this, "enable-schedule-trace",
                                   llvm::cl::desc("Enable compile time schedule analysis and trace"),
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

    StrOption computeLayersWithHigherPrecision{
            *this, "compute-layers-with-higher-precision",
            llvm::cl::desc("Enable compute layers with higher precision for the specified layer types"),
            llvm::cl::init("")};

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
            llvm::cl::init(100000)};

    BoolOption enableSimpleSchedule{*this, "simple-schedule", llvm::cl::desc("Enable schedule simplification"),
                                    llvm::cl::init(true)};

    BoolOption shareWaitAndUpdateBarriers{*this, "share-wait-and-update-barriers",
                                          llvm::cl::desc("Share wait and update barriers"), llvm::cl::init(true)};

    BoolOption reduceParallelControlFlows{*this, "reduce-parallel-control-flows",
                                          llvm::cl::desc("Reduce parallel overlapping control flows where possible"),
                                          llvm::cl::init(true)};
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

    // Extended Tiling options - Incremental Pipeline
    BoolOption readStrategyFromJson{*this, "read-strategy-from-json",
                                    llvm::cl::desc("Read the multiclustering and tiling strategy from a JSON file"),
                                    llvm::cl::init(false)};

    BoolOption writeStrategyToJson{*this, "write-strategy-to-json",
                                   llvm::cl::desc("Write the multiclustering and tiling strategy to a JSON file"),
                                   llvm::cl::init(false)};

    BoolOption enableVPUNNCost{*this, "vpunn-cost",
                               llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                               llvm::cl::init(false)};

    BoolOption enableOutputPipelining{*this, "output-pipelining", llvm::cl::desc("Enable output pipelining"),
                                      llvm::cl::init(false)};

    StrOption enableShaveDDRAccessOptimization{
            *this, "enable-shave-ddr-access-optimization",
            llvm::cl::desc("SHAVE DDR access optimization option. (true, false or auto)"), llvm::cl::init("true")};

    BoolOption enableExplicitDistributedTensorAttr{
            *this, "enable-explicit-distributed-attr",
            llvm::cl::desc("Enable DistributedTensorAttr with explicit per cluster memory/compute shapes & offsets"),
            llvm::cl::init(false)};

    MCAndTilingOptionsBase() = default;

    template <class OtherOptions>
    explicit MCAndTilingOptionsBase(const OtherOptions& options) {
        enablePrefetching = options.enablePrefetching;
        enableVerticalFusion = options.enableVerticalFusion;
        enablePipelining = options.enablePipelining;
        enableVPUNNCost = options.enableVPUNNCost;
        enableOutputPipelining = options.enableOutputPipelining;
        enableShaveDDRAccessOptimization = options.enableShaveDDRAccessOptimization;
        readStrategyFromJson = options.readStrategyFromJson;
        writeStrategyToJson = options.writeStrategyToJson;
        enableExplicitDistributedTensorAttr = options.enableExplicitDistributedTensorAttr;
    }
};

}  // namespace vpux
