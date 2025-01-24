//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace vpux {
namespace VPU {

enum class SpillingType { SPILL_WRITE, SPILL_READ };

//
// LayerCostModel for layer cost estimation given by different strategies
//
class LayerCostModel final {
public:
    struct SpillingCost {
        double writeCost;
        double readCost;
    };

    explicit LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling, Logger log,
                            SiblingOpsAnalysis& siblingsOpsAnalysis);
    ~LayerCostModel() = default;

    double getLayerCost(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                        bool useTimeBasedCost = true);
    double getNCELayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy, bool useTimeBasedCost = true);
    double getSWLayerCost(VPU::SWOpInterface swOp, VPU::MultiClusterStrategy strategy) const;
    double getDPUandDMATimeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double getDPUandDMATimeCostWithCustomTiling(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                                const OutputTiling& outTiles) const;
    double getEfficiencyCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;

    bool hasMultiClusterStrategy(mlir::Operation* op) const;
    VPU::MultiClusterStrategy getMultiClusterStrategyValue(VPU::ClusteredOpInterface clusteredOp) const;
    std::pair<vpux::NDTypeInterface, vpux::NDTypeInterface> getDistributionTypesWithStrategy(
            VPU::ClusteredOpInterface parentOp, VPU::MultiClusterStrategy parentStrategy,
            VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userStrategy) const;
    std::pair<std::pair<mlir::Type, TensorDistributionMap>, std::pair<mlir::Type, TensorDistributionMap>>
    getDistributionsWithStrategy(VPU::ClusteredOpInterface parentOp, VPU::MultiClusterStrategy parentStrategy,
                                 VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userStrategy) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, vpux::NDTypeInterface srcTensorType,
                     vpux::NDTypeInterface dstTensorType) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, VPU::ClusteredOpInterface userOp) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                     VPU::ClusteredOpInterface userOp) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, VPU::ClusteredOpInterface userOp,
                     VPU::MultiClusterStrategy userOpStrategy) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                     VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userOpStrategy) const;
    bool hasSpilling(VPU::ClusteredOpInterface /*clusteredOp*/,
                     std::pair<mlir::Type, TensorDistributionMap>& srcTensorType,
                     std::pair<mlir::Type, TensorDistributionMap>& dstTensorType) const;

    bool doesLayerRequireTiling(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) const;
    bool doesLayerHaveVPUNNSupportedTypes(VPU::ClusteredOpInterface clusteredOp) const;
    double getSpillingReadCost(vpux::NDTypeInterface srcTensorType) const;
    double getSpillingWriteCost(vpux::NDTypeInterface srcTensorType) const;
    double getSpillingReadCost(vpux::NDTypeInterface srcTensorType, const TensorDistributionMap& distributions) const;

    double getSpillingWriteCost(vpux::NDTypeInterface srcTensorType, const TensorDistributionMap& distributions) const;
    SpillingCost getSpillingCost(vpux::NDTypeInterface srcTensorType, vpux::NDTypeInterface dstTensorType,
                                 VPU::ClusteredOpInterface parentOp, VPU::ClusteredOpInterface userOp) const;
    SpillingCost getSpillingCost(vpux::NDTypeInterface srcTensorType, const TensorDistributionMap& srcDistribution,
                                 vpux::NDTypeInterface dstTensorType, const TensorDistributionMap& dstDistribution,
                                 VPU::ClusteredOpInterface parentOp, VPU::ClusteredOpInterface userOp) const;
    SpillingCost calculateSpillingCost(VPU::ClusteredOpInterface parentOp, VPU::ClusteredOpInterface userOp,
                                       VPU::MultiClusterStrategy parentStrategy,
                                       VPU::MultiClusterStrategy userStrategy) const;
    vpux::NDTypeInterface getNormalInputType(VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp) const;
    vpux::NDTypeInterface getNormalOutputType(VPU::ClusteredOpInterface origOp) const;
    VPU::DistributedTypeInterface getDistributedInputType(VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
                                                          VPU::MultiClusterStrategy specifiedStrategy) const;
    VPU::DistributedTypeInterface getDistributedInputType(VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
                                                          VPU::MultiClusterStrategy specifiedStrategy,
                                                          mlir::ArrayAttr customAlignment) const;
    std::pair<mlir::Type, TensorDistributionMap> getInputWithDistribution(
            VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
            VPU::MultiClusterStrategy specifiedStrategy) const;
    std::pair<mlir::Type, TensorDistributionMap> getInputWithDistribution(
            VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp, VPU::MultiClusterStrategy specifiedStrategy,
            mlir::ArrayRef<int64_t> customAlignment) const;
    VPU::DistributedTypeInterface getDistributedOutputType(VPU::ClusteredOpInterface origOp,
                                                           VPU::MultiClusterStrategy specifiedStrategy) const;
    std::pair<mlir::Type, TensorDistributionMap> getOutputWithDistribution(
            VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy specifiedStrategy) const;

    VPU::MultiClusterStrategy getOptimalLayerStrategy(VPU::ClusteredOpInterface clusteredOp);
    bool isUnderSubgraphOpt() const;
    void setUnderSubgraphOpt(bool underSubgraphOpt);

    double static constexpr COST_MAX = std::numeric_limits<double>::infinity();

private:
    // CostCache has two-levels mappings:
    // The first-level mapping is from NCE op to op costs, op costs are represent by SmallVector<double>.
    // The second-level mapping is from op costs to op stratey cost value.
    using CostCache = DenseMap<VPU::NCEOpInterface, SmallVector<double>>;
    CostCache _costCache;

    double calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const;

    double computeSplitEfficiency(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double clusterComputeTime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double totalDMATime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double getDMACostOfType(vpux::NDTypeInterface srcTensorType, SpillingType spillingType) const;
    double getSpillingDMACost(vpux::NDTypeInterface srcTensorType, SpillingType spillingType) const;
    double getDMACostOfType(vpux::NDTypeInterface srcType, const DistributionInfo& distribution,
                            SpillingType spillingType) const;
    double getSpillingDMACost(vpux::NDTypeInterface srcTensorType, const TensorDistributionMap& distributions,
                              SpillingType spillingType) const;

    // Note that the cost model will be integrated with VPUNN as part of  E#37379
    // after that these platform specific variables will be removed
    double _NCEThroughput = 0.0;               // NCE throughput for U8 data, unit is MOPS
    const double _DDRLatency = 100;            // DDR latency is ~100 cycles per dma
    double _DMABandwidth = 0.0;                // Transition Bytes per cycle
    const double _CMXLatency = 5;              // Cycles, attempt to capture cost accessing CMX
    const double _CMXMulticastBandwidth = 32;  // 32 Bytes per cycle for multicast
    double _NCEFrequency = 0.0;                // NCE frequency, unit is MHz
    const int64_t _numChannelAlignment = 16;
    const int64_t _cmxAddressAlignment = 16;  // Kernel address alignment
    int64_t _numTiles = 0;                    // Number of Tiles
    int64_t _numDPUs = 0;                     // Number of DPUs per cluster
    int64_t _numShaveActs = 0;                // Number of ACT_SHVs per cluster
    int64_t _numDMAPorts = 1;                 // Number of the DMA ports
    VPU::ArchKind _arch;
    VPUNN::VPUDevice _vpuDeviceType;
    std::shared_ptr<VPUNN::VPULayerCostModel> _layerCostModel;
    mlir::func::FuncOp _func;
    bool _enablePrefetchTiling;
    Logger _log;
    SiblingOpsAnalysis& _siblingsOpsAnalysis;
    bool _underSubgraphOpt = false;
};

std::optional<VPU::MultiClusterStrategy> getDefaultLayerStrategy(VPU::ClusteredOpInterface clusteredOp);

bool isStrategyCompatibleShape(VPU::ClusteredOpInterface clusteredOp, const vpux::TileInfo& outputTile,
                               VPU::MultiClusterStrategy strategy, Logger log);

bool isStrategySOXCompatible(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                             size_t numTiles);

SmallVector<uint32_t> getDPUCostForNCEOp(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy mcStrategy,
                                         const OutputTiling& outTiles, const VPUIP::WorkloadCostParams& costParams,
                                         VPUNN::VPULayerStrategy vpunnStrategy,
                                         const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel, Logger log);

SmallVector<uint32_t> getPerTileWeightsDMACosts(
        VPU::NCEOpInterface nceOp, SiblingOpsAnalysis& siblingsAnalysis,
        ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc);

SmallVector<uint32_t> getPerTileActivationDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc,
        VPU::MultiClusterStrategy strategy, int64_t numTiles);

SmallVector<uint32_t> getPerTileOutputDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>> tilesTypes,
        std::function<uint32_t(NDTypeInterface, const TensorDistributionMap& distributions)> getSpillingReadCostFunc);

uint32_t getWeightsDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                   SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                   bool enablePrefetchTiling, vpux::Logger log);

uint32_t getActivationDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                      SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                      bool enablePrefetchTiling, vpux::Logger log);

uint32_t getOutputDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                  SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                  bool enablePrefetchTiling, vpux::Logger log);

size_t getNumNonConstantOperands(mlir::Operation* op);

bool hasLayerWithMultipleInputs(mlir::Operation* op);

bool isSingleBatchRequired(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
