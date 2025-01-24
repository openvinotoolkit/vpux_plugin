//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/device.hpp"

#include <llvm/Support/JSON.h>

#include <array>
#include <fstream>
#include <optional>
#include <set>
#include <string>

namespace nb {
enum class CaseType {
    DMA,
    ZMajorConvolution,
    SparseZMajorConvolution,
    DepthWiseConv,
    DoubleZMajorConvolution,
    EltwiseDense,
    EltwiseMultDW,
    EltwiseSparse,
    MaxPool,
    AvgPool,
    DifferentClustersDPU,
    MultiClustersDPU,
    HaloMultiClustering,
    ActShave,
    ReadAfterWriteDPUDMA,
    ReadAfterWriteDMADPU,
    ReadAfterWriteACTDMA,
    ReadAfterWriteDMAACT,
    ReadAfterWriteDPUACT,
    ReadAfterWriteACTDPU,
    RaceConditionDMA,
    RaceConditionDPU,
    RaceConditionDPUDMA,
    RaceConditionDPUDMAACT,
    RaceConditionDPUACT,
    RaceCondition,
    M2iTask,
    StorageElementTableDPU,
    DualChannelDMA,
    DMACompressActDense,
    DMACompressActSparse,
    GatherDMA,
    GenerateScaleTable,
    ReduceMean,
    ReduceSumSquare,
    ReduceOut,
    Unknown
};

std::string to_string(CaseType case_);
CaseType to_case(llvm::StringRef str);

enum class BackendFlow { Default, WLMPartial };
std::string to_string(BackendFlow backendFlow);

enum class WeightTableFormats { WT_DEFAULT, WT_LEGACY };
std::string to_string(WeightTableFormats weightTableFormat);

enum class DType { U1, U2, U3, U4, U5, U6, I4, U8, I8, I32, BF8, HF8, FP16, FP32, BF16, I64, U64, UNK };

enum class PalletMode { NO_PLT = 0, ONE_BIT_PLT = 1, TWO_BIT_PLT = 2, FOUR_BIT_PLT = 3 };

// For CMX the enum constant values correspond to the tile indexes
enum class MemoryLocation { CMX0 = 0, CMX1 = 1, CMX2 = 2, CMX3 = 3, CMX4 = 4, CMX5 = 5, DDR, Unknown };
MemoryLocation to_memory_location(llvm::StringRef str);
std::string to_string(MemoryLocation memoryLocation);

DType to_dtype(llvm::StringRef str);
std::string to_string(DType dtype);
PalletMode to_palletMode(llvm::StringRef str);
unsigned to_pltDataWidth(PalletMode mode);
vpux::VPUIP::Permutation to_odu_permutation(llvm::StringRef str);
struct QuantParams {
    bool present = false;
    std::vector<double> scale;
    std::int64_t zeropoint = 0;
    std::int64_t low_range = 0;
    std::int64_t high_range = 0;
};

struct PalletTableInfo {
    PalletMode pMode{PalletMode::NO_PLT};
    DType quantileType{DType::UNK};
    unsigned quantileLUTSize{};
    // it might be preferable to not carry the quantile table directly in the
    // type itself due to the fact that it can have u8/i8/bf16/fp16 types
    std::vector<double> quantileLUT;
    std::string filename;
};

enum class SegmentationType { SOK = 1, SOH = 2, SOW = 3, SOHW = 4, SOHK = 5, SOHK3 = 6, SOHW3 = 7 };
std::string to_string(SegmentationType segmentationType);

enum class SwizzlingKey { key0, key1, key2, key3, key4, key5 };

template <typename E>
constexpr auto to_underlying(E e) -> typename std::underlying_type<E>::type {
    return static_cast<typename std::underlying_type<E>::type>(e);
}

struct Shape {};

// Input and weight layers have similar structure in rtl config descriptor
struct InputLayer {
    DType dtype = DType::UNK;
    QuantParams qp;
    std::array<std::int64_t, 4> shape = {0};
};

struct WeightLayer {
    DType dtype = DType::UNK;
    QuantParams qp;
    PalletTableInfo plt;
    std::array<std::int64_t, 4> shape = {0};
    std::string filename;
};

struct SM {
    DType dtype = DType::U1;
    std::array<std::int64_t, 4> shape = {0};
};

struct GatherIndices {
    DType dtype = DType::U64;
    std::array<std::int64_t, 4> shape = {0};
};

struct DMAparams {
    MemoryLocation srcLocation = MemoryLocation::Unknown;
    // multiple destination locations for DMA broadcast (otherwise one location)
    std::vector<MemoryLocation> dstLocations;
    MemoryLocation indicesLocation = MemoryLocation::Unknown;
    int64_t engine = 0;
    bool doConvert = false;
    bool zeroSizeTask = false;
    bool testMemSideCache = false;
    bool cacheTrashing = false;
    bool cacheEnabled = false;
};

struct ConvLayer {
    std::array<std::int64_t, 2> stride = {0};
    std::array<std::int64_t, 4> pad = {0};
    std::int64_t group = 0;
    std::int64_t dilation = 0;
    bool act_sparsity = false;
    bool compress = false;
    vpux::VPU::MPEMode cube_mode = vpux::VPU::MPEMode::CUBOID_16x16;
};

// IDU_CMX_MUX_MODE
enum class ICM_MODE { DEFAULT = 0, MODE_0 = 1, MODE_1 = 2, MODE_2 = 3 };
struct EltwiseLayer {
    std::int64_t seSize = 0;
    vpux::VPU::EltwiseType mode = vpux::VPU::EltwiseType::ADD;
    ICM_MODE iduCmxMuxMode = ICM_MODE::DEFAULT;
};

struct PoolLayer {
    std::string pool_type = "max";
    std::array<std::int64_t, 2> kernel_shape = {0};
    std::array<std::int64_t, 2> stride = {0};
    std::array<std::int64_t, 4> pad = {0};
    std::int64_t group = 0;
    std::int64_t dilation = 0;
};

struct RaceConditionParams {
    size_t iterationsCount = 0;
    size_t requestedClusters = 0;

    // requested DPU's per cluster for DPUop case
    // requested ActShave's per cluster for ActShaveOp case
    // requested DMA engines for DMAop case
    size_t requestedUnits = 0;
};

struct DPUTaskParams {
    std::size_t inputCluster = 0;
    std::vector<std::size_t> outputClusters;
    std::size_t weightsCluster = 0;
    std::size_t weightsTableCluster = 0;
};

struct MultiClusterDPUParams {
    std::vector<std::size_t> taskClusters;
    SegmentationType segmentation = SegmentationType::SOK;
    bool broadcast = false;
};

struct HaloParams {
    // clusters on which the DPUTasks will run
    std::vector<std::size_t> taskClusters;
    // type of multiclustering tensor segmentation (SOK, SOH, SOW etc.)
    SegmentationType segmentation = SegmentationType::SOK;
    // number of clusters assigned per each dim, for segmentation on multiple dims
    // e.g. SOHK clustersPerDim = {2, 3} -> height is split in 2, output channels are split in 3
    // e.g. SOHW clustersPerDim = {3, 2} -> height is split in 3, width is split in 2
    std::vector<std::size_t> clustersPerDim;
    // halo size over height, to be used for strategies SOH, SOHK
    int64_t heightHaloSize = 0;
    // halo size over width, to be used for strategies SOW, SOHW
    int64_t widthHaloSize = 0;
};

struct OutputLayer {
    std::array<std::int64_t, 4> shape = {0};
    DType dtype = DType::UNK;
    QuantParams qp;
};

struct ReduceOutLayer {
    bool doReduceMaxPerXY = false;
    bool doReduceMinPerXY = false;
    bool doReduceMinMaxPerTensor = false;
    llvm::SmallVector<OutputLayer> output;
    bool isMultiTile = false;
};

enum class ActivationType {
    None,
    ReLU,
    ReLUX,
    LeakyReLU,
    Mish,
    HSwish,
    Sigmoid,
    Softmax,
    round_trip_b8h8_to_fp16,
    sau_sumx_fp16_to_fp32,
    cmu_perm_x8,
    cmu_perm,
    PopulateWeightTable,
    Rsqrt,
    Sin,
    Tanh,
    Cos,
    Unknown
};

ActivationType to_activation_type(llvm::StringRef str);
std::string to_string(ActivationType activationType);

struct ActivationLayer {
    ActivationType activationType = ActivationType::None;
    double alpha = 1.0;
    double maximum = 0;
    size_t axis = 0;
    std::optional<int64_t> weightsOffset = std::nullopt;
    std::optional<int64_t> weightsPtrStep = std::nullopt;
    // TODO: add support for activation functions that take parameters
    int64_t permBlend = 0;
};

// M2I i/o color formats
enum class M2iFmt {
    PL_YUV420_8,
    SP_NV12_8,
    PL_FP16_RGB,
    PL_RGB24,
    IL_RGB888,
    IL_BGR888,
    Unknown
    // Note: other less common formats exist
};
M2iFmt to_m2i_fmt(llvm::StringRef str);

enum class M2iInterp { NEAREST, BILINEAR, UNKNOWN };

M2iInterp to_m2i_interp(llvm::StringRef str);

struct M2iLayer {
    bool doCsc = false;     // do color-space-conversion
    bool doNorm = false;    // do normalization
    bool doTiling = false;  // do tiling
    M2iFmt iFmt = M2iFmt::Unknown;
    M2iFmt oFmt = M2iFmt::Unknown;
    std::vector<int> outSizes;              // output sizes (optional)
    std::vector<float> normCoefs;           // normalization coefs (optional)
    M2iInterp interp = M2iInterp::UNKNOWN;  // interpolation
};

enum class SETablePattern { SwitchLines, OriginalInput };

struct SETableParams {
    bool seOnlyEn = false;
    SETablePattern seTablePattern = SETablePattern::SwitchLines;
};

struct ProfilingParams {
    bool dpuProfilingEnabled = false;
    bool dmaProfilingEnabled = false;
    bool swProfilingEnabled = false;
    bool m2iProfilingEnabled = false;
    bool workpointEnabled = false;

    bool profilingEnabled() const {
        return dpuProfilingEnabled || dmaProfilingEnabled || swProfilingEnabled || m2iProfilingEnabled ||
               workpointEnabled;
    }
};

struct WLMParams {
    bool isWLMPartialEnabled = false;
};

struct ActShaveBroadcastingParams {
    MemoryLocation srcLocation = MemoryLocation::Unknown;
    // one or many destination locations for ActShave broadcast
    std::vector<MemoryLocation> dstLocations;
};

class TestCaseJsonDescriptor {
public:
    TestCaseJsonDescriptor(llvm::StringRef jsonString = "");
    TestCaseJsonDescriptor(llvm::json::Object jsonObject);
    void parse(llvm::json::Object jsonObject);
    llvm::SmallVector<InputLayer> getInputLayerList() const {
        return inLayers_;
    }
    llvm::SmallVector<SM> getInputSMList() const {
        return inSMs_;
    }
    GatherIndices getInputGatherIndices() const {
        return gatherIndices_;
    }
    llvm::SmallVector<WeightLayer> getWeightLayers() const {
        return wtLayer_;
    }
    llvm::SmallVector<SM> getWeightSMs() const {
        return wtSMs_;
    }
    llvm::SmallVector<OutputLayer> getOutputLayers() const {
        return outLayers_;
    }
    DMAparams getDMAparams() const {
        return DMAparams_;
    }
    ConvLayer getConvLayer() const {
        return convLayer_;
    }
    EltwiseLayer getEltwiseLayer() const {
        return eltwiseLayer_;
    }
    PoolLayer getPoolLayer() const {
        return poolLayer_;
    }
    ActivationLayer getActivationLayer() const {
        return activationLayer_;
    }
    M2iLayer getM2iLayer() const {
        return m2iLayer_;
    }
    RaceConditionParams getRaceConditionParams() const {
        return raceConditionParams_;
    }
    DPUTaskParams getDPUTaskParams() const {
        return DPUTaskParams_;
    }
    MultiClusterDPUParams getMultiClusterDPUParams() const {
        return multiClusterDPUParams_;
    }
    HaloParams getHaloParams() const {
        return haloParams_;
    }
    CaseType getCaseType() const {
        return caseType_;
    }
    llvm::StringRef getKernelFilename() const {
        return kernelFilename_;
    }
    std::string getCaseStr() const {
        return caseTypeStr_;
    }
    vpux::VPU::PPEMode getPPELayerType() const {
        return ppeLayerType_;
    }
    vpux::VPUIP::Permutation getODUPermutation() const {
        return odu_permutation_;
    }
    std::size_t getIterationCount() const {
        return iterationCount_;
    }
    std::size_t getClusterNumber() const {
        return clusterNumber_;
    }
    std::size_t getNumClusters() const {
        return numClusters_;
    }
    SwizzlingKey getWeightsSwizzlingKey() const {
        return weightsSwizzlingKey_;
    }
    SwizzlingKey getActivationSwizzlingKey() const {
        return activationSwizzlingKey_;
    }
    SETableParams getSETableParams() const {
        return seTableParams_;
    }
    std::shared_ptr<TestCaseJsonDescriptor> getUnderlyingOp() const {
        return underlyingOp_;
    }

    vpux::VPU::ArchKind getArchitecture() const {
        return architecture_;
    }

    ProfilingParams getProfilingParams() const {
        return profilingParams_;
    }

    vpux::VPUIP::NCETaskType getReductionType() const {
        return reductionType_;
    }

    ReduceOutLayer getReduceOutLayer() const {
        return reduceOutLayer_;
    }

    WLMParams getWLMParams() const {
        return WLMParams_;
    }

    ActShaveBroadcastingParams getActShaveBroadcastingParams() const {
        return actShaveBroadcastingParams_;
    }
    std::optional<WeightTableFormats> getWeightTableFormat() const {
        return WeightTableFormat_;
    }

    bool ArePalletizationTypesLegal(const WeightLayer&) const;

private:
    llvm::SmallVector<InputLayer> loadInputLayer(llvm::json::Object* jsonObj);
    llvm::SmallVector<WeightLayer> loadWeightLayer(llvm::json::Object* jsonObj);
    llvm::SmallVector<SM> loadInputSMs(llvm::json::Object* jsonObj);
    llvm::SmallVector<SM> loadWeightSMs(llvm::json::Object* jsonObj);
    llvm::SmallVector<OutputLayer> loadOutputLayer(llvm::json::Object* jsonObj);
    DMAparams loadDMAParams(llvm::json::Object* jsonObj);
    GatherIndices loadGatherIndices(llvm::json::Object* jsonObj);
    ConvLayer loadConvLayer(llvm::json::Object* jsonObj);
    EltwiseLayer loadEltwiseLayer(llvm::json::Object* jsonObj);
    PoolLayer loadPoolLayer(llvm::json::Object* jsonObj);
    ActivationLayer loadActivationLayer(llvm::json::Object* jsonObj);
    M2iLayer loadM2iLayer(llvm::json::Object* jsonObj);
    CaseType loadCaseType(llvm::json::Object* jsonObj);
    QuantParams loadQuantizationParams(llvm::json::Object* obj);
    PalletTableInfo loadPalletTableInfoLayers(llvm::json::Object* obj);
    RaceConditionParams loadRaceConditionParams(llvm::json::Object* obj);
    DPUTaskParams loadDPUTaskParams(llvm::json::Object* obj);
    MultiClusterDPUParams loadMultiClusterDPUParams(llvm::json::Object* obj);
    HaloParams loadHaloTaskParams(llvm::json::Object* jsonObj);
    std::size_t loadIterationCount(llvm::json::Object* obj);
    std::size_t loadClusterNumber(llvm::json::Object* obj);
    std::size_t loadNumClusters(llvm::json::Object* obj);
    SETableParams loadSETableParams(llvm::json::Object* obj);
    SwizzlingKey loadSwizzlingKey(llvm::json::Object* obj, std::string keyType);
    ProfilingParams loadProfilingParams(llvm::json::Object* obj);
    vpux::VPUIP::NCETaskType loadReductionType(llvm::json::Object* obj);
    ReduceOutLayer loadReduceOutLayer(llvm::json::Object* obj);
    WLMParams loadWLMParams(llvm::json::Object* obj);
    std::optional<WeightTableFormats> loadWeightTableFormat(llvm::json::Object* obj);
    ActShaveBroadcastingParams loadActShaveBroadcastingParams(llvm::json::Object* obj);

    CaseType caseType_ = CaseType::Unknown;
    DMAparams DMAparams_;
    ConvLayer convLayer_;
    EltwiseLayer eltwiseLayer_;
    PoolLayer poolLayer_;
    llvm::SmallVector<InputLayer> inLayers_;
    llvm::SmallVector<SM> inSMs_;
    llvm::SmallVector<WeightLayer> wtLayer_;
    llvm::SmallVector<SM> wtSMs_;
    llvm::SmallVector<OutputLayer> outLayers_;
    ActivationLayer activationLayer_;
    GatherIndices gatherIndices_;
    M2iLayer m2iLayer_;
    std::string kernelFilename_;
    std::string caseTypeStr_;
    vpux::VPU::PPEMode ppeLayerType_ = vpux::VPU::PPEMode::ADD;
    vpux::VPUIP::Permutation odu_permutation_ = vpux::VPUIP::Permutation::Permutation_ZXY;
    std::size_t iterationCount_ = 0;
    std::size_t clusterNumber_ = 0;
    std::size_t numClusters_ = 0;
    std::shared_ptr<TestCaseJsonDescriptor> underlyingOp_;
    SwizzlingKey weightsSwizzlingKey_ = SwizzlingKey::key0;
    SwizzlingKey activationSwizzlingKey_ = SwizzlingKey::key0;
    RaceConditionParams raceConditionParams_;
    DPUTaskParams DPUTaskParams_;
    MultiClusterDPUParams multiClusterDPUParams_;
    HaloParams haloParams_;
    SETableParams seTableParams_;
    vpux::VPU::ArchKind architecture_ = vpux::VPU::ArchKind::UNKNOWN;
    ProfilingParams profilingParams_;
    vpux::VPUIP::NCETaskType reductionType_ = vpux::VPUIP::NCETaskType::REDUCEMEAN;
    ReduceOutLayer reduceOutLayer_;
    WLMParams WLMParams_;
    std::optional<WeightTableFormats> WeightTableFormat_;
    ActShaveBroadcastingParams actShaveBroadcastingParams_;
};

}  // namespace nb
