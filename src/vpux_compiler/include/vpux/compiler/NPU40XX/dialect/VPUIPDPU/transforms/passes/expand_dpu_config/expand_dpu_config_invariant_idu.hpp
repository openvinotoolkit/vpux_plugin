//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIPDPU::arch40xx::IDU {

struct IDUConfig {
    struct InActivations {
        bool inSparse = false;
    } inActivations;
    struct Weights {
        mlir::Type wMode;
        std::optional<int64_t> poolWtData;
        bool wtSparse = false;
    } weights;
    struct InputLayerCfg {
        int64_t sparsityPattern = 0;
        bool inputCompressed = false;
    } inputLayerCfg;
    struct StorageElement {
        int64_t seSize = 0;
        std::optional<int64_t> numSEsInZDir;
    } storageElement;
    struct Kernel {
        int64_t kernelX = 1;
        int64_t kernelY = 1;
    } kernel;
    struct Stride {
        int64_t strideX = 1;
        int64_t strideY = 1;
    } stride;
    struct WorkloadCfg {
        IDUWorkloadType workloadType = IDUWorkloadType::CONV;
    } workloadCfg;
    struct DepthWiseCfg {
        bool dw3x3s2OptDisable = false;
        std::optional<int64_t> dwOptOffset;
    } depthWiseCfg;
    struct EltWiseCfg {
        bool eltWiseCfgOp = false;
        bool elopScapeFp = false;
        bool bf16FlowOn = false;
        int64_t elopScaleA = 1;
        int64_t elopScaleB = 1;
        float fpElopScaleA = 1;
        float fpElopScaleB = 1;
    } eltWiseCfg;
};

struct PPETask {
    std::optional<SmallVector<int64_t>> in1QuantMult;
    std::optional<SmallVector<int64_t>> in2QuantMult;
    std::optional<SmallVector<float>> in1QuantMultFp;
    std::optional<SmallVector<float>> in2QuantMultFp;
    std::optional<IDUEltwiseType> eltwiseType;
};

PPETask evalPPETasks(mlir::Region& ppeRegion, std::optional<VPUIP::NCETaskType> taskType);

mlir::LogicalResult configureIDU(const Logger& log, IDUConfig& config, const vpux::NDTypeInterface& inActType,
                                 mlir::Type weightsElementType, VPUIP::NCETaskType taskType,
                                 std::optional<int64_t> spPattern, std::optional<bool> inChannelsCompression,
                                 std::optional<bool> smallKernelOptimization, bool inActSparse, bool weightsSparse,
                                 std::optional<mlir::ArrayAttr> kernelSize,
                                 std::optional<mlir::ArrayAttr> kernelStrides, std::optional<int64_t> seSize,
                                 const PPETask& ppeTask, VPU::ArchKind arch);

mlir::LogicalResult buildIDUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const IDUConfig& config,
                                   mlir::Value inAct);

}  // namespace vpux::VPUIPDPU::arch40xx::IDU
