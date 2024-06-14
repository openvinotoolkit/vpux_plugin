//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FileSystem.h>

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

namespace {

constexpr int64_t numDPU = 5;
constexpr int64_t maxSplitNum = 5;
constexpr int64_t initDimensionValue = 16;
constexpr int64_t maxDimensionValue = 64;
constexpr int64_t testStep = 2;

struct NceOpTensorShape {
    NceOpTensorShape(vpux::ShapeRef input, vpux::ShapeRef output): inputShape(input.raw()), outputShape(output.raw()) {
    }
    vpux::Shape inputShape;
    vpux::Shape outputShape;
};

vpux::VPUIP::WorkloadCostParams buildWorkloadCost(const NceOpTensorShape& tensorShape, mlir::MLIRContext* ctx) {
    vpux::VPUIP::WorkloadCostParams costParams;
    costParams.inDataType = mlir::Float16Type::get(ctx);
    costParams.outDataType = mlir::Float16Type::get(ctx);
    costParams.fullInputShape = tensorShape.inputShape;
    costParams.inputShape = tensorShape.inputShape;
    costParams.outputShape = tensorShape.outputShape;
    costParams.padInfo = vpux::PadInfo(0, 0, 0, 0);
    costParams.kernelSize = {1, 1};
    costParams.kernelStride = {1, 1};
    costParams.nceTaskType = vpux::VPUIP::NCETaskType::CONV;
    costParams.arch = vpux::VPU::ArchKind::NPU30XX;
    costParams.numDPU = numDPU;
    return costParams;
}
}  // namespace

TEST(MLIR_VPU_WorkloadCost, VPUNNCostInterface) {
    mlir::MLIRContext ctx;

    // For VPUX30XX, only VECTOR & MATRIX mode are supported. CUBOID mode is not supported.
    llvm::SmallVector<vpux::VPU::MPEMode> mpeModeList{vpux::VPU::MPEMode::VECTOR_FP16, vpux::VPU::MPEMode::VECTOR,
                                                      vpux::VPU::MPEMode::MATRIX};

    const auto costModel = vpux::VPU::createCostModel(vpux::VPU::ArchKind::NPU30XX);

    llvm::SmallVector<NceOpTensorShape> testTensorLists;
    for (int64_t h = initDimensionValue; h < maxDimensionValue; h *= testStep) {
        for (int64_t w = initDimensionValue; w < maxDimensionValue; w *= testStep) {
            for (int64_t c = initDimensionValue; c < maxDimensionValue; c *= testStep) {
                NceOpTensorShape tensorShape(vpux::ShapeRef({1, c, h, w}), vpux::ShapeRef({1, c, h, w}));
                testTensorLists.push_back(std::move(tensorShape));
            }
        }
    }
    for (auto& testTensor : testTensorLists) {
        for (auto& mpeMode : mpeModeList) {
            auto costParams = buildWorkloadCost(testTensor, &ctx);

            vpux::VPUIP::DpuTiler dpuTiler(costParams.outputShape, mpeMode);

            const auto splitNumPool = dpuTiler.generateSplitNumberPool(numDPU, maxSplitNum);

            vpux::Shape nTilesOnDim(costParams.outputShape.size(), 1);
            auto alignment = llvm::SmallVector<int64_t>(costParams.outputShape.size(), 1);
            alignment[vpux::Dims4D::Act::C.ind()] = vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
            const auto optionalAlignment = std::optional<llvm::ArrayRef<int64_t>>(alignment);
            auto outTilesWithSingleSplit =
                    vpux::fillDividedTiles(nTilesOnDim, costParams.outputShape, optionalAlignment);
            VPUX_THROW_WHEN(mlir::failed(outTilesWithSingleSplit), "Invalid tiling");

            vpux::VPUIP::WorkloadSplit singleSplit;
            for (auto& outTile : outTilesWithSingleSplit.value()) {
                singleSplit.emplace_back(std::move(outTile), mpeMode);
            }

            auto baseHardwareExecutionCost =
                    vpux::VPUIP::computeSplitCostForVPUX30XX(singleSplit, costParams, costModel);

            vpux::VPUIP::WorkloadSplitPool splitPool;

            dpuTiler.tileOverH(numDPU, splitPool);
            for (auto& splitNum : splitNumPool) {
                dpuTiler.tileOverZ(splitNum, splitPool);
            }

            for (auto iter = splitPool.begin(); iter != splitPool.end(); iter++) {
                auto hardwareExecutionCost = vpux::VPUIP::computeSplitCostForVPUX30XX(*iter, costParams, costModel);
                EXPECT_LE(hardwareExecutionCost, baseHardwareExecutionCost);
            }
        }
    }
}
