//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// DDRAccessGatherOpModel
//

class DDRAccessGatherOpModel final : public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessGatherOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        auto gatherOp = mlir::dyn_cast<VPU::GatherOp>(op);
        VPUX_THROW_WHEN(gatherOp == nullptr, "Unexpected op {0} at '{1}'", op->getName(), op->getLoc());

        const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(gatherOp).to<Byte>().count();
        const auto inputType = gatherOp.getInput().getType().cast<vpux::NDTypeInterface>();
        const auto inputShape = inputType.getShape().raw();
        const auto inputByteSize = inputType.getElemTypeSize().to<Byte>().count();
        int64_t axisValue = gatherOp.getAxisValueAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
        const auto axisDimSizeBytes = inputShape[axisValue] * inputByteSize;

        // Can't get feasible tiling strategy because axis dimension of gatherOp can't be tiled.
        if (axisDimSizeBytes > cmxAvailableBytes) {
            log.nest(1).trace("Can't still fit into CMX after tiling. The case should be solved with DDR solution.");
            return true;
        }

        // "DDR Access" is preferred for scenarios with large inputs and small outputs
        // If the Output buffer exceeds CMX memory size, memory allocation follows:
        // Input (DDR) + Indices (CMX) -> Output (DDR)
        // Experiments indicate significant performance degradation (E#123794 for details)
        int64_t batchDims = 0;
        const auto batchDimAttr = gatherOp.getBatchDimsAttr();
        if (batchDimAttr != nullptr) {
            batchDims = batchDimAttr.dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
        }

        const auto indicesType = gatherOp.getIndices().getType().cast<vpux::NDTypeInterface>();
        const auto indicesRank = indicesType.getShape().size();
        if (batchDims == 0 && axisValue == 0 && indicesRank == 2) {
            const auto outputType = gatherOp.getOutput().getType().cast<vpux::NDTypeInterface>();
            const auto outputByteSize = outputType.getElemTypeSize().to<Byte>().count();
            const auto isBeneficialScenario = (inputType.getShape().totalSize() * inputByteSize) > cmxAvailableBytes &&
                                              (outputType.getShape().totalSize() * outputByteSize) < cmxAvailableBytes;
            if (isBeneficialScenario) {
                log.nest(1).trace("Gather layer {0} has large input and output buffer in CMX, DDR Access is preferred "
                                  "for better performance.",
                                  gatherOp);
                return true;
            }
        }

        return false;
    }
};

//
// DDRAccessGRUSequenceOpModel
//

class DDRAccessGRUSequenceOpModel final : public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessGRUSequenceOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        auto gruSequenceOp = mlir::dyn_cast<VPU::GRUSequenceOp>(op);
        auto outputShape = gruSequenceOp.getMiddleHiddenState().getType().cast<vpux::NDTypeInterface>().getShape();
        Shape minShapeAfterTiling(outputShape.size(), 1);
        minShapeAfterTiling[Dim(3)] = outputShape[Dim(3)];
        auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
        if (!iface.isSupportedTiling({TileInfo(minShapeAfterTiling)}, TilingMode::ISOLATED, log.nest())) {
            log.nest(1).trace("Can't still fit into CMX after tiling.DDR access will be used for GRUSequenceOp.");
            return true;
        }
        return false;
    }
};

//
// DDRAccessGRUSequenceLastPartOpModel
//

class DDRAccessGRUSequenceLastPartOpModel final :
        public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessGRUSequenceLastPartOpModel> {
public:
    bool isDDRAccessNecessaryOrBeneficial(mlir::Operation* op, Logger log) const {
        auto gruSequenceLastPartOp = mlir::dyn_cast<VPU::GRUSequenceLastPartOp>(op);
        auto outputShape =
                gruSequenceLastPartOp.getMiddleHiddenState().getType().cast<vpux::NDTypeInterface>().getShape();
        Shape minShapeAfterTiling(outputShape.size(), 1);
        minShapeAfterTiling[Dim(3)] = outputShape[Dim(3)];
        auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
        if (!iface.isSupportedTiling({TileInfo(minShapeAfterTiling)}, TilingMode::ISOLATED, log.nest())) {
            log.nest(1).trace(
                    "Can't still fit into CMX after tiling.DDR access will be used for GRUSequenceLastPartOp.");
            return true;
        }
        return false;
    }
};

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPU::arch37xx::registerDDRAccessOpModelInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::GatherOp::attachInterface<DDRAccessGatherOpModel>(*ctx);
        VPU::GRUSequenceOp::attachInterface<DDRAccessGRUSequenceOpModel>(*ctx);
        VPU::GRUSequenceLastPartOp::attachInterface<DDRAccessGRUSequenceLastPartOpModel>(*ctx);
    });
}
