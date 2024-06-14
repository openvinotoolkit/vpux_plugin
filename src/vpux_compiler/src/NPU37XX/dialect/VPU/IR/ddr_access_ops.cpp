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
// DDRAccessOpModel
//

class DDRAccessOpModel final : public VPU::DDRAccessOpInterface::FallbackModel<DDRAccessOpModel> {
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

        // DDR access is preferred for Gather layer with large input but small output
        int64_t batchDims = 0;
        const auto batchDimAttr = gatherOp.getBatchDimsAttr();
        if (batchDimAttr != nullptr) {
            batchDims = batchDimAttr.dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
        }

        const auto indicesType = gatherOp.getIndices().getType().cast<vpux::NDTypeInterface>();
        const auto indicesShape = indicesType.getShape().raw();
        const auto indicesRank = indicesShape.size();
        if (batchDims == 0 && axisValue == 0 && indicesRank == 2) {
            if (inputShape[axisValue] / (indicesShape[indicesRank - 1]) >= DDR_ACCESS_GATHER_IO_RATIO) {
                log.nest(1).trace("Gather layer {0} has large input size but small output size, DDR access "
                                  "is preferred for better performance.",
                                  gatherOp);
                return true;
            }
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
        VPU::GatherOp::attachInterface<DDRAccessOpModel>(*ctx);
    });
}
