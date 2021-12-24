//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// TaskOpInterface
//

void vpux::VPUIP::getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects) {
    if (auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op)) {
        for (const auto input : layer.getInputs()) {
            auto inputType = input.getType().cast<mlir::MemRefType>();
            auto* resource = VPU::getMemoryResource(inputType);
            effects.emplace_back(mlir::MemoryEffects::Read::get(), input, resource);
        }

        for (const auto output : layer.getOutputs()) {
            auto outputType = output.getType().cast<mlir::MemRefType>();
            auto* resource = VPU::getMemoryResource(outputType);
            effects.emplace_back(mlir::MemoryEffects::Write::get(), output, resource);
        }
    }
}

mlir::Attribute vpux::VPUIP::getExecutorAttr(uint32_t& numUnits, mlir::Operation* op, VPU::ExecutorKind kind,
                                             Optional<int64_t> opNumUnits) {
    const auto kindAttr = VPU::ExecutorKindAttr::get(op->getContext(), kind);

    if (opNumUnits.hasValue()) {
        numUnits = checked_cast<uint32_t>(opNumUnits.getValue());
    } else {
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto resources = IE::RunTimeResourcesOp::getFromModule(module);
        auto available = resources.getExecutor(kindAttr);
        VPUX_THROW_UNLESS(available != nullptr, "Executor for '{0}' is not available", kind);
        numUnits = checked_cast<uint32_t>(available.count());
    }

    return kindAttr;
}

mlir::Attribute vpux::VPUIP::getTaskOpExecutor(mlir::Operation* op, uint32_t& numUnits) {
    auto task = mlir::cast<VPUIP::TaskOpInterface>(op);
    const auto executor = task.getExecutorKind();

    switch (executor) {
    case VPU::ExecutorKind::DMA_NN:
        return VPUIP::getExecutorAttr(numUnits, op, VPU::ExecutorKind::DMA_NN, 1);
    case VPU::ExecutorKind::NCE:
        return VPUIP::getExecutorAttr(numUnits, op, VPU::ExecutorKind::NCE, 1);
    case VPU::ExecutorKind::SHAVE_ACT:
        return VPUIP::getExecutorAttr(numUnits, op, VPU::ExecutorKind::SHAVE_ACT, 1);
    case VPU::ExecutorKind::SHAVE_UPA: {
        auto upaTask = mlir::cast<VPUIP::UPATaskOpInterface>(op);
        return VPUIP::getExecutorAttr(numUnits, op, VPU::ExecutorKind::SHAVE_UPA, upaTask.maxShaves());
    }
    default:
        VPUX_THROW("Unsupported executor '{0}'", executor);
    }
}

//
// UPATaskOpInterface
//

mlir::LogicalResult vpux::VPUIP::verifyUPATask(mlir::Operation* op) {
    auto task = mlir::dyn_cast<TaskOpInterface>(op);
    if (task == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement VPUIP Task interface", op->getName());
    }

    auto upaTask = mlir::dyn_cast<UPATaskOpInterface>(op);
    if (upaTask == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement VPUIP UPATask interface", op->getName());
    }

    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement RT Layer interface", op->getName());
    }

    for (auto& operand : layer.getOpOperands()) {
        const auto opVal = operand.get();
        const auto type = opVal.getType().cast<mlir::MemRefType>();
        const auto mem = VPU::getMemoryKind(type);

        if (type.getRank() == 0) {
            return errorAt(op, "SCALARS are not supported");
        }

        if (mem == VPU::MemoryKind::CMX_NN) {
            return errorAt(op, "Can't operate with '{0}' memory", mem);
        }

        const auto strideReqs = StrideReqs::simple(type.getRank());

        if (!strideReqs.checkStrides(opVal)) {
            return errorAt(op, "Value '{0}' strides do not match requirements '{1}'", opVal, strideReqs);
        }
    }

    if (upaTask.maxShaves().hasValue()) {
        auto resources = IE::RunTimeResourcesOp::getFromModule(op->getParentOfType<mlir::ModuleOp>());
        if (resources == nullptr) {
            return errorAt(op, "Missing IERT run-time resources definition");
        }

        auto available =
                resources.getExecutor(VPU::ExecutorKindAttr::get(op->getContext(), VPU::ExecutorKind::SHAVE_UPA));
        if (available == nullptr) {
            return errorAt(op, "SHAVE_UPA executor is not avaialble in run-time");
        }
        if (upaTask.maxShaves().getValue() > available.count()) {
            return errorAt(op, "maxShaves attribute '{0}' exceeds available count '{1}'", upaTask.maxShaves(),
                           available.count());
        }
    }

    return mlir::success();
}

//
// Legacy4D
//

mlir::LogicalResult vpux::VPUIP::verifyLegacy4D(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement RT Layer interface", op->getName());
    }

    for (const auto& val : layer.getOpOperands()) {
        const auto shape = getShape(val.get());
        const auto order = DimsOrder::fromValue(val.get());

        if (shape.size() != 3 && shape.size() != 4) {
            return errorAt(op, "Got unsupported shape '{0}', only 3D/4D are supported", shape);
        }

        if (shape.size() == 3) {
            if (order != DimsOrder::CHW && order != DimsOrder::HWC) {
                return errorAt(op, "Got unsupported input DimsOrder '{0}', only CHW and HWC are supported", order);
            }
        } else if (shape.size() == 4) {
            if (order != DimsOrder::NCHW && order != DimsOrder::NHWC) {
                return errorAt(op, "Got unsupported input DimsOrder '{0}', only NCHW and NHWC are supported", order);
            }

            if (shape.front() != 1) {
                return errorAt(op, "Batch size != 1 is not supported");
            }
        }
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/ops_interfaces.cpp.inc>
