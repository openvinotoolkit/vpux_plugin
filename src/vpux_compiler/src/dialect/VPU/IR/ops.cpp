//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/asm.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::LogicalResult VPU::sameLayout(VPU::DistributedTensorType inDistributedType,
                                    VPU::DistributedTensorType outDistributedType, LogCb logCb) {
    if (inDistributedType.getOrder() != outDistributedType.getOrder()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getOrder(),
                      outDistributedType.getOrder()));
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult VPU::sameLayout(VPUIP::DistributedBufferType inDistributedType,
                                    VPUIP::DistributedBufferType outDistributedType, LogCb logCb) {
    auto isContinuousWithSameOrder = [&]() {
        const auto inStrideReqs = StrideReqs::compact(inDistributedType.getShape().size());
        const auto outStrideReqs = StrideReqs::compact(outDistributedType.getShape().size());
        auto inRes = inStrideReqs.checkStrides(inDistributedType);
        auto outRes = outStrideReqs.checkStrides(outDistributedType);
        return inRes && outRes && inDistributedType.getDimsOrder() == outDistributedType.getDimsOrder();
    };

    // The strides will be checked when comparing the layouts. So the function will return true if the layouts are equal
    // or the buffers are compact with same dim order
    if (inDistributedType.getLayout() != outDistributedType.getLayout() && !isContinuousWithSameOrder()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getLayout(),
                      outDistributedType.getLayout()));
        return mlir::failure();
    }
    return mlir::success();
}

bool VPU::isVFNCESupported(VPU::NCEOpInterface op) {
    auto isOne = [](auto val) {
        return val == 1;
    };

    if (llvm::all_of(op.getStridesVal(), isOne)) {
        return true;
    }

    return false;
}

//
// materializeConstant
//

mlir::Operation* vpux::VPU::VPUDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                            mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize VPU Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType>()) {
        (void)errorAt(loc, "Can't materialize VPU Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, value.cast<Const::ContentAttr>());
}
