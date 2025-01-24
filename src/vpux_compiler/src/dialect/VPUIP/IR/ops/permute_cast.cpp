//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::Value VPUIP::PermuteCastOp::getViewSource() {
    return getSource();
}

//
// fold
//

mlir::OpFoldResult vpux::VPUIP::PermuteCastOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        auto restored = static_cast<Const::ContentAttr>(attr);
        if (restored.getType().getShape() != getShape(getResult())) {
            restored = restored.transform().reshape(getShape(getResult())).get();
        }
        return restored.transform().reorder(DimsOrder::fromAffineMap(getDstOrder())).get();
    }

    return nullptr;
}

mlir::LogicalResult vpux::VPUIP::PermuteCastOp::verify() {
    const auto op = getOperation();
    auto distributedInType = getSource().getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributedOutType = getResult().getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedInType && distributedOutType) {
        auto outputDistribution = distributedOutType.getDistribution();

        auto expectedOutputDistribution = applyPermutationOnDistributionInfoAttr(
                distributedInType, getMemPerm(), distributedInType.getDimsOrder(), distributedOutType.getDimsOrder(),
                distributedInType.getShape(), distributedOutType.getShape());
        if (mlir::failed(expectedOutputDistribution)) {
            return errorAt(op, "PermuteCast unsupported input distribution: in = {0}",
                           distributedInType.getDistribution());
        }

        if (outputDistribution != expectedOutputDistribution.value()) {
            return errorAt(op,
                           "PermuteCast input and output distributions are incompatible: in = {0}, out = {1},"
                           "expected = {2}",
                           distributedInType.getDistribution(), outputDistribution, expectedOutputDistribution.value());
        }
    }

    const auto inType = getSource().getType().cast<vpux::NDTypeInterface>();
    const auto outType = getResult().getType().cast<vpux::NDTypeInterface>();

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "PermuteCast input and output must have the same number of elements");
    }

    return mlir::success();
}
