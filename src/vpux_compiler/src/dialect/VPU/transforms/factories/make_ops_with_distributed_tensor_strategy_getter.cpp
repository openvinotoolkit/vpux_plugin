//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/make_ops_with_distributed_tensor_strategy_getter.hpp"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/logger.hpp"

using namespace vpux;

std::unique_ptr<IGreedilyPassStrategy> VPU::createMakeOpsWithDistributedTensorStrategyGetter(
        mlir::func::FuncOp funcOp, llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
        bool enableExplicitDistributedTensorAttr) {
    const auto arch = VPU::getArch(funcOp);

    switch (arch) {
    case VPU::ArchKind::NPU37XX: {
        return std::make_unique<arch37xx::MakeOpsWithDistributedTensorStrategy>(overlapParamsLookup,
                                                                                enableExplicitDistributedTensorAttr);
    }
    case VPU::ArchKind::NPU40XX: {
        return std::make_unique<arch40xx::MakeOpsWithDistributedTensorStrategy>(overlapParamsLookup,
                                                                                enableExplicitDistributedTensorAttr);
    }
    case VPU::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unable to get MakeOpsWithDistributedTensorStrategy for arch {0}", arch);
    }
    }
}
