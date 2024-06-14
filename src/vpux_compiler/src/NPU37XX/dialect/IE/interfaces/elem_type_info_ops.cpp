//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"

using namespace vpux;
using namespace IE;

void vpux::IE::arch37xx::registerElemTypeInfoOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::AffineReshapeOp::attachInterface<ElemTypeInfoAffineReshapeOpModel>(*ctx);
        IE::ClampOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ConcatOp::attachInterface<ElemTypeInfoConcatOpModel>(*ctx);
        IE::DepthToSpaceOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ExpandOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ExpandDilatedOp::attachInterface<ElemTypeInfoExpandDilatedOpModel>(*ctx);
        IE::InterpolateOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::MaxPoolOp::attachInterface<ElemTypeInfoMaxPoolOpModel>(*ctx);
        IE::ReduceMaxOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ReorderOp::attachInterface<ElemTypeInfoReorderOpModel>(*ctx);
        IE::ReshapeOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SliceOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SpaceToDepthOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SplitOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SqueezeOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::TileOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::TransposeOp::attachInterface<ElemTypeInfoTransposeOpModel>(*ctx);
        IE::UnsqueezeOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::UpsamplingOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
    });
}
