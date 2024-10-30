//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RMSOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RMSOpAdaptor rms(operands, attrs, prop);
    if (mlir::failed(rms.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = rms.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto gammaType = rms.getGamma().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();
    const auto gammaShape = gammaType.getShape().raw();
    const auto inputRank = inputShape.size();

    if ((inputRank < 3) || (inputRank > 4)) {
        return errorAt(loc, "Input tensor rank should be 3 or 4. Got {0}D tensor.", inputRank);
    }

    if (inputShape[inputRank - 1] != gammaShape[0]) {
        return errorAt(loc, "Input width should be the same as gamma. Got input width = {0} and gamma width = {1}",
                       inputShape[inputRank - 1], gammaShape[0]);
    }

    inferredReturnTypes.push_back(inType);
    return mlir::success();
}
