//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"

using namespace vpux;

namespace {
class AddPlatformInfo : public VPUMI40XX::AddPlatformInfoBase<AddPlatformInfo> {
public:
    explicit AddPlatformInfo(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddPlatformInfo::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto ctx = &(getContext());
    auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

    mlir::OpBuilder builder(&(funcOp.getBody().front().back()));
    builder.create<VPUMI40XX::PlatformInfoOp>(builder.getUnknownLoc(), trivialIndexType);
}
}  // namespace

//
// createAddPlatformInfoPass
//

std::unique_ptr<mlir::Pass> VPUMI40XX::createAddPlatformInfoPass(Logger log) {
    return std::make_unique<AddPlatformInfo>(log);
}
