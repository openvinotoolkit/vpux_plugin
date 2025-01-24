//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

#include <npu_40xx_nnrt.hpp>

namespace {

class UpdateMappedInferenceVersionOpPass :
        public VPUMI40XX::UpdateMappedInferenceVersionOpBase<UpdateMappedInferenceVersionOpPass> {
public:
    UpdateMappedInferenceVersionOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UpdateMappedInferenceVersionOpPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto ctx = &(getContext());
    auto mpi = VPUMI40XX::getMPI(netFunc);
    // In case barrier configuration to be used for barrier FIFO programming by VPU-FW
    // is present, what is the case when compiler wants to use barrier FIFOs, new API version
    // needs to be reported
    if (mpi.getBarrierConfigurationTasksCount().has_value()) {
        auto mappedInferenceVersion = to_small_vector(netFunc.getOps<VPUMI40XX::MappedInferenceVersionOp>());
        VPUX_THROW_WHEN(mappedInferenceVersion.size() != 1, "IR needs to have exactly one MPI version OP. Got {0}",
                        mappedInferenceVersion.size());
        auto mpiVersion = mappedInferenceVersion[0];
        mpiVersion.setMajorAttr(mlir::IntegerAttr::get(vpux::getUInt32Type(ctx), VPU_NNRT_40XX_API_VER_MAJOR));
        mpiVersion.setMinorAttr(mlir::IntegerAttr::get(vpux::getUInt32Type(ctx), VPU_NNRT_40XX_API_VER_MINOR));
        mpiVersion.setPatchAttr(mlir::IntegerAttr::get(vpux::getUInt32Type(ctx), VPU_NNRT_40XX_API_VER_PATCH));
    }
}

}  // namespace

//
// createUpdateMappedInferenceVersionOp
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createUpdateMappedInferenceVersionOpPass(Logger log) {
    return std::make_unique<UpdateMappedInferenceVersionOpPass>(log);
}
