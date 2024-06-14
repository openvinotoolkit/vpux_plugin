//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

using namespace vpux;

namespace {

//
//  CollectUsedMemoryPass
//

class CollectUsedMemoryPass final : public VPUIP::CollectUsedMemoryBase<CollectUsedMemoryPass> {
public:
    explicit CollectUsedMemoryPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void CollectUsedMemoryPass::safeRunOnModule() {
    auto module = getOperation();

    mlir::DenseMap<VPU::MemoryKind, int64_t> maxUsedMemory;
    for (const auto& funcOp : module.getOps<mlir::func::FuncOp>()) {
        auto usedMemVec = IE::getUsedMemory(funcOp);

        for (auto& memResourceOp : usedMemVec) {
            const auto memKind = VPU::getKindValue<VPU::MemoryKind>(memResourceOp);
            const auto currByteSize = memResourceOp.getByteSize();

            maxUsedMemory[memKind] =
                    maxUsedMemory.count(memKind) ? std::max(currByteSize, maxUsedMemory[memKind]) : currByteSize;
        }

        // TODO E#105253: consider not using temporary modules to store data in functions
        IE::eraseUsedMemory(funcOp);
    }

    for (const auto& pair : maxUsedMemory) {
        const auto memKind = pair.first;
        const auto size = pair.second;

        IE::setUsedMemory(module, memKind, Byte{size});
    }
}

}  // namespace

//
// createCollectUsedMemoryPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCollectUsedMemoryPass(Logger log) {
    return std::make_unique<CollectUsedMemoryPass>(log);
}
