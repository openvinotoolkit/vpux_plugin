//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/VPUMI40XX/utils.hpp>
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;

namespace vpux {

namespace {

template <VPURegMapped::TaskType type>
struct MetadataBufferSize {};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::DPUInvariant> {
    static constexpr auto size = nn_public::VPU_INVARIANT_COUNT;
};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::DPUVariant> {
    static constexpr auto size = nn_public::VPU_VARIANT_COUNT;
};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::ActKernelRange> {
    static constexpr auto size = nn_public::VPU_KERNEL_RANGE_COUNT;
};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::ActKernelInvocation> {
    static constexpr auto size = nn_public::VPU_KERNEL_INVO_COUNT;
};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::DMA> {
    // metadata buffers sizes for DMA tasks are hard-coded to the same value as in symbol table (Loader)
    // otherwise it must be written to the blob and extracted in Loader, passed to symbol table
    // however we do not want to expose blob content details (mapped inference) to the loader
    // E#81910
    // static constexpr auto size = 32UL;
};

template <>
struct MetadataBufferSize<VPURegMapped::TaskType::M2I> {
    static constexpr auto size = nn_public::VPU_MEDIA_COUNT;
};

class ResolveTaskLocationPass final : public VPUMI40XX::ResolveTaskLocationBase<ResolveTaskLocationPass> {
public:
    ResolveTaskLocationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    template <VPURegMapped::TaskType type>
    void populate(size_t listsCount = 1) {
        for (auto& sizesPerTaskType : _metadataBuffersSizes) {
            auto& sizesPerList = sizesPerTaskType[type];
            sizesPerList.resize(listsCount);
            for (auto& size : sizesPerList) {
                size = MetadataBufferSize<type>::size;
            }
        }
    }

    void safeRunOnFunc() final;
};

void ResolveTaskLocationPass::safeRunOnFunc() {
    _metadataBuffersSizes.resize(nn_public::VPU_MAX_TILES);
    populate<VPURegMapped::TaskType::DPUInvariant>();
    populate<VPURegMapped::TaskType::DPUVariant>();
    populate<VPURegMapped::TaskType::ActKernelRange>();
    populate<VPURegMapped::TaskType::ActKernelInvocation>();

    auto funcOp = getOperation();
    auto dmaOps = funcOp.getOps<VPUMI40XX::NNDMAOp>();
    size_t ddrDmaCount = 64, cmxDmaCount = 16;
    if (!dmaOps.empty()) {
        ddrDmaCount = std::count_if(dmaOps.begin(), dmaOps.end(), [](VPUMI40XX::NNDMAOp dmaOp) {
            return dmaOp.getIndexType().getListIdx() == static_cast<uint32_t>(VPUMI40XX::DmaNnSrcType::DDR);
        });

        cmxDmaCount = std::count_if(dmaOps.begin(), dmaOps.end(), [](VPUMI40XX::NNDMAOp dmaOp) {
            return dmaOp.getIndexType().getListIdx() == static_cast<uint32_t>(VPUMI40XX::DmaNnSrcType::CMX_NN);
        });

        std::tie(ddrDmaCount, cmxDmaCount) = VPUMI40XX::compute_dma_split(ddrDmaCount, cmxDmaCount);
    }

    for (auto& sizesPerTaskType : _metadataBuffersSizes) {
        auto& sizesPerList = sizesPerTaskType[VPURegMapped::TaskType::DMA];
        sizesPerList.resize(2);
        sizesPerList[0] = ddrDmaCount;
        sizesPerList[1] = cmxDmaCount;
    }

    populate<VPURegMapped::TaskType::M2I>();

    createTaskLocationBuffers();
}

}  // namespace

std::unique_ptr<mlir::Pass> VPUMI40XX::createResolveTaskLocationPass(Logger log) {
    return std::make_unique<ResolveTaskLocationPass>(log);
}

}  // namespace vpux
