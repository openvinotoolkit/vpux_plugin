//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ManagedMappedInferenceOp
//

void vpux::NPUReg40XX::ManagedMappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    npu40xx::nn_public::VpuManagedMappedInference mmi = {};

    mmi.vpu_nnrt_api_ver = VPU_NNRT_40XX_API_VER;
    mmi.final_barrier = getFinalBarrier();
    mmi.work_items.count = getWorkItemsCount();
    mmi.task_configs.count = getTaskConfigsCount();
    mmi.initial_barriers.count = getBootstrapTaskCount();
    mmi.bootstrap_workitems_count = getBootsrapWorkItemsCount();
    mmi.actshv_used = getActshvUsed();
    mmi.dpu_used = getDpuUsed();
    mmi.media_used = getMediaUsed();
    mmi.dma_from_cmx_used = getDmaFromCmxUsed();
    mmi.dma_from_ddr_used = getDmaFromDdrUsed();

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&mmi);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::NPUReg40XX::ManagedMappedInferenceOp::getBinarySize() {
    return sizeof(npu40xx::nn_public::VpuManagedMappedInference);
}
