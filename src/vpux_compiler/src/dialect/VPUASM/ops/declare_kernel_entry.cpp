//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// DeclareKernelEntryOp
//

uint32_t vpux::VPUASM::DeclareKernelEntryOp::getKernelEntry() {
    const auto elfBlob = ELF::getKernelELF(getOperation(), getKernelPath());

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(elfBlob.data(), elfBlob.size());
    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>(&accessor);

    auto actKernelHeader = elf_reader.getHeader();
    return actKernelHeader->e_entry;
}
