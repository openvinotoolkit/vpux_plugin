//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

void vpux::ELF::PadOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto padSize = getPaddingSize();

    auto padValue = getPaddingValue().value_or(0);

    SmallVector<uint8_t> padding(padSize, padValue);

    binDataSection.appendData(padding.data(), padSize);
}

size_t vpux::ELF::PadOp::getBinarySize() {
    return getPaddingSize();
}
