//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/act_kernels/shave_binary_resources.h"

#include <string>
#include <unordered_map>
#include <utility>

using namespace vpux;

const ShaveBinaryResources& ShaveBinaryResources::getInstance() {
    static ShaveBinaryResources instance;
    return instance;
}

llvm::ArrayRef<uint8_t> ShaveBinaryResources::getElf(llvm::StringRef kernelPath) const {
    auto symbolName = printToString("{0}_elf", kernelPath);
    const auto it = shaveBinaryResourcesMap.find(symbolName);

    VPUX_THROW_UNLESS(it != shaveBinaryResourcesMap.end(), "Can't find 'elf' for kernel symbol '{0}'", symbolName);

    const auto [symbolData, symbolSize] = it->second;
    return llvm::ArrayRef<uint8_t>(symbolData, symbolSize);
}
