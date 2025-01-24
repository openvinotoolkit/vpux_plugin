//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/act_kernels/shave_binary_resources.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>

using namespace vpux;

extern std::unordered_map<std::string, const std::pair<const uint8_t*, size_t>> shaveBinaryResourcesMap;

ShaveBinaryResources& ShaveBinaryResources::getInstance() {
    static ShaveBinaryResources instance;
    return instance;
}

vpux::SmallString ShaveBinaryResources::getSwKernelArchString(VPU::ArchKind archKind) {
    switch (archKind) {
    case VPU::ArchKind::NPU37XX:
        return vpux::SmallString("3720xx");
    case VPU::ArchKind::NPU40XX:
        return vpux::SmallString("4000xx");
    default:
        VPUX_THROW("unsupported archKind {0}", archKind);
        return vpux::SmallString("");
    }
}

llvm::ArrayRef<uint8_t> ShaveBinaryResources::getElf(llvm::StringRef kernelPath) const {
    auto symbolName = printToString("{0}_elf", kernelPath);
    const auto it = shaveBinaryResourcesMap.find(symbolName);

    VPUX_THROW_UNLESS(it != shaveBinaryResourcesMap.end(), "Can't find 'elf' for kernel symbol '{0}'", symbolName);

    const auto [symbolData, symbolSize] = it->second;
    return llvm::ArrayRef<uint8_t>(symbolData, symbolSize);
}

void ShaveBinaryResources::addCompiledElf(llvm::StringRef funcName, std::vector<uint8_t>& binary,
                                          llvm::StringRef arch) {
    auto symbolName = printToString("{0}_{1}_elf", funcName, arch);
    auto data = shaveBinaryResourcesMap.find(symbolName);

    if (data != shaveBinaryResourcesMap.end()) {
        shaveBinaryResourcesMap.erase(symbolName);
    }

    uint8_t* permArray = new uint8_t[binary.size()];
    memcpy(permArray, binary.data(), binary.size() * sizeof(uint8_t));

    shaveBinaryResourcesMap.insert(std::make_pair(symbolName, std::make_pair(permArray, binary.size())));
}

void ShaveBinaryResources::loadElfData(mlir::ModuleOp module) {
    ShaveBinaryResources& sbr = ShaveBinaryResources::getInstance();

    std::string line;

    std::ifstream ifileList("FileList.in", std::ifstream::in);
    if (!ifileList.is_open()) {
        return;
    }

    while (std::getline(ifileList, line)) {
        std::vector<uint8_t> binary;

        std::ifstream ifileElf(line, std::ifstream::in);
        VPUX_THROW_UNLESS(ifileElf.is_open(), "ELF file not found.");

        // Get length of file:
        ifileElf.seekg(0, std::ios::end);
        int length = ifileElf.tellg();
        ifileElf.seekg(0, std::ios::beg);

        auto buffer = std::vector<char>(length);
        ifileElf.read(buffer.data(), length);
        ifileElf.close();

        binary.insert(binary.end(), buffer.begin(), buffer.end());

        std::string funcName;
        std::getline(ifileList, funcName);

        VPU::ArchKind archKind = VPU::getArch(module.getOperation());
        auto kernelArch = getSwKernelArchString(archKind);

        sbr.addCompiledElf(funcName, binary, kernelArch);
    }

    ifileList.close();
}
