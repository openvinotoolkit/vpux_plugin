//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"

#include <cstdint>
#include <utility>

extern std::unordered_map<std::string, const std::pair<const uint8_t*, size_t>> shaveBinaryResourcesMap;

namespace vpux {

class ShaveBinaryResources {
public:
    static const ShaveBinaryResources& getInstance();

private:
    ShaveBinaryResources() = default;

public:
    ShaveBinaryResources(ShaveBinaryResources const&) = delete;
    void operator=(ShaveBinaryResources const&) = delete;

    template <typename... Args>
    std::string concatenateArgs(Args&&... args) const {
        std::string result;
        ((result += ("_" + std::forward<Args>(args).str())), ...);
        return result;
    }

    template <typename... Args>
    llvm::ArrayRef<uint8_t> getData(llvm::StringRef entry, llvm::StringRef cpu, Args&&... args) const {
        auto result = printToString("{0}_{1}", entry, cpu);
        auto argsConcat = concatenateArgs(std::forward<Args>(args)...);
        auto symbolName = printToString("sk_{0}{1}_data", result, argsConcat);
        const auto it = shaveBinaryResourcesMap.find(symbolName);

        // For a shave kernel, the data section may be missing, in which case this symbol will not be found in the map.
        const uint8_t* symbolData = nullptr;
        size_t symbolSize = 0;
        if (it != shaveBinaryResourcesMap.end()) {
            std::tie(symbolData, symbolSize) = it->second;
        }

        return llvm::ArrayRef<uint8_t>(symbolData, symbolSize);
    }

    template <typename... Args>
    llvm::ArrayRef<uint8_t> getText(llvm::StringRef entry, llvm::StringRef cpu, Args&&... args) const {
        auto result = printToString("{0}_{1}", entry, cpu);
        auto argsConcat = concatenateArgs(std::forward<Args>(args)...);
        auto symbolName = printToString("sk_{0}{1}_text", result, argsConcat);
        const auto it = shaveBinaryResourcesMap.find(symbolName);

        VPUX_THROW_UNLESS(it != shaveBinaryResourcesMap.end(), "Can't find '.text' for kernel symbol '{0}'",
                          symbolName);

        const auto [symbolData, symbolSize] = it->second;
        return llvm::ArrayRef<uint8_t>(symbolData, symbolSize);
    }

    template <typename... Args>
    llvm::ArrayRef<uint8_t> getElf(llvm::StringRef entry, llvm::StringRef cpu, Args&&... args) const {
        auto result = printToString("{0}_{1}", entry, cpu);
        auto argsConcat = concatenateArgs(std::forward<Args>(args)...);
        auto symbolName = printToString("{0}{1}_elf", result, argsConcat);
        const auto it = shaveBinaryResourcesMap.find(symbolName);

        VPUX_THROW_UNLESS(it != shaveBinaryResourcesMap.end(), "Can't find 'elf' for kernel symbol '{0}'", symbolName);

        const auto [symbolData, symbolSize] = it->second;
        return llvm::ArrayRef<uint8_t>(symbolData, symbolSize);
    }

    llvm::ArrayRef<uint8_t> getElf(llvm::StringRef kernelPath) const;
};

}  // namespace vpux
