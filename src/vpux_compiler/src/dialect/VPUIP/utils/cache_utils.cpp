//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/cache_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPUIP {

ShaveL2CacheSimulator::ShaveL2CacheSimulator(size_t maxCapacity): _maxCapacity(maxCapacity), _freeSize(maxCapacity) {
}

size_t ShaveL2CacheSimulator::getCapacity() const {
    return _maxCapacity;
}

size_t ShaveL2CacheSimulator::getFreeSize() const {
    return _freeSize;
}

void ShaveL2CacheSimulator::loadKernel(const std::string& kernelName, size_t kernelSize) {
    VPUX_THROW_WHEN(kernelSize > _maxCapacity, "Kernel {0} is too large to be fully loaded into L2 cache", kernelSize);

    if (_loadedKernels.count(kernelName) != 0) {
        handleHit(kernelName);
        return;
    }
    ensureFreeSize(kernelSize);

    _freeSize -= kernelSize;
    _loadedKernels[kernelName] = kernelSize;
    _usageHistory.push_front(kernelName);
}

bool ShaveL2CacheSimulator::isLoaded(const std::string& kernelName) const {
    return _loadedKernels.count(kernelName) != 0;
}

void ShaveL2CacheSimulator::invalidate() {
    _freeSize = _maxCapacity;
    _loadedKernels.clear();
    _usageHistory.clear();
}

const std::list<std::string>& ShaveL2CacheSimulator::getUsageHistory() const {
    return _usageHistory;
}

void ShaveL2CacheSimulator::handleHit(const std::string& kernelName) {
    const auto historyIt = std::find(_usageHistory.begin(), _usageHistory.end(), kernelName);
    _usageHistory.erase(historyIt);
    _usageHistory.push_front(kernelName);
}

void ShaveL2CacheSimulator::ensureFreeSize(size_t kernelSize) {
    while (kernelSize > _freeSize) {
        evict();
    }
}

void ShaveL2CacheSimulator::evict() {
    const std::string leastRecentUsedKernel = _usageHistory.back();
    const size_t kernelSize = _loadedKernels[leastRecentUsedKernel];

    _freeSize += kernelSize;
    _usageHistory.pop_back();
    _loadedKernels.erase(leastRecentUsedKernel);
}

};  // namespace vpux::VPUIP
