//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <list>
#include <string>
#include <unordered_map>

namespace vpux::VPUIP {

// Helper class to imitate L2 LRU cache behavior
class ShaveL2CacheSimulator {
public:
    ShaveL2CacheSimulator(size_t maxCapacity);

    size_t getCapacity() const;

    size_t getFreeSize() const;

    // Load new kernel entry into the cache
    void loadKernel(const std::string& kernelName, size_t kernelSize);

    // Check if the kernel is loaded into cache.
    // **DOESNT** change usage history and least recent task!
    bool isLoaded(const std::string& kernelName) const;

    // Invalidate all content of the cache
    void invalidate();

    // Return usage history, where least recent kernel is last
    const std::list<std::string>& getUsageHistory() const;

private:
    // Move kernel to the front of the usage history
    void handleHit(const std::string& kernelName);

    // Evict old kernels untill we achieve at least kernelSize free bytes
    void ensureFreeSize(size_t kernelSize);

    // Evict least recent kernel
    void evict();

private:
    size_t _maxCapacity;
    size_t _freeSize;
    std::unordered_map<std::string, size_t> _loadedKernels;
    std::list<std::string> _usageHistory;
};

};  // namespace vpux::VPUIP
