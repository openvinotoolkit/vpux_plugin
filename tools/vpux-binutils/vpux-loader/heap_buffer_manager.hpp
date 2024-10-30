//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <string_view>

#include <llvm/Support/FormatVariadic.h>

#include <vpux_headers/buffer_manager.hpp>
#include <vpux_headers/buffer_specs.hpp>
#include <vpux_headers/device_buffer.hpp>

#include <vpux_elf/utils/error.hpp>
#include <vpux_elf/utils/utils.hpp>

// Class which implements the BufferManager interface through heap allocations
class HeapBufferManager : public elf::BufferManager {
public:
    struct AllocStats {
        size_t mCurrentTotalCount = 0;
        size_t mCurrentTotalSize = 0;
        size_t mTotalCPUCount = 0;
        size_t mTotalCPUSize = 0;
        size_t mTotalNPUCount = 0;
        size_t mTotalNPUSize = 0;
    };

    HeapBufferManager() = default;
    HeapBufferManager(std::string_view name): mName(name) {
    }

    HeapBufferManager(const HeapBufferManager& other) = delete;
    HeapBufferManager(HeapBufferManager&& other) = delete;

    HeapBufferManager operator=(const HeapBufferManager& rhs) = delete;
    HeapBufferManager operator=(HeapBufferManager&& rhs) = delete;

    ~HeapBufferManager() = default;

    elf::DeviceBuffer allocate(const elf::BufferSpecs& buffSpecs) override {
        auto ptr = std::aligned_alloc(buffSpecs.alignment, buffSpecs.size);
        VPUX_ELF_THROW_UNLESS(ptr, elf::RuntimeError, "Allocation failure");

        // All allocations have CPU VA
        auto cpuAddr = reinterpret_cast<uint8_t*>(ptr);
        // Only NPU allocations have NPU VA
        // Initializing to 0 could help early detection of faulty allocation logic from loader
        auto npuAddr = static_cast<uint64_t>(0);

        // Update statistics
        ++mAllocStats.mCurrentTotalCount;
        mAllocStats.mCurrentTotalSize += buffSpecs.size;
        if (elf::utils::hasNPUAccess(buffSpecs.procFlags)) {
            npuAddr = reinterpret_cast<uint64_t>(ptr);

            ++mAllocStats.mTotalNPUCount;
            mAllocStats.mTotalNPUSize += buffSpecs.size;
        } else {
            ++mAllocStats.mTotalCPUCount;
            mAllocStats.mTotalCPUSize += buffSpecs.size;
        }

        return elf::DeviceBuffer(cpuAddr, npuAddr, buffSpecs.size);
    }

    void deallocate(elf::DeviceBuffer& devBuffer) override {
        free(devBuffer.cpu_addr());
        VPUX_ELF_THROW_WHEN(mAllocStats.mCurrentTotalSize < devBuffer.size(), elf::RuntimeError,
                            "Freeing more memory than allocated");
        mAllocStats.mCurrentTotalSize -= devBuffer.size();
    }

    void lock(elf::DeviceBuffer&) override {
    }

    void unlock(elf::DeviceBuffer&) override {
    }

    size_t copy(elf::DeviceBuffer& to, const uint8_t* from, size_t count) override {
        std::memcpy(to.cpu_addr(), from, count);
        return count;
    }

    const AllocStats& getStats() {
        return mAllocStats;
    }

    void printAllocationStats() {
        llvm::outs() << llvm::formatv(
                "================================================================================\n");
        llvm::outs() << llvm::formatv("{0} allocation statistics:\n", mName);

        llvm::outs() << llvm::formatv(" - Current allocated size: {1} bytes in {2} buffers\n", mName,
                                      mAllocStats.mCurrentTotalSize, mAllocStats.mCurrentTotalCount);
        llvm::outs() << llvm::formatv(" - All-time CPU allocated size: {1} bytes in {2} buffers\n", mName,
                                      mAllocStats.mTotalCPUSize, mAllocStats.mTotalCPUCount);
        llvm::outs() << llvm::formatv(" - All-time NPU allocated size: {1} bytes in {2} buffers\n", mName,
                                      mAllocStats.mTotalNPUSize, mAllocStats.mTotalNPUCount);

        llvm::outs() << llvm::formatv(
                "================================================================================\n");
    }

private:
    std::string mName = {};
    AllocStats mAllocStats = {};
};
