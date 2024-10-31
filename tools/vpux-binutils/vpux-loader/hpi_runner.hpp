//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/utils/error.hpp>
#include <vpux_hpi.hpp>

#include "heap_buffer_manager.hpp"

enum class AccessManagerType { DDRAccessManager = 0, FSAccessManager };

// Helper class intended to remove repetitive code needed to create supporting environment when creating
// HostParsedInference objects
template <typename HPIRunnerDerived>
class HPIRunner {
public:
    void run() {
        try {
            static_cast<HPIRunnerDerived*>(this)->runImpl();
            VPUX_ELF_THROW_WHEN(
                    mHpiBufferManager->getStats().mCurrentTotalSize || mIoBufferManager->getStats().mCurrentTotalSize,
                    elf::RuntimeError, "Memory leak occurred");
        } catch (std::exception& e) {
            llvm::outs() << llvm::formatv("Caught exception: {0}\n", e.what());
            // Rethrow to application level
            throw(e);
        } catch (...) {
            llvm::outs() << llvm::formatv("Caught exception unknown exception\n");
            VPUX_ELF_THROW(elf::RuntimeError, "Unkown exception occurred");
        }

        llvm::outs() << "\n\nRun completed successfully\n";
    }

protected:
    std::shared_ptr<HeapBufferManager> mHpiBufferManager = nullptr;
    std::shared_ptr<HeapBufferManager> mIoBufferManager = nullptr;
    std::shared_ptr<elf::AccessManager> mAccessManager = nullptr;
    // Blob vector storage for DDRAccessManager which doesn't work with smart pointers to ensure lifetime of blob
    // storage
    std::vector<uint8_t> mBlobBinVector = {};
    elf::HPIConfigs mHpiConfig = {};

private:
    friend HPIRunnerDerived;

    HPIRunner(const std::string& archName, const std::string& blobPathAndName,
              const AccessManagerType& accessManagerType) {
        mHpiBufferManager = std::make_shared<HeapBufferManager>("HPI buffer manager");
        mIoBufferManager = std::make_shared<HeapBufferManager>("IO buffer manager");

        switch (accessManagerType) {
        case AccessManagerType::DDRAccessManager: {
            // To simulate host vs NPU memory allocations distribution during HPI loading, build accessor with:
            // - NeverEmplace - all buffers are explicitly allocated
            // - AllocatedDeviceBufferFactory - ensure all buffers are allocated by a BufferManager
            mAccessManager = getDDRAccessManager<elf::DDRNeverEmplace>(
                    blobPathAndName.data(), mBlobBinVector,
                    std::make_shared<elf::AllocatedDeviceBufferFactory>(mHpiBufferManager.get()));
            break;
        }
        case AccessManagerType::FSAccessManager: {
            // To simulate host vs NPU memory allocations distribution during HPI loading, build accessor with:
            // - AllocatedDeviceBufferFactory - ensure all buffers are allocated by a BufferManager
            mAccessManager =
                    getFSAccessManager(blobPathAndName.data(),
                                       std::make_shared<elf::AllocatedDeviceBufferFactory>(mHpiBufferManager.get()));
            break;
        }
        default: {
            VPUX_ELF_THROW(elf::RuntimeError, "Unknown AccessManager type");
        }
        }

        mHpiConfig.archKind = elf::platform::mapArchStringToArchKind(archName);
    }

    template <typename EmplaceLogic, typename BufferFactory>
    static std::shared_ptr<elf::AccessManager> getDDRAccessManager(
            const std::string& filePathAndName, std::vector<uint8_t>& storageVector,
            std::shared_ptr<BufferFactory> bufferFactory = std::make_shared<BufferFactory>()) {
        storageVector.clear();
        std::ifstream inputStream(filePathAndName, std::ios::binary | std::ios::ate);
        storageVector.resize(inputStream.tellg());
        inputStream.seekg(0, inputStream.beg);
        inputStream.read(reinterpret_cast<char*>(storageVector.data()), storageVector.size());
        inputStream.close();

        return std::make_shared<elf::DDRAccessManager<EmplaceLogic, BufferFactory>>(
                storageVector.data(), storageVector.size(), bufferFactory);
    }

    template <typename BufferFactory>
    static std::shared_ptr<elf::AccessManager> getFSAccessManager(
            const std::string& filePathAndName,
            std::shared_ptr<BufferFactory> bufferFactory = std::make_shared<BufferFactory>()) {
        return std::make_shared<elf::FSAccessManager<BufferFactory>>(filePathAndName, bufferFactory);
    }
};
