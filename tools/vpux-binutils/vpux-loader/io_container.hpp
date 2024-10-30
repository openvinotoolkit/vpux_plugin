//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vector>

#include <vpux_headers/buffer_manager.hpp>
#include <vpux_headers/buffer_specs.hpp>
#include <vpux_headers/device_buffer.hpp>
#include <vpux_headers/device_buffer_container.hpp>

// Helper class for allocating and storing IO buffers for HPI object(s)
class IOBuffersContainer {
public:
    IOBuffersContainer(std::shared_ptr<elf::BufferManager> bufferManager,
                       const std::vector<elf::DeviceBuffer>& inputDescriptions,
                       const std::vector<elf::DeviceBuffer>& outputDescriptions,
                       const std::vector<elf::DeviceBuffer>& profilingDescriptions)
            : mBufferManager(bufferManager),
              mInputBuffersContainer(bufferManager.get()),
              mOutputBuffersContainer(bufferManager.get()),
              mProfilingBuffersContainer(bufferManager.get()) {
        allocIO(mInputBuffersContainer, inputDescriptions);
        allocIO(mOutputBuffersContainer, outputDescriptions);
        allocIO(mProfilingBuffersContainer, profilingDescriptions);

        mInputBuffers = mInputBuffersContainer.getBuffersAsVector();
        mOutputBuffers = mOutputBuffersContainer.getBuffersAsVector();
        mProfilingBuffers = mProfilingBuffersContainer.getBuffersAsVector();
    }

    std::vector<elf::DeviceBuffer>& getInputBuffers() {
        return mInputBuffers;
    }

    std::vector<elf::DeviceBuffer>& getOutputBuffers() {
        return mOutputBuffers;
    }

    std::vector<elf::DeviceBuffer>& getProfilingBuffers() {
        return mProfilingBuffers;
    }

    static void allocIO(elf::DeviceBufferContainer& bufferContainer,
                        const std::vector<elf::DeviceBuffer>& bufferSpecs) {
        for (size_t index = 0; index < bufferSpecs.size(); ++index) {
            auto& bufferInfo = bufferContainer.safeInitBufferInfoAtIndex(index);
            bufferInfo.mBuffer =
                    bufferContainer.buildAllocatedDeviceBuffer(elf::BufferSpecs(1024, bufferSpecs[index].size(), 0));
        }
    }

private:
    // DeviceBufferContainer operates with raw pointer of BufferManager, so keep the BufferManager referenced to be able
    // to deallocate all buffers when destroying the DeviceBufferContainer objects
    std::shared_ptr<elf::BufferManager> mBufferManager;
    elf::DeviceBufferContainer mInputBuffersContainer;
    elf::DeviceBufferContainer mOutputBuffersContainer;
    elf::DeviceBufferContainer mProfilingBuffersContainer;

    std::vector<elf::DeviceBuffer> mInputBuffers = {};
    std::vector<elf::DeviceBuffer> mOutputBuffers = {};
    std::vector<elf::DeviceBuffer> mProfilingBuffers = {};
};
