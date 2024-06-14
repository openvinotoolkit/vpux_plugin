//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/types/symbol_entry.hpp>
#include <vpux_elf/utils/error.hpp>
#include <vpux_hpi.hpp>

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_headers/buffer_specs.hpp>
#include <vpux_headers/device_buffer.hpp>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#ifndef VPUX_ELF_LOG_UNIT_NAME
#define VPUX_ELF_LOG_UNIT_NAME "VpuxLoaderMain"
#endif
#include <vpux_elf/utils/log.hpp>

using namespace elf;
using namespace std;
using namespace chrono;

namespace {

constexpr uint32_t DEFAULT_BASE_BUFF = 0x80000000;
constexpr uint32_t DEFAULT_BUF_SIZE = 128 * 1024 * 1024;

enum InputSource { Random, Zeroes, Ones, CheckerBoard, Files };

llvm::cl::opt<std::string> elfFilePath(llvm::cl::Positional, llvm::cl::Required, llvm::cl::value_desc("filename"),
                                       llvm::cl::init("vpu.elf"), llvm::cl::desc("<Input ELF file>"));

llvm::cl::opt<InputSource> inputSource(
        "i",
        llvm::cl::values(clEnumValN(Random, "random", "Random number generation"),
                         clEnumValN(Zeroes, "zeroes", "Fill with Zeores"), clEnumValN(Ones, "ones", "Fill ones"),
                         clEnumValN(CheckerBoard, "checkerBoard", "Fill with checkerBoardPattern"),
                         clEnumValN(Files, "files", "FileList specified by -f commandLine list")),
        llvm::cl::init(Zeroes), llvm::cl::desc("Input buffer source: "));

llvm::cl::list<std::string> inputFiles("f", llvm::cl::value_desc("Input Files"),
                                       llvm::cl::desc("Input file names. If buffer source is set to a generator "
                                                      "then this is ignored"));

llvm::cl::opt<std::string> outputFile("o", llvm::cl::value_desc("filename"), llvm::cl::init("vpu.bin"),
                                      llvm::cl::desc("Output file name"));

llvm::cl::opt<uint32_t> baseAddr("addr", llvm::cl::value_desc("baseAddr"), llvm::cl::init(DEFAULT_BASE_BUFF),
                                 llvm::cl::desc("DEVELOPER ONLY: VPU base adress of the HEXFILE to be linked to."
                                                "This value needs to  reflect the InferenceManagerDemoHex"
                                                "application configuration"));

llvm::cl::opt<size_t> memSize("size", llvm::cl::value_desc("memSize"), llvm::cl::init(DEFAULT_BUF_SIZE),
                              llvm::cl::desc("DEVELOPER ONLY: Maximum size of the allocated buffer on VPU side"
                                             "for the Loaded and linked hex-file. This value needs to reflect"
                                             "the InferenceManagerDemoHex application configuration"));

llvm::cl::opt<bool> useFScache("useFScache", llvm::cl::desc("Use FS AccessManager"));

llvm::cl::opt<bool> verbose("v", llvm::cl::desc("Set verbosity"));

}  // namespace

class FlatHexBufferManager : public BufferManager {
public:
    FlatHexBufferManager(uint32_t startAddr, size_t size)
            : m_startAddr(startAddr), m_totalSize(size), m_buffer(new uint8_t[m_totalSize]), m_tracker(m_buffer) {
    }
    FlatHexBufferManager(const FlatHexBufferManager&) = delete;
    FlatHexBufferManager& operator=(const FlatHexBufferManager&) = delete;

    DeviceBuffer allocate(const BufferSpecs& buffSpecs) override {
        VPUX_ELF_LOG(LogLevel::LOG_DEBUG, "Allocation request --> size %lu | alignment %lu | procFlags 0x%lx",
                     buffSpecs.size, buffSpecs.alignment, buffSpecs.procFlags);

        if (!m_buffer) {
            VPUX_ELF_LOG(LogLevel::LOG_ERROR, "Failed to allocate overall buffer of size %lu", buffSpecs.size);
            return DeviceBuffer(nullptr, 0, 0);
        }

        m_tracker = align_up<uint8_t>(m_tracker, buffSpecs.alignment);
        uint8_t* start = m_tracker;
        m_tracker += buffSpecs.size;

        if (m_tracker >= m_buffer + m_totalSize) {
            VPUX_ELF_LOG(
                    LogLevel::LOG_ERROR,
                    "Failed to allocate required buff of size %lu alignment %lu . Exceeding total device buffer space",
                    buffSpecs.size, buffSpecs.alignment);
            return DeviceBuffer(nullptr, 0, 0);
        }

        return DeviceBuffer(start, vpuWindow(start), buffSpecs.size);
    }

    void deallocate(DeviceBuffer& devBuffer) override {
        (void)devBuffer;
    }

    void lock(DeviceBuffer& devBuffer) override {
        void* cpu_addr = reinterpret_cast<void*>(devBuffer.cpu_addr());
        void* vpu_addr = reinterpret_cast<void*>(devBuffer.vpu_addr());
        size_t len = devBuffer.size();

        VPUX_ELF_LOG(LogLevel::LOG_DEBUG, "Locking buffer -> cpu_addr = %p | vpu_addr = %p | size = %lu", cpu_addr,
                     vpu_addr, len);
    }

    void unlock(DeviceBuffer& devBuffer) override {
        void* cpu_addr = reinterpret_cast<void*>(devBuffer.cpu_addr());
        void* vpu_addr = reinterpret_cast<void*>(devBuffer.vpu_addr());
        size_t len = devBuffer.size();

        VPUX_ELF_LOG(LogLevel::LOG_DEBUG, "Unlocking buffer -> cpu_addr = %p | vpu_addr = %p | size = %lu", cpu_addr,
                     vpu_addr, len);
    }

    ~FlatHexBufferManager() {
        delete[] m_buffer;
    }

    size_t copy(DeviceBuffer& to, const uint8_t* from, size_t count) override {
        memcpy(to.cpu_addr(), from, count);
        return count;
    }

    uint8_t* buffer() const {
        return m_buffer;
    }
    size_t size() const {
        return m_tracker - m_buffer;
    }

private:
    template <typename T>
    bool isPowerOfTwo(T val) {
        return val && ((val & (val - 1)) == 0);
    }

    template <typename T>
    T* align_up(const T* val, const size_t to) {
        VPUX_ELF_THROW_UNLESS(isPowerOfTwo(to), ArgsError, " VPU only supports power of 2 alignments");
        std::uintptr_t intPtr = reinterpret_cast<std::uintptr_t>(val);
        std::uintptr_t alignedAddr = (intPtr + to - 1) & ~(to - 1);
        return reinterpret_cast<T*>(alignedAddr);
    }

    uint32_t vpuWindow(uint8_t const* const addr) const {
        return m_startAddr + static_cast<uint32_t>(addr - m_buffer);
    }

    uint32_t const m_startAddr;
    size_t const m_totalSize;
    uint8_t* const m_buffer;
    uint8_t* m_tracker;
};

void allocIO(BufferManager* mngr, std::vector<DeviceBuffer>& ioVec, const std::vector<DeviceBuffer>& sizes,
             uint32_t* ioPtrs, uint32_t* ioSizesP, uint32_t* ioSize) {
    auto ioPtrBuf = mngr->allocate(BufferSpecs(1, sizes.size() * sizeof(uint32_t), SHF_NONE));
    auto ioSizesBuf = mngr->allocate(BufferSpecs(1, sizes.size() * sizeof(uint32_t), SHF_NONE));

    *ioPtrs = static_cast<uint32_t>(ioPtrBuf.vpu_addr());
    *ioSizesP = static_cast<uint32_t>(ioSizesBuf.vpu_addr());
    *ioSize = static_cast<uint32_t>(sizes.size());

    for (size_t i = 0; i < sizes.size(); ++i) {
        size_t size = sizes[i].size();
        auto buffer = mngr->allocate(BufferSpecs(1, size, SHF_NONE));

        ioVec.push_back(buffer);

        uint32_t* ioPtr = reinterpret_cast<uint32_t*>(ioPtrBuf.cpu_addr());
        uint32_t* ioSize = reinterpret_cast<uint32_t*>(ioSizesBuf.cpu_addr());

        ioPtr[i] = static_cast<uint32_t>(buffer.vpu_addr());
        ioSize[i] = static_cast<uint32_t>(buffer.size());
    }
}

void setInputs(std::vector<DeviceBuffer>& vec) {
    if (inputSource == InputSource::Files) {
        if (vec.size() != inputFiles.size()) {
            VPUX_ELF_LOG(LogLevel::LOG_ERROR, "Elf has %lu required inputs. Only %lu specified", vec.size(),
                         inputFiles.size());
            VPUX_ELF_THROW(ArgsError, "Input Parameter mismatch");
        }
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        auto& buf = vec[i];
        if (inputSource == InputSource::Random) {
            std::srand(static_cast<unsigned int>(std::time(0)));
            auto rand = []() -> uint8_t {
                return static_cast<uint8_t>(std::rand() % std::numeric_limits<uint8_t>::max());
            };
            std::generate_n(buf.cpu_addr(), buf.size(), rand);
        } else if (inputSource == InputSource::Zeroes) {
            std::fill_n(buf.cpu_addr(), buf.size(), 0);
        } else if (inputSource == InputSource::Ones) {
            std::fill_n(buf.cpu_addr(), buf.size(), 1);
        } else if (inputSource == InputSource::CheckerBoard) {
            uint8_t flip = 0;
            auto flipper = [&]() -> uint8_t {
                flip = (flip + 1) % 2;
                return flip;
            };
            std::generate_n(buf.cpu_addr(), buf.size(), flipper);
        } else if (inputSource == InputSource::Files) {
            auto iFile = inputFiles[i];
            std::ifstream iFileStream(iFile.data(), std::ios::binary);
            iFileStream.read(reinterpret_cast<char*>(buf.cpu_addr()), buf.size());
            iFileStream.close();
        }
    }
}

void dumpOutputs(const std::vector<DeviceBuffer>& vec) {
    llvm::outs() << llvm::formatv("Output count {0}:\n", vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << llvm::formatv("\t {0}: {1:x} -> size {2}\n", i, vec[i].vpu_addr(), vec[i].size());
    }
}

void dumpProfiling(const std::vector<DeviceBuffer>& vec) {
    llvm::outs() << llvm::formatv("Profiling count {0}:\n", vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << llvm::formatv("\t {0}: {1:x} -> size {2}\n", i, vec[i].vpu_addr(), vec[i].size());
    }
}

void dumpInputs(const std::vector<DeviceBuffer>& vec) {
    llvm::outs() << llvm::formatv("Input count {0}:\n", vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << llvm::formatv("\t {0}: {1:x} -> size {2}\n", i, vec[i].vpu_addr(), vec[i].size());
    }
}

struct HexMappedInferenceEntry {
    uint32_t elfEntryPtr;
    uint32_t totalSize;
    uint32_t inputsPtr;
    uint32_t inputSizesPtr;
    uint32_t inputsCount;
    uint32_t outputsPtr;
    uint32_t outputSizesPtr;
    uint32_t outputsCount;
    uint32_t profilingPtr;
    uint32_t profilingSizesPtr;
    uint32_t profilingCount;
};

int main(int argc, char* argv[]) {
    try {
        llvm::cl::ParseCommandLineOptions(argc, argv);

        if (verbose.getValue()) {
            Logger::setGlobalLevel(LogLevel::LOG_DEBUG);
        }

        std::unique_ptr<AccessManager> accessor;
        std::vector<uint8_t> elfFile;

        if (useFScache.getValue()) {
            accessor = std::make_unique<ElfFSAccessManager>(elfFilePath.data());
        } else {
            std::ifstream inputStream(elfFilePath.data(), std::ios::binary | std::ios::ate);
            elfFile.resize(inputStream.tellg());
            inputStream.seekg(0, inputStream.beg);
            inputStream.read(reinterpret_cast<char*>(elfFile.data()), elfFile.size());
            inputStream.close();

            ElfDDRAccessManager::Config accessorConfig;
            accessorConfig.mBufferManager = nullptr;
            accessorConfig.mInPlaceConfig.mIsInPlaceEnabled = true;
            accessorConfig.mInPlaceConfig.mIsAlignmentCheckEnabled = false;
            accessor = std::make_unique<ElfDDRAccessManager>(elfFile.data(), elfFile.size(), accessorConfig);
        }

        FlatHexBufferManager bufferManager(baseAddr, memSize);
        HexMappedInferenceEntry* hexEntry = reinterpret_cast<HexMappedInferenceEntry*>(
                bufferManager.allocate(BufferSpecs(1, sizeof(HexMappedInferenceEntry), SHF_NONE)).cpu_addr());

        auto start = high_resolution_clock::now();

        elf::HPIConfigs hpiConfigs;
        hpiConfigs.archKind = elf::platform::ArchKind::VPUX37XX;

        HostParsedInference loader(&bufferManager, accessor.get(), hpiConfigs);
        loader.load();

        auto end = high_resolution_clock::now();
        llvm::outs() << llvm::formatv("ELF loaded in {0}ms\n", duration_cast<milliseconds>(end - start).count());

        std::vector<DeviceBuffer> inputs;
        std::vector<DeviceBuffer> outputs;
        std::vector<DeviceBuffer> profiling;
        auto inputSizes = loader.getInputBuffers();
        auto outputSizes = loader.getOutputBuffers();
        auto profilingSizes = loader.getProfBuffers();

        allocIO(&bufferManager, inputs, inputSizes, &hexEntry->inputsPtr, &hexEntry->inputSizesPtr,
                &hexEntry->inputsCount);
        allocIO(&bufferManager, outputs, outputSizes, &hexEntry->outputsPtr, &hexEntry->outputSizesPtr,
                &hexEntry->outputsCount);
        allocIO(&bufferManager, profiling, profilingSizes, &hexEntry->profilingPtr, &hexEntry->profilingSizesPtr,
                &hexEntry->profilingCount);

        setInputs(inputs);

        dumpInputs(inputs);
        dumpOutputs(outputs);
        dumpProfiling(profiling);

        loader.applyInputOutput(inputs, outputs, profiling);

        hexEntry->totalSize = static_cast<uint32_t>(bufferManager.size());

        std::ofstream outFileStream;

        outFileStream.open(outputFile.data(), std::ios::out | std::ios::binary);
        outFileStream.write(reinterpret_cast<char*>(bufferManager.buffer()), bufferManager.size());
        outFileStream.close();

        VPUX_ELF_THROW_UNLESS(outFileStream.good(), AccessError, "Error at writing the output file");
    } catch (std::exception& e) {
        llvm::outs() << llvm::formatv("Caught exception: {0}\n", e.what());
    } catch (...) {
        llvm::outs() << llvm::formatv("Caught exception unknown exception\n");
    }

    return 0;
}
