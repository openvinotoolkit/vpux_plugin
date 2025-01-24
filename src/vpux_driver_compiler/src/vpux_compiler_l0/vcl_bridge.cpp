//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

/**
 * @file vcl_bridge.cpp
 * @brief The bridge from L0 driver compiler to user API
 */

#include "vcl_common.hpp"
#include "vcl_compiler.hpp"
#include "vcl_executable.hpp"
#include "vcl_profiling.hpp"
#include "vcl_query_network.hpp"

#include "intel_npu/config/compiler.hpp"

using namespace vpux;

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

DLLEXPORT vcl_result_t vclCompilerCreate(vcl_compiler_desc_t desc, vcl_compiler_handle_t* compiler,
                                         vcl_log_handle_t* logHandle) {
    VPUXDriverCompiler::VCLLogger* vclLogger = nullptr;
    if (logHandle != nullptr) {
        /// Create logger which saves latest error messages, output other messages to terminal
        vclLogger = new VPUXDriverCompiler::VCLLogger("VCL", LogLevel::Error, true);
    } else {
        /// Create logger which output all message to terminal
        vclLogger = new VPUXDriverCompiler::VCLLogger("VCL", LogLevel::Error, false);
    }

    if (compiler == nullptr) {
        vclLogger->outputError("Null argument to create compiler!");
        delete vclLogger;
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    /// Set all default configs here
    std::map<std::string, std::string> config;
    config[ov::intel_npu::compiler_type.name()] = "MLIR";

    /// Set default log level based on the compiler description passed by user
    auto& log_level = config[ov::log::level.name()];
    switch (desc.debug_level) {
    case VCL_LOG_NONE:
        log_level = "LOG_NONE";
        break;
    case VCL_LOG_ERROR:
        log_level = "LOG_ERROR";
        break;
    case VCL_LOG_WARNING:
        log_level = "LOG_WARNING";
        break;
    case VCL_LOG_INFO:
        log_level = "LOG_INFO";
        break;
    case VCL_LOG_DEBUG:
        log_level = "LOG_DEBUG";
        break;
    case VCL_LOG_TRACE:
        log_level = "LOG_TRACE";
        break;
    default:
        log_level = "LOG_ERROR";
        desc.debug_level = VCL_LOG_ERROR;
    };

    /// Change the output level of logger
    int debugLevel = static_cast<int>(desc.debug_level);
    if (debugLevel > 0) {
        // OV does not have CONFIG_VALUE(LOG_FATAL), so does not use LogLevel::Fatal in VCL.
        vclLogger->setLevel(static_cast<LogLevel>(debugLevel + 1));
    }

    /// Create compiler
    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = nullptr;
    try {
        pCompiler = new VPUXDriverCompiler::VPUXCompilerL0(desc, config, vclLogger);
        *compiler = reinterpret_cast<vcl_compiler_handle_t>(pCompiler);
    } catch (const std::exception& error) {
        vclLogger->outputError(error.what());
        delete vclLogger;
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        vclLogger->outputError("Internal exception during compiler creation!");
        delete vclLogger;
        return VCL_RESULT_ERROR_UNKNOWN;
    }

    /// Create logger to save error msg, pass the handle here
    if (logHandle != nullptr) {
        *logHandle = reinterpret_cast<vcl_log_handle_t>(vclLogger);
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclCompilerGetProperties(vcl_compiler_handle_t compiler, vcl_compiler_properties_t* properties) {
    if (!properties || !compiler) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    *properties = pCompiler->getCompilerProp();
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclQueryNetworkCreate(vcl_compiler_handle_t compiler, vcl_query_desc_t desc,
                                                         vcl_query_handle_t* query) {
    /// Format of modelIRData is defined in L0 adaptor
    /// The modelIRData is parsed into model data and weights info
    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    VPUXDriverCompiler::VCLLogger* vclLogger = pCompiler->getLogger();

    if (!desc.modelIRData) {
        vclLogger->outputError("Invalid IR buffer!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (!desc.modelIRSize) {
        vclLogger->outputError("Invalid IR size!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    /// To avoid access violation, need to convert to string
    std::string descOptions(desc.options, desc.optionsSize);
    vclLogger->info("config: {0}", descOptions);

    VPUXDriverCompiler::BuildInfo buildInfo(pCompiler);

    /// Parse and update the config
    vcl_result_t ret = buildInfo.prepareConfig(descOptions);
    if (ret != VCL_RESULT_SUCCESS) {
        vclLogger->outputError("Failed to prepare config! Incorrect format!");
        return ret;
    }

    /// Parse the seralized model data and create model container for compiler
    ret = buildInfo.prepareModel(desc.modelIRData, desc.modelIRSize);
    if (ret != VCL_RESULT_SUCCESS) {
        vclLogger->outputError("Failed to prepare model! Incorrect format!");
        return ret;
    }

    /// Query which layers of the model are supported by current compiler
    VPUXDriverCompiler::VPUXQueryNetworkL0* pQueryNetwork = nullptr;
    pQueryNetwork = new VPUXDriverCompiler::VPUXQueryNetworkL0(vclLogger);
    ret = pCompiler->queryNetwork(buildInfo, pQueryNetwork);
    if (ret != VCL_RESULT_SUCCESS) {
        return ret;
    }

    *query = reinterpret_cast<vcl_query_handle_t>(pQueryNetwork);
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclQueryNetwork(vcl_query_handle_t query, uint8_t* queryResult, uint64_t* size) {
    if (query == nullptr || size == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXQueryNetworkL0* pvq = reinterpret_cast<VPUXDriverCompiler::VPUXQueryNetworkL0*>(query);
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    if (queryResult == nullptr) {
        /// First time calling vclQueryNetwork, get size of queryResultString
        ret = pvq->getQueryResultSize(size);
    } else {
        /// Second time calling vclQueryNetwork, get data of queryResultString
        ret = pvq->getQueryString(queryResult, *size);
    }
    return ret;
}

DLLEXPORT vcl_result_t vclQueryNetworkDestroy(vcl_query_handle_t query) {
    if (query != nullptr) {
        VPUXDriverCompiler::VPUXQueryNetworkL0* pvq = reinterpret_cast<VPUXDriverCompiler::VPUXQueryNetworkL0*>(query);
        delete pvq;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclExecutableCreate(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                           vcl_executable_handle_t* executable) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    bool enableProfiling = false;

    if (!compiler || !executable || !desc.modelIRData) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    VPUXDriverCompiler::VCLLogger* vclLogger = pCompiler->getLogger();

    /// To avoid access violation, need to convert to string
    std::string descOptions(desc.options, desc.optionsSize);
    vclLogger->info("config: {0}", descOptions);

    /// Create info parser
    VPUXDriverCompiler::BuildInfo buildInfo(pCompiler);
    /// Parse user dscriptions and store the input && output settings, compilation configs
    ret = buildInfo.prepareBuildFlags(descOptions);
    if (ret != VCL_RESULT_SUCCESS) {
        vclLogger->outputError(formatv("Failed to prepare ioinfo and config! DescOptions: {0}", descOptions));
        return ret;
    }

    /// Parse serialized model data and create the model container for compiler
    ret = buildInfo.prepareModel(desc.modelIRData, desc.modelIRSize);
    if (ret != VCL_RESULT_SUCCESS) {
        vclLogger->outputError("Failed to parse model info! Incorrect format!");
        return ret;
    }

    /// Use compiler to compile model and store the result blob
    std::pair<VPUXDriverCompiler::VPUXExecutableL0*, vcl_result_t> status;
    try {
        status = pCompiler->importNetwork(buildInfo);
    } catch (const std::exception& error) {
        vclLogger->outputError(error.what());
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        vclLogger->outputError("Internal exception! Can't compile model!");
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (status.second != VCL_RESULT_SUCCESS || ret != VCL_RESULT_SUCCESS) {
        /// Release memory if we failed to compile model
        if (status.first != nullptr)
            delete status.first;
        *executable = nullptr;
        vclLogger->outputError("Failed to create executable");
        return status.second;
    } else {
        /// Get blob from compiled result and store in executable
        VPUXDriverCompiler::VPUXExecutableL0* pExecutable = status.first;
        ret = pExecutable->serializeNetwork();
        if (ret != VCL_RESULT_SUCCESS) {
            delete pExecutable;
            *executable = nullptr;
            vclLogger->outputError("Failed to get compiled network");
            return ret;
        }
        /// Return the executable which holds the blob
        *executable = reinterpret_cast<vcl_executable_handle_t>(pExecutable);
    }
    return ret;
}

DLLEXPORT vcl_result_t vclAllocatedExecutableCreate(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                                    vcl_allocator_t const* allocator, uint8_t** blob, uint64_t* size) {
    uint8_t* blobResult = nullptr;
    uint64_t sizeResult = 0;
    auto scoped = VPUXDriverCompiler::Scoped{[&]() {
        *blob = blobResult;
        *size = sizeResult;
    }};

    if (!compiler || !allocator || !blob || !size || !desc.modelIRData) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    VPUXDriverCompiler::VCLLogger* vclLogger = pCompiler->getLogger();

    /// To avoid access violation, need to convert to string
    std::string descOptions(desc.options, desc.optionsSize);
    vclLogger->info("config: {0}", descOptions);

    /// Create info parser
    VPUXDriverCompiler::BuildInfo buildInfo(pCompiler);
    /// Parse user descriptions and store the input && output settings, compilation configs
    if (auto ret = buildInfo.prepareBuildFlags(descOptions); ret != VCL_RESULT_SUCCESS) {
        vclLogger->outputError(formatv("Failed to prepare io info and config! DescOptions: {0}", descOptions));
        return ret;
    }

    /// Parse serialized model data and create the model container for compiler
    if (auto ret = buildInfo.prepareModel(desc.modelIRData, desc.modelIRSize); ret != VCL_RESULT_SUCCESS) {
        vclLogger->outputError("Failed to parse model info! Incorrect format!");
        return ret;
    }

    try {
        // NetworkMetadata is part of the result, but unused in VCL
        // it'd just get destroyed at function call here
        auto result = pCompiler->importNetwork(buildInfo, allocator);
        blobResult = result.compiledNetwork.ptr;
        sizeResult = result.compiledNetwork.size;
    } catch (const std::exception& error) {
        vclLogger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        vclLogger->outputError("Internal exception! Can't compile model!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclExecutableGetSerializableBlob(vcl_executable_handle_t executable, uint8_t* blobBuffer,
                                                        uint64_t* blobSize) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    if (!blobSize || !executable) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VPUXExecutableL0* pExecutable =
            reinterpret_cast<VPUXDriverCompiler::VPUXExecutableL0*>(executable);
    VPUXDriverCompiler::VCLLogger* vclLogger = pExecutable->getLogger();

    if (!blobBuffer) {
        /// When we call this function the first time, shall pass empty pointer to blob buffer and return the size of
        /// blob. User will use the size to alloc memory to store the blob
        ret = pExecutable->getNetworkSize(blobSize);
    } else {
        /// When we call this function the second time, the value of blobSize shall be the result of first call.
        /// Store the real blob data to the passed buffer
        ret = pExecutable->exportNetwork(blobBuffer, *blobSize);
    }
    if (ret != VCL_RESULT_SUCCESS) {
        vclLogger->outputError("Failed to get blob");
        return ret;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclExecutableDestroy(vcl_executable_handle_t executable) {
    if (executable) {
        VPUXDriverCompiler::VPUXExecutableL0* pExecutable =
                reinterpret_cast<VPUXDriverCompiler::VPUXExecutableL0*>(executable);
        delete pExecutable;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclCompilerDestroy(vcl_compiler_handle_t compiler) {
    if (compiler) {
        VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
        /// Logger is released with compiler.
        /// If we decide to save error log, user can not use the handle of logger to read error after this.
        VPUXDriverCompiler::VCLLogger* vclLogger = pCompiler->getLogger();
        delete vclLogger;
        delete pCompiler;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclProfilingCreate(p_vcl_profiling_input_t profilingInput,
                                                      vcl_profiling_handle_t* profilingHandle,
                                                      vcl_log_handle_t* logHandle) {
    if (!profilingInput || !profilingHandle) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VCLLogger* vclLogger = nullptr;
    if (logHandle != nullptr) {
        /// Create logger which saves latest error messages, output other messages to terminal
        vclLogger = new VPUXDriverCompiler::VCLLogger("VCL", LogLevel::Error, true);
    } else {
        /// Create logger which output all message to terminal
        vclLogger = new VPUXDriverCompiler::VCLLogger("VCL", LogLevel::Error, false);
    }

    VPUXDriverCompiler::VPUXProfilingL0* profHandle =
            new (std::nothrow) VPUXDriverCompiler::VPUXProfilingL0(profilingInput, vclLogger);
    if (!profHandle) {
        vclLogger->outputError("Failed to create profiler");
        delete vclLogger;
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }

    *profilingHandle = reinterpret_cast<vcl_profiling_handle_t>(profHandle);

    /// Create logger to save error msg, pass the handle here
    if (logHandle != nullptr) {
        *logHandle = reinterpret_cast<vcl_log_handle_t>(vclLogger);
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclGetDecodedProfilingBuffer(vcl_profiling_handle_t profilingHandle,
                                                                vcl_profiling_request_type_t requestType,
                                                                p_vcl_profiling_output_t profilingOutput) {
    if (!profilingHandle || !profilingOutput) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXProfilingL0* prof = reinterpret_cast<VPUXDriverCompiler::VPUXProfilingL0*>(profilingHandle);
    VPUXDriverCompiler::VCLLogger* vclLogger = prof->getLogger();
    switch (requestType) {
    case VCL_PROFILING_LAYER_LEVEL:
        return prof->getLayerInfo(profilingOutput);
    case VCL_PROFILING_TASK_LEVEL:
        return prof->getTaskInfo(profilingOutput);
    case VCL_PROFILING_RAW:
        return prof->getRawInfo(profilingOutput);
    default:
        vclLogger->outputError("Request type is not supported.");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclProfilingDestroy(vcl_profiling_handle_t profilingHandle) {
    if (profilingHandle) {
        VPUXDriverCompiler::VPUXProfilingL0* pvp =
                reinterpret_cast<VPUXDriverCompiler::VPUXProfilingL0*>(profilingHandle);
        VPUXDriverCompiler::VCLLogger* vclLogger = pvp->getLogger();
        delete vclLogger;
        delete pvp;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclProfilingGetProperties(vcl_profiling_handle_t profilingHandle,
                                                             vcl_profiling_properties_t* properties) {
    if (!profilingHandle || !properties) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VPUXProfilingL0* pvp = reinterpret_cast<VPUXDriverCompiler::VPUXProfilingL0*>(profilingHandle);
    *properties = pvp->getProperties();
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclLogHandleGetString(vcl_log_handle_t logHandle, size_t* logSize, char* log) {
    if (!logHandle) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VCLLogger* vclLogger = reinterpret_cast<VPUXDriverCompiler::VCLLogger*>(logHandle);
    return vclLogger->getString(logSize, log);
}

#ifdef __cplusplus
}
#endif
