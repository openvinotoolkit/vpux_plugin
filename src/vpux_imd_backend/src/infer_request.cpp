//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/infer_request.hpp"

#include <openvino/runtime/make_tensor.hpp>
#include <openvino/util/file_util.hpp>

#include "vpux/IMD/executor.hpp"
#include "vpux/IMD/parsed_properties.hpp"
#include "vpux/IMD/platform_helpers.hpp"

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/config/runtime.hpp"

#include <device_helpers.hpp>
#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>

using vpux::printToString;

namespace intel_npu {

IMDInferRequest::IMDInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel,
                                 const std::shared_ptr<IExecutor>& executor, const Config& config)
        : SyncInferRequest(compiledModel),
          _executorPtr(executor),
          _config(config),
          _logger("IMDInferRequest", vpux::getLogLevel(config)) {
    const auto& meta = compiledModel->get_network_metadata();
    _inputOrder = meta.inputOrder;
    _outputOrder = meta.outputOrder;

    for (const std::string& inputName : meta.inputNames) {
        const IONodeDescriptor& parameterDescriptor = meta.parameters.at(inputName);

        // No I/O buffers have been allocated so far by the plugin - allocate new ones here
        allocate_tensor(inputName, parameterDescriptor);
    }

    for (const std::string& outputName : meta.outputNames) {
        const IONodeDescriptor& resultDescriptor = meta.results.at(outputName);
        allocate_tensor(outputName, resultDescriptor);
    }

    for (const std::string& stateName : meta.stateNames) {
        const IONodeDescriptor& stateDescriptor = meta.states.at(stateName);
        allocate_tensor(stateName, stateDescriptor, TensorType::State);
    }

    for (const std::string& shapeName : meta.shapeNames) {
        const IONodeDescriptor& shapeDescriptor = meta.shapes.at(shapeName);
        allocate_tensor(shapeName, shapeDescriptor, TensorType::Shape);
    }
}

void IMDInferRequest::infer() {
    infer_async();
    get_result();
}

void IMDInferRequest::infer_async() {
    _logger.debug("InferRequest::infer_async started");
    _logger.info("Run inference using InferenceManagerDemo application");

    _workDirectory = create_temporary_work_directory();

    store_compiled_model();
    store_network_inputs();
    run_app();
    _logger.debug("InferRequest::infer_async finished");
}

void IMDInferRequest::get_result() {
    _logger.debug("InferRequest::get_result started");

    load_network_outputs();

    _logger.trace("Remove the temporary working directory '{0}'", _workDirectory);
    const auto errc = llvm::sys::fs::remove_directories(_workDirectory);

    if (errc) {
        _logger.error("Failed to remove temporary working directory : {0}", errc.message());
    }

    _logger.debug("InferRequest::get_result finished");
}

vpux::SmallString IMDInferRequest::create_temporary_work_directory() {
    _logger.trace("Create unique temporary working directory");

    vpux::SmallString _workDirectory;
    const auto errc = llvm::sys::fs::createUniqueDirectory("vpux-IMD", _workDirectory);
    VPUX_THROW_WHEN(errc, "Failed to create temporary working directory : {0}", errc.message());

    _logger.nest().trace("{0}", _workDirectory);

    return _workDirectory;
}

void IMDInferRequest::store_compiled_model() {
    _logger.trace("Store the compile model");

    IMDExecutor* executor = static_cast<IMDExecutor*>(_executorPtr.get());
    const auto& compiledModel = executor->getNetworkDesc()->compiledNetwork;

    const std::string fileName = "vpuip.blob";
    const auto modelFilePath = printToString("{0}/{1}", _workDirectory.str(), fileName);
    std::ofstream file(modelFilePath, std::ios::binary);

    VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for write", modelFilePath);

    file.write(reinterpret_cast<const char*>(compiledModel.data()), compiledModel.size());

    _logger.nest().trace("{0}", modelFilePath);
}

void IMDInferRequest::store_network_inputs() {
    _logger.trace("Store the network inputs");

    size_t inputIndex;

    for (const auto& name : _inputAndStateInputNames) {
        std::shared_ptr<ov::ITensor>& inputTensor = _allTensors.at(name);

        if (vpux::isShapeTensorName(name)) {
            const auto actualTensorName = name.substr(SHAPE_TENSOR_PREFIX.size());
            const auto& inputDims = _allTensors.at(actualTensorName)->get_shape();

            for (size_t i = 0; i < inputTensor->get_size(); ++i) {
                const auto reverseIdx = inputDims.size() - 1 - i;
                inputTensor->data<uint32_t>()[i] = vpux::checked_cast<uint32_t>(inputDims[reverseIdx]);
            }
            inputIndex = _inputOrder.at(name);
        } else if (vpux::isStateOutputName(name)) {
            inputIndex = _inputOrder.at(vpux::stateOutputToStateInputName(name));
        } else {
            inputIndex = _inputOrder.at(name);
        }

        const auto inputFilePath = printToString("{0}/input-{1}.bin", _workDirectory.str(), inputIndex);
        std::ofstream file(inputFilePath, std::ios_base::binary | std::ios_base::out);

        VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for write", inputFilePath);

        file.write(reinterpret_cast<const char*>(inputTensor->data()), inputTensor->get_byte_size());

        _logger.nest().trace("{0} - {1}", name, inputFilePath);
    }
}

void IMDInferRequest::run_app() {
    _logger.trace("Run the application");

    vpux::SmallString curPath;
    auto errc = llvm::sys::fs::current_path(curPath);
    VPUX_THROW_WHEN(errc, "Failed to get current path : {0}", errc.message());

    VPUX_SCOPE_EXIT {
        _logger.nest().trace("Restore current working directory '{0}'", curPath);
        errc = llvm::sys::fs::set_current_path(curPath);

        if (errc) {
            _logger.error("Failed to restore current path : {0}", errc.message());
        }
    };

    _logger.nest().trace("Change current working directory to the new temporary folder '{0}'", _workDirectory.str());
    errc = llvm::sys::fs::set_current_path(_workDirectory.str());
    VPUX_THROW_WHEN(errc, "Failed to change current path : {0}", errc.message());

    const std::string emptyString;
    llvm::SmallVector<std::optional<llvm::StringRef>> redirects = {
            std::nullopt,  // stdin(0)
            std::nullopt,  // stdout(1)
            std::nullopt   // stderr(2)
    };

    if (_logger.level() < vpux::LogLevel::Error) {
        // diconnect stderr file descriptor
        redirects[2] = llvm::StringRef(emptyString);
    }

    if (_logger.level() < vpux::LogLevel::Info) {
        // diconnect stdout file descriptor
        redirects[1] = llvm::StringRef(emptyString);
    }

    std::string errMsg;
    auto app = static_cast<IMDExecutor*>(_executorPtr.get())->getApp();
    llvm::SmallVector<llvm::StringRef> args(app.runArgs.begin(), app.runArgs.end());
    _logger.trace("exec: {0}", app.runProgram);
    _logger.trace("args: {0}", args);

    const auto procErr = llvm::sys::ExecuteAndWait(app.runProgram, args,
                                                   /*Env=*/std::nullopt, llvm::ArrayRef(redirects),
                                                   vpux::checked_cast<uint32_t>(app.timeoutSec),
                                                   /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_WHEN(procErr != 0, "Failed to run InferenceManagerDemo ({0}) : {1}", procErr, errMsg);
}

void IMDInferRequest::read_from_file(const std::string& path, const std::shared_ptr<ov::ITensor>& tensor,
                                     const bool isDynamic) {
    VPUX_THROW_UNLESS(tensor->data() != nullptr, "Tensor was not allocated");

    std::ifstream file(path, std::ios_base::binary | std::ios_base::ate);
    VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for reading", path);

    const std::size_t tensorByteSize = tensor->get_byte_size();
    const auto fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios_base::beg);
    VPUX_THROW_UNLESS(fileSize == tensorByteSize || isDynamic, "File '{0}' contains {1} bytes, but {2} expected", path,
                      fileSize, tensorByteSize);

    file.read(reinterpret_cast<char*>(tensor->data()), static_cast<std::streamsize>(tensorByteSize));
}

void IMDInferRequest::load_network_outputs() {
    _logger.trace("Load the network outputs");

    const auto contains = [](const auto& container, const auto& value) {
        return std::find(container.begin(), container.end(), value) != container.end();
    };

    for (const auto& name : _outputAndStateOutputNames) {
        const std::shared_ptr<ov::ITensor>& outputTensor = _allTensors.at(name);

        const auto outputFilePath = printToString("{0}/output-{1}.bin", _workDirectory.str(), _outputOrder.at(name));

        auto legacyNameMatch = _nodeNameToLegacyName.find(name);
        const auto isDynamic = legacyNameMatch != _nodeNameToLegacyName.end()
                                       ? contains(_metadata.shapeNames, legacyNameMatch->second)
                                       : false;
        read_from_file(outputFilePath, outputTensor, isDynamic);

        if (vpux::isShapeTensorName(name)) {
            ov::Shape actualDims;
            for (size_t i = 0; i < outputTensor->get_size(); ++i) {
                const auto reverseIdx = outputTensor->get_size() - 1 - i;
                actualDims.push_back(outputTensor->data<uint32_t>()[reverseIdx]);
            }

            const auto actualTensorName = name.substr(SHAPE_TENSOR_PREFIX.size());
            const auto& shapeNameMatch = _legacyNameToNodeName.find(actualTensorName);
            if (shapeNameMatch != _legacyNameToNodeName.end()) {
                std::shared_ptr<ov::ITensor>& tensorToBeReshaped = _allTensors.at(shapeNameMatch->second);
                tensorToBeReshaped->set_shape(actualDims);
            }
        }

        _logger.nest().trace("{0} - {1}", name, outputFilePath);
    }

    const std::shared_ptr<const NetworkDescription>& networkDescription =
            static_cast<IMDExecutor*>(_executorPtr.get())->getNetworkDesc();
    const IONodeDescriptorMap& profilingOutputDescriptors = networkDescription->metadata.profilingOutputs;

    if (profilingOutputDescriptors.size()) {
        _logger.info("Load profiling output");
        VPUX_THROW_UNLESS(profilingOutputDescriptors.size() == 1, "Expected single profiling output");

        const IONodeDescriptor& profilingOutputDescriptor = profilingOutputDescriptors.begin()->second;
        const std::shared_ptr<ov::ITensor>& profilingOutputTensor = ov::make_tensor(
                profilingOutputDescriptor.precision, profilingOutputDescriptor.transposedShape.get_max_shape());
        read_from_file(printToString("{0}/profiling-0.bin", _workDirectory.str()), profilingOutputTensor);
        _rawProfilingData = profilingOutputTensor;
    }
}

void IMDInferRequest::check_network_precision(const ov::element::Type_t precision) {
    switch (precision) {
    case ov::element::Type_t::f32:
        break;
    case ov::element::Type_t::f16:
        break;
    case ov::element::Type_t::u4:
        break;
    case ov::element::Type_t::i4:
        break;
    case ov::element::Type_t::u8:
        break;
    case ov::element::Type_t::i8:
        break;
    case ov::element::Type_t::u16:
        break;
    case ov::element::Type_t::i16:
        break;
    case ov::element::Type_t::u32:
        break;
    case ov::element::Type_t::i32:
        break;
    case ov::element::Type_t::u64:
        break;
    case ov::element::Type_t::i64:
        break;
    case ov::element::Type_t::f64:
        break;
    case ov::element::Type_t::boolean:
        break;
    default:
        OPENVINO_THROW(
                "Unsupported tensor precision: " + ov::element::Type(precision).get_type_name() +
                "! Supported precisions: FP32, FP16, U4, I4, U8, I8, U16, I16, U32, I32, U64, I64, FP64, BOOLEAN");
    }
}

std::vector<ov::ProfilingInfo> IMDInferRequest::get_profiling_info() const {
    const auto& compiledModel = *std::dynamic_pointer_cast<const ICompiledModel>(_compiledModel);
    const auto& compilerConfig = compiledModel.get_config();
    if (!compilerConfig.get<PERF_COUNT>()) {
        return {};
    }

    VPUX_THROW_WHEN(compilerConfig.get<COMPILER_TYPE>() != ov::intel_npu::CompilerType::MLIR,
                    "IMD backend does not support profiling with Level Zero compiler");

    const auto& networkDesc = compiledModel.get_network_description();
    const auto& compiler = compiledModel.get_compiler();
    const auto& blob = networkDesc->compiledNetwork;
    auto profData = get_raw_profiling_data();
    return compiler->process_profiling_output(profData, blob, compilerConfig);
}

std::vector<uint8_t> IMDInferRequest::get_raw_profiling_data() const {
    VPUX_THROW_WHEN(_rawProfilingData == nullptr, "No profiling data");
    auto begin = reinterpret_cast<uint8_t*>(_rawProfilingData->data());
    auto end = begin + _rawProfilingData->get_byte_size();
    return {begin, end};
}

}  // namespace intel_npu
