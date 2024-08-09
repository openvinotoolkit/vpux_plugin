//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov2_layer_test.hpp"
#include <gtest/internal/gtest-internal.h>
#include <npu_private_properties.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/make_tensor.hpp>
#include <sstream>
#include <vpux/utils/IE/config.hpp>

#include "vpu_test_report.hpp"

namespace ov::test::utils {

const VpuTestEnvConfig& VpuOv2LayerTest::envConfig = VpuTestEnvConfig::getInstance();

VpuOv2LayerTest::VpuOv2LayerTest(): testTool(envConfig) {
    OPENVINO_ASSERT(core != nullptr, "ov::Core instance is null");

    abs_threshold = 0.01;

    _log.setName("VPUTest");
    _log.setLevel(vpux::LogLevel::Info);
    this->targetDevice = ov::test::utils::DEVICE_NPU;

    if (!envConfig.IE_NPU_TESTS_LOG_LEVEL.empty()) {
        const auto logLevel = ::intel_npu::OptionParser<ov::log::Level>::parse(envConfig.IE_NPU_TESTS_LOG_LEVEL);
        _log.setLevel(vpux::getLogLevel(logLevel));
    }
}

void VpuOv2LayerTest::importModel() {
    OPENVINO_ASSERT(core != nullptr);
    compiledModel = testTool.importModel(core, filesysName(testing::UnitTest::GetInstance()->current_test_info(),
                                                           ".net", !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
}

void VpuOv2LayerTest::exportModel() {
    testTool.exportModel(compiledModel, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ".net",
                                                    !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
}

void VpuOv2LayerTest::exportInput() {
    for (const auto& input : compiledModel.inputs()) {
        const auto& input_name = input.get_node()->get_name();
        const auto ext = vpux::printToString(".{0}.{1}", input_name, "in");

        auto it = std::find_if(inputs.begin(), inputs.end(),
                               [&input](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& pair) {
                                   return pair.first->get_friendly_name() == input.get_node()->get_friendly_name();
                               });
        if (it != inputs.end()) {
            testTool.exportTensor(it->second, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                          !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
            continue;
        }
        OPENVINO_THROW("Compiled model input node was not found within generated inputs");
    }
}

void VpuOv2LayerTest::importInput() {
    for (const auto& input : compiledModel.inputs()) {
        const auto& input_name = input.get_node()->get_name();
        const auto ext = vpux::printToString(".{0}.{1}", input_name, "in");

        auto it = std::find_if(inputs.begin(), inputs.end(),
                               [&input](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& pair) {
                                   return pair.first->get_friendly_name() == input.get_node()->get_friendly_name();
                               });
        if (it != inputs.end()) {
            testTool.importTensor(it->second, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                          !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
            continue;
        }
        OPENVINO_THROW("Compiled model input node was not found within generated inputs");
    }
}

void VpuOv2LayerTest::exportOutput() {
    for (const auto& output : compiledModel.outputs()) {
        const auto& output_name = output.get_node()->get_name();
        const auto ext = vpux::printToString(".{0}.{1}", output_name, "out");
        testTool.exportTensor(inferRequest.get_tensor(output),
                              filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                          !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
    }
}

void VpuOv2LayerTest::exportReference(const std::vector<ov::Tensor>& refs) {
    size_t i = 0;
    for (const auto& output : compiledModel.outputs()) {
        const auto& name = output.get_node()->get_name();

        auto& ref = refs.at(i++);
        const auto ext = vpux::printToString(".{0}.{1}", name, "ref");
        testTool.exportTensor(ref, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                               !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
    }
}

std::vector<ov::Tensor> VpuOv2LayerTest::importReference() {
    std::vector<ov::Tensor> refs;
    for (const auto& output : compiledModel.outputs()) {
        const auto& name = output.get_node()->get_name();

        auto ref = ov::Tensor{output.get_element_type(), output.get_shape()};
        const auto ext = vpux::printToString(".{0}.{1}", name, "ref");
        testTool.importTensor(ref, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                               !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
        refs.push_back(ref);
    }
    return refs;
}

void VpuOv2LayerTest::run(const std::string_view platform) {
    setPlatform(platform);
    if (envConfig.IE_NPU_SINGLE_CLUSTER_MODE) {
        setSingleClusterMode();
    }
    run();
}

void VpuOv2LayerTest::run() {
    summary.setDeviceName(targetDevice);

    if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
        summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::SKIPPED);
        GTEST_SKIP_("Disabled test due to configuration");
    }

    summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::CRASHED);

    ASSERT_FALSE(targetStaticShapes.empty()) << "Target Static Shape is empty!";

    auto crashHandler = std::make_unique<ov::test::utils::CrashHandler>();

#ifdef _WIN32
    switch (setjmp(ov::test::utils::env)) {
#else
    switch (sigsetjmp(ov::test::utils::env, 1)) {
#endif
    case ov::test::utils::JMP_STATUS::ok:
        crashHandler->StartTimer();
        if (auto errorMessage = runTest()) {
            _log.error("Test has failed: {0}", errorMessage->c_str());
            GTEST_FATAL_FAILURE_(errorMessage->c_str());
            summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::FAILED);
        } else {
            summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::PASSED);
        }
        break;
    case ov::test::utils::JMP_STATUS::anyError:
        GTEST_FATAL_FAILURE_("Crash happened");
        break;
    case ov::test::utils::JMP_STATUS::alarmErr:
        summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::HANGED);
        GTEST_FATAL_FAILURE_("Application hanged");
        break;
    default:
        GTEST_FATAL_FAILURE_("Test failed: Unsupported failure type");
    }
}

VpuOv2LayerTest::ErrorMessage VpuOv2LayerTest::runTest() {
    try {
        auto& report = LayerTestsUtils::VpuTestReport::getInstance();
        const auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
        report.run(testInfo);

        printNetworkConfig();

        if (skipCompilationImpl()) {
            return std::nullopt;
        }

        if (envConfig.IE_NPU_TESTS_RUN_COMPILER) {
            compile_model();
            report.compiled(testInfo);

            if (envConfig.IE_NPU_TESTS_RUN_EXPORT) {
                _log.debug("`VpuOv2LayerTest::ExportModel()`");
                exportModel();
            }
        } else {
            _log.debug("`VpuOv2LayerTest::ImportModel()`");
            importModel();
            report.imported(testInfo);
        }

        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            try {
                generate_inputs(targetStaticShapeVec);
                if (envConfig.IE_NPU_TESTS_EXPORT_INPUT) {
                    _log.debug("`VpuOv2LayerTest::ExportInput()`");
                    exportInput();
                }
                if (envConfig.IE_NPU_TESTS_IMPORT_INPUT) {
                    _log.debug("`VpuOv2LayerTest::ImportInput()`");
                    importInput();
                }
            } catch (const std::exception& ex) {
                return ErrorMessage{"Impossible to reshape ov::Model using the shape: " +
                                    ov::test::utils::vec2str(targetStaticShapeVec) + " " + ex.what()};
            }

            try {
                if (auto errorMessage = skipInferenceImpl()) {
                    if (errorMessage == "SKIP") {
                        continue;
                    }
                    return errorMessage;
                }

                // [Track number: C#104172]
                // The infer() function is called inside validate() -> get_plugin_outputs() function
                _log.debug("`VpuOv2LayerTest::validate()`");
                validate();
                report.inferred(testInfo);
                report.validated(testInfo);

            } catch (const std::exception& ex) {
                return ErrorMessage{"Test failed on static shape: " + ov::test::utils::vec2str(targetStaticShapeVec) +
                                    "\n" + ex.what()};
            }
        }
    } catch (const std::exception& ex) {
        return ErrorMessage{ex.what()};
    } catch (...) {
        return ErrorMessage{"Unknown failure occurred."};
    }

    return std::nullopt;
}

void VpuOv2LayerTest::validate() {
    std::vector<ov::Tensor> expectedOutputs, actualOutputs;

    if (envConfig.IE_NPU_TESTS_RUN_INFER) {
        _log.info("Infer using '{0}' backend", getBackendName(*core));
        actualOutputs = get_plugin_outputs();
    }
    expectedOutputs = calculate_refs();

    if (envConfig.IE_NPU_TESTS_EXPORT_REF) {
        _log.debug("`VpuOv2LayerTest::ExportReference()`");
        exportReference(expectedOutputs);
    }

    if (envConfig.IE_NPU_TESTS_EXPORT_OUTPUT) {
        _log.debug("`VpuOv2LayerTest::ExportOutput()`");
        exportOutput();
    }

    if (envConfig.IE_NPU_TESTS_IMPORT_REF) {
        _log.debug("`VpuOv2LayerTest::ImportReference()`");
        expectedOutputs = importReference();
    }

    ASSERT_FALSE(expectedOutputs.empty()) << "Expected ouputs cannot be empty";

    ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
            << "TEMPLATE plugin has " << expectedOutputs.size() << " outputs, while " << targetDevice << " "
            << actualOutputs.size();
    if (is_report_stages) {
        _log.debug("[ COMPARISON ] `ov_tensor_utils.hpp::compare()` is started");
    }
    auto start_time = std::chrono::system_clock::now();

    compare(expectedOutputs, actualOutputs);
    if (is_report_stages) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        _log.debug("[ COMPARISON ] `ov_tensor_utils.hpp::compare()` is finished successfully. Duration is {}s",
                   duration.count());
    }
}

void VpuOv2LayerTest::setSkipCompilationCallback(SkipCallback skipCallback) {
    skipCompilationCallback = skipCallback;
}

void VpuOv2LayerTest::setSkipInferenceCallback(SkipCallback skipCallback) {
    skipInferenceCallback = skipCallback;
}

bool VpuOv2LayerTest::skipCompilationImpl() {
    if (skipCompilationCallback != nullptr) {
        std::stringstream skipStream;
        skipCompilationCallback(skipStream);

        const auto skipMessage = skipStream.str();
        if (!skipMessage.empty()) {
            _log.warning("Compilation skipped: {0}", skipMessage);
            return true;
        }
    }

    return false;
}

VpuOv2LayerTest::ErrorMessage VpuOv2LayerTest::skipInferenceImpl() {
    const auto backendName = getBackendName(*core);

    if (backendName.empty()) {
        return ErrorMessage{"Inference cannot run: backend is empty (no device)"};
    }

    if (skipInferenceCallback != nullptr) {
        std::stringstream skipStream;
        skipInferenceCallback(skipStream);

        const auto skipMessage = skipStream.str();
        if (!skipMessage.empty()) {
            _log.warning("Inference skipped: {0}", skipStream.str());
            return ErrorMessage{"SKIP"};
        }
    }

    return std::nullopt;
}

void VpuOv2LayerTest::printNetworkConfig() const {
    std::ostringstream ostr;
    for (const auto& item : configuration) {
        ostr << item.first << "=";
        item.second.print(ostr);
        ostr << "; ";
    }
    _log.info("NPU Plugin config: {0}", ostr.str());
}

void VpuOv2LayerTest::setPlatform(const std::string_view platform) {
    // [Track number: E#70404]
    // Multiple different ways of setting the platform
    configuration[ov::intel_npu::platform.name()] = std::string(platform);
    configuration[ov::device::id.name()] = std::string(platform);
}

void VpuOv2LayerTest::setReferenceSoftwareMode() {
    configuration[ov::intel_npu::compilation_mode.name()] = "ReferenceSW";
}

void VpuOv2LayerTest::setDefaultHardwareMode() {
    configuration[ov::intel_npu::compilation_mode.name()] = "DefaultHW";
}

bool VpuOv2LayerTest::isReferenceSoftwareMode() const {
    const auto compilationMode = configuration.at(ov::intel_npu::compilation_mode.name()).as<std::string>();
    return compilationMode == "ReferenceSW";
}

bool VpuOv2LayerTest::isDefaultHardwareMode() const {
    const auto compilationMode = configuration.at(ov::intel_npu::compilation_mode.name()).as<std::string>();
    return compilationMode == "DefaultHW";
}

void VpuOv2LayerTest::setSingleClusterMode() {
    configuration[ov::intel_npu::dpu_groups.name()] = "1";
    configuration[ov::intel_npu::dma_engines.name()] = "1";
}

void VpuOv2LayerTest::setPerformanceHintLatency() {
    configuration[ov::hint::performance_mode.name()] = "LATENCY";
}

void VpuOv2LayerTest::useELFCompilerBackend() {
    configuration[ov::intel_npu::use_elf_compiler_backend.name()] = "YES";
}

}  // namespace ov::test::utils
