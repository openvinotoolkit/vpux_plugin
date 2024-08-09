//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/compiler.hpp"

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/profiling.hpp"

#include "vpux/compiler/NPU37XX/pipeline_strategy.hpp"
#include "vpux/compiler/NPU37XX/pipelines.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/export.hpp"
#include "vpux/compiler/NPU40XX/pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipelines.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/export.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/network_description.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/network_description.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/compiler/options_mapper.hpp"
#include "vpux/compiler/utils/dot_printer.hpp"
#include "vpux/compiler/utils/locations_verifier.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/private_properties.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/memory_usage.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>

#include <llvm/Support/Regex.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/raw_ostream.h>

#include <openvino/core/dimension.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/iplugin.hpp>

#include <device_helpers.hpp>
#include <transformations/common_optimizations/dimension_tracking.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include <algorithm>
#include <regex>

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#include "vpux/compiler/core/developer_build_utils.hpp"
#endif

using namespace vpux;

using intel_npu::ICompiler;
using intel_npu::NetworkDescription;
using intel_npu::NetworkMetadata;

namespace {

constexpr std::string_view UNSUPPORTED_PLATFORM_ERROR_MESSAGE =
        "Unsupported platform: '{0}'\nThe current version of the compiler is unable to compile on the given platform. "
        "If you're using the compiler inside the plugin, please try using the '{1}' configuration option to set a "
        "supported platform explicitly.";

void checkPlaformSupportedForCompilation(const std::string_view platform) {
    const std::unordered_set supportedPlatforms{ov::intel_npu::Platform::NPU3720, ov::intel_npu::Platform::NPU4000};

    if (!supportedPlatforms.count(ov::intel_npu::Platform::standardize(platform))) {
        VPUX_THROW(UNSUPPORTED_PLATFORM_ERROR_MESSAGE.data(), platform, intel_npu::PLATFORM::key());
    }
}
constexpr uint32_t SUPPORTED_OPSET = 11;

//
// createPipelineStrategy
//

std::unique_ptr<IPipelineStrategy> createPipelineStrategy(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return std::make_unique<PipelineStrategy37XX>();
    case VPU::ArchKind::NPU40XX:
        return std::make_unique<PipelineStrategy40XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

//
// DeveloperConfig
//

class DeveloperConfig final {
public:
    explicit DeveloperConfig(Logger log);
    DeveloperConfig(const DeveloperConfig& other) = delete;
    DeveloperConfig& operator=(const DeveloperConfig& other) = delete;
    ~DeveloperConfig();

    void setup(mlir::DefaultTimingManager& tm) const;
    void setup(mlir::PassManager& pm) const;
    void dump(mlir::PassManager& pm) const;

    // Specifies whether to duplicate IE constants in MLIR when importing a network
    bool useSharedConstants() const {
        // Historically, some usages required IE constants to be verbosely printed. By MLIR's design,
        // the constants have to be *copied* in this case. As a result, the generated IR is more
        // human-readable as each constant is printed as an array of individual decimal values e.g.:
        // `/* const.Declare = */ dense<[1.0, 4.75391, 9.97656, 7.48438 /* , ... */]>`.
        return _crashReproducerFile.empty() && _irPrintingFilter.empty();
    }

private:
    Logger _log;

    std::string _crashReproducerFile;
    bool _localReproducer = true;

    std::string _irPrintingFilter;
    std::string _irPrintingFile;
    std::string _irPrintingOrderStr;
    bool _printFullIR = false;
    bool _printFullConstant = false;
    bool _allowPrintingHexConstant = true;
    bool _printDebugInfo = false;
    std::string _printAsTextualPipelineFilePath = "";
    std::string _printDotOptions;

    llvm::raw_ostream* _timingStream = nullptr;

    std::unique_ptr<llvm::Regex> _irDumpFilter;
    std::unique_ptr<llvm::raw_fd_ostream> _irDumpFile;
    llvm::raw_ostream* _irDumpStream = nullptr;
    IRPrintingOrder _irPrintingOrder = IRPrintingOrder::AFTER;
};

DeveloperConfig::DeveloperConfig(Logger log): _log(log) {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    parseEnv("IE_NPU_CRASH_REPRODUCER_FILE", _crashReproducerFile);
    parseEnv("IE_NPU_GEN_LOCAL_REPRODUCER", _localReproducer);

    parseEnv("IE_NPU_IR_PRINTING_FILTER", _irPrintingFilter);
    parseEnv("IE_NPU_IR_PRINTING_FILE", _irPrintingFile);
    parseEnv("IE_NPU_IR_PRINTING_ORDER", _irPrintingOrderStr);
    parseEnv("IE_NPU_PRINT_FULL_IR", _printFullIR);
    parseEnv("IE_NPU_PRINT_FULL_CONSTANT", _printFullConstant);
    parseEnv("IE_NPU_PRINT_HEX_CONSTANT", _allowPrintingHexConstant);
    parseEnv("IE_NPU_PRINT_DEBUG_INFO", _printDebugInfo);
    parseEnv("IE_NPU_PRINT_AS_TEXTUAL_PIPELINE_FILE", _printAsTextualPipelineFilePath);

    parseEnv("IE_NPU_PRINT_DOT", _printDotOptions);
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    if (_log.isActive(LogLevel::Info)) {
        _timingStream = &Logger::getBaseStream();
    }

    if (!_irPrintingOrderStr.empty()) {
        auto orderString = _irPrintingOrderStr;
        std::transform(orderString.begin(), orderString.end(), orderString.begin(), [](unsigned char c) {
            return std::toupper(c);
        });
        if (orderString == "BEFORE") {
            _irPrintingOrder = IRPrintingOrder::BEFORE;
        } else if (orderString == "AFTER") {
            _irPrintingOrder = IRPrintingOrder::AFTER;
        } else if (orderString == "BEFORE_AFTER") {
            _irPrintingOrder = IRPrintingOrder::BEFORE_AFTER;
        } else {
            VPUX_THROW("Invalid IR printing order: {0}.\nValid cases are: before, after and before_after. They are not "
                       "case-sensitive.\nExample: IE_NPU_IR_PRINTING_ORDER=Before",
                       _irPrintingOrderStr);
        }
    }

    if (!_irPrintingFilter.empty()) {
        _irDumpFilter = std::make_unique<llvm::Regex>(_irPrintingFilter, llvm::Regex::IgnoreCase);

        std::string regexErr;
        if (!_irDumpFilter->isValid(regexErr)) {
            VPUX_THROW("Invalid regular expression '{0}' : {1}", _irPrintingFilter, regexErr);
        }

        if (_irPrintingFile.empty()) {
            _irDumpStream = &Logger::getBaseStream();
        } else {
            std::error_code err;
            _irDumpFile = std::make_unique<llvm::raw_fd_ostream>(_irPrintingFile, err);
            if (err) {
                VPUX_THROW("Failed to open file '{0}' for write : {1}", _irPrintingFile, err.message());
            }

            _irDumpStream = _irDumpFile.get();
        }
    }
}

DeveloperConfig::~DeveloperConfig() {
    if (_timingStream != nullptr) {
        _timingStream->flush();
    }

    if (_irDumpStream != nullptr) {
        _irDumpStream->flush();
    }
}

void DeveloperConfig::setup(mlir::DefaultTimingManager& tm) const {
    if (_timingStream == nullptr) {
        tm.setEnabled(false);
    } else {
        tm.setEnabled(true);
        tm.setDisplayMode(mlir::DefaultTimingManager::DisplayMode::Tree);
        tm.setOutput(*_timingStream);
    }
}

void DeveloperConfig::setup(mlir::PassManager& pm) const {
    // Crash reproducer

    if (!_crashReproducerFile.empty()) {
        if (_localReproducer) {
            pm.getContext()->disableMultithreading();
        }

        pm.enableCrashReproducerGeneration(_crashReproducerFile, _localReproducer);
    }

    // IR printing

    if (_irDumpFilter != nullptr) {
        const bool printAfterOnlyOnChange = false;
        const bool printAfterOnlyOnFailure = false;

        const auto shouldPrintBeforePass = [&](mlir::Pass* pass, mlir::Operation*) {
            return (_irDumpFilter->match(pass->getName()) || _irDumpFilter->match(pass->getArgument())) &&
                   (_irPrintingOrder == IRPrintingOrder::BEFORE || _irPrintingOrder == IRPrintingOrder::BEFORE_AFTER);
        };
        const auto shouldPrintAfterPass = [&](mlir::Pass* pass, mlir::Operation*) {
            return (_irDumpFilter->match(pass->getName()) || _irDumpFilter->match(pass->getArgument())) &&
                   (_irPrintingOrder == IRPrintingOrder::AFTER || _irPrintingOrder == IRPrintingOrder::BEFORE_AFTER);
        };

        if (_printFullIR) {
            pm.getContext()->disableMultithreading();
        }

        mlir::OpPrintingFlags flags;
        if (!_printFullConstant) {
            flags.elideLargeElementsAttrs();
        }
        if (!_allowPrintingHexConstant) {
            flags.setAllowPrintingElementsAttrAsHex(false);
        }
        if (_printDebugInfo) {
            flags.enableDebugInfo(true);
        }

        pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, _printFullIR, printAfterOnlyOnChange,
                            printAfterOnlyOnFailure, *_irDumpStream, flags);
    }

    // Dot printing
    if (!_printDotOptions.empty()) {
        addDotPrinter(pm, _printDotOptions);
    }
    // Locations verifier
    addLocationsVerifier(pm);
}

void DeveloperConfig::dump(mlir::PassManager& pm) const {
    if (!_printAsTextualPipelineFilePath.empty()) {
        std::error_code err;
        auto passesDumpFile = std::make_unique<llvm::raw_fd_ostream>(_printAsTextualPipelineFilePath, err);
        if (err) {
            VPUX_THROW("Failed to open file '{0}' for write : {1}", _printAsTextualPipelineFilePath, err.message());
        }
        pm.printAsTextualPipeline(*passesDumpFile);
    }
}
}  // namespace

//
// CompilerImpl::query
//

ov::SupportedOpsMap vpux::CompilerImpl::query(const std::shared_ptr<const ov::Model>& model,
                                              const intel_npu::Config& config) const {
    Logger log("vpux-compiler", getLogLevel(config));
    log.setName("vpux::CompilerImpl::query");

    ov::SupportedOpsMap result;

    const std::string plugin_name = DEVICE_NAME;
    const auto arch = getArchKind(config);

    DeveloperConfig devConf(log);
    mlir::DefaultTimingManager tm;
    devConf.setup(tm);
    auto rootTiming = tm.getRootScope();

    log.trace("Get supported nodes.");
    auto supportedNodes = ov::get_supported_nodes(
            model,
            [&](const std::shared_ptr<ov::Model>& model) {
                log.trace("Run common nGraph passes.");
                IE::NGraphPasses::runNGraphPasses(model, rootTiming, arch);
            },
            [&](const std::shared_ptr<ov::Node>& op) {
                log.trace("Get supported operations list.");
                return IE::NGraphImporter::isOpSupported(op);
            });

    for (auto&& layerName : supportedNodes) {
        result.emplace(layerName, plugin_name);
    }

    return result;
}

//
// CompilerImpl::compile
//

namespace {

auto importNetwork(mlir::MLIRContext* ctx, const std::shared_ptr<ov::Model>& model, const DeveloperConfig& devConf,
                   mlir::TimingScope& rootTiming, bool enableProfiling, bool stubLayers, bool dynamicShapeToStatic,
                   vpux::VPU::ArchKind arch, Logger log) {
    auto importTiming = rootTiming.nest("Import network");
    return IE::importNetwork(ctx, model, devConf.useSharedConstants(), importTiming, enableProfiling, stubLayers,
                             dynamicShapeToStatic, arch, log.nest());
}

void compileNetwork(mlir::ModuleOp module, mlir::PassManager& pm, mlir::TimingScope& rootTiming) {
    auto compileTiming = rootTiming.nest("Compile network");
    pm.enableTiming(compileTiming);
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");
}

auto exportToELF(mlir::ModuleOp module, const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                 const std::vector<std::shared_ptr<const ov::Node>>& results) {
    const auto arch = VPU::getArch(module);
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return vpux::ELFNPU37XX::exportToELF(module, parameters, results);
    case VPU::ArchKind::NPU40XX:
        return vpux::ELF::exportToELF(module, parameters, results);
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

bool isIR10(const ov::Model& model) {
    const auto& rtInfo = model.get_rt_info();
    const auto it = rtInfo.find("version");
    if (it != rtInfo.end()) {
        const int64_t irVersion = it->second.as<int64_t>();
        return irVersion == 10;
    }
    return false;
}

std::vector<std::shared_ptr<const ov::Node>> buildOVParams(const std::shared_ptr<const ov::Model>& model) {
    std::vector<std::shared_ptr<const ov::Node>> constParams;
    VPUX_THROW_WHEN(model == nullptr, "Null OV model");

    // Here we decide whether we need to add operation_names as tensor names for
    // getInputs / getOutputs. Since these functions are designed to be used in new API only
    // always need to add operation names for IR v10
    const auto addOpNames = isIR10(*model);

    for (const auto& param : model->get_parameters()) {
        auto newParam = ov::as_type_ptr<ov::op::v0::Parameter>(param->copy_with_new_inputs({}));
        newParam->set_friendly_name(param->get_friendly_name());
        if (addOpNames) {
            newParam->output(0).get_tensor().add_names({newParam->get_friendly_name()});
        }
        newParam->validate_and_infer_types();
        constParams.emplace_back(newParam);
    }

    return constParams;
}

std::vector<std::shared_ptr<const ov::Node>> buildOVResults(const std::shared_ptr<const ov::Model>& model) {
    std::vector<std::shared_ptr<const ov::Node>> constResults;
    VPUX_THROW_WHEN(model == nullptr, "Null OV model");

    // Here we decide whether we need to add operation_names as tensor names for
    // getInputs / getOutputs. Since these functions are designed to be used in new API only
    // always need to add operation names for IR v10
    const auto addOpNames = isIR10(*model);

    for (const auto& result : model->get_results()) {
        auto fakeParam = std::make_shared<ov::op::v0::Parameter>(result->get_output_element_type(0),
                                                                 result->get_output_partial_shape(0));
        const std::string paramName = ov::op::util::create_ie_output_name(result->input_value(0));
        fakeParam->set_friendly_name(paramName);
        fakeParam->validate_and_infer_types();
        auto newResult = result->copy_with_new_inputs({fakeParam});
        newResult->set_friendly_name(result->get_friendly_name());
        if (addOpNames) {
            newResult->output(0).get_tensor().add_names({fakeParam->get_friendly_name()});
        }
        constResults.emplace_back(newResult);
    }

    return constResults;
}

NetworkDescription exportNetwork(mlir::ModuleOp module, mlir::TimingScope& rootTiming, Logger log,
                                 const std::shared_ptr<const ov::Model>& model,
                                 const intel_npu::Config& configuration) {
    const auto parameters = buildOVParams(model);
    const auto results = buildOVResults(model);

    if (isELFEnabled(configuration)) {
        auto blob = exportToELF(module, parameters, results);
        auto meta = VPUMI37XX::getNetworkMetadata(blob);

        return NetworkDescription(std::move(blob), std::move(meta));
    } else {
        auto exportTiming = rootTiming.nest("Export to blob");
        auto blob = VPUIP::exportToBlob(module, exportTiming, parameters, results, log);
        auto finalTiming = rootTiming.nest("Wrap into NetworkDescription");
        std::vector<uint8_t> compiledNetwork(blob.data(), blob.data() + blob.size());

        auto meta = VPUIP::getNetworkMetadata(compiledNetwork);
        return NetworkDescription(std::move(compiledNetwork), std::move(meta));
    }
}

template <typename Options>
bool getDummyOpReplacement(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    VPUX_THROW_UNLESS(options != nullptr, "failed to parse COMPILATION_MODE_PARAMS");
    return options->enableDummyOpReplacement;
}

template <typename ReferenceSWOptions, typename ReferenceHWOptions, typename DefaultHWOptions>
bool getDummyOpReplacement(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getDummyOpReplacement<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ReferenceHW) {
        return getDummyOpReplacement<ReferenceHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getDummyOpReplacement<DefaultHWOptions>(config);
    } else {
        VPUX_THROW("Unsupported compilation mode: {0}", compilationMode);
    }
}

bool getDummyOpReplacement(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getDummyOpReplacement<ReferenceSWOptions37XX, ReferenceHWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getDummyOpReplacement<ReferenceSWOptions40XX, ReferenceHWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        VPUX_THROW("Unsupported device type: {0}", arch);
    }
}

std::optional<size_t> getBatchSize(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config) {
    std::set<ov::Output<const ov::Node>> batchedInputs;
    std::set<ov::Output<const ov::Node>> batchedOutputs;
    std::set<size_t> sBatchSize;

    vpux::Logger logger("getBatchSize", getLogLevel(config));

    if (!config.has<intel_npu::BATCH_MODE>()) {
        return 1;
    }

    if (config.get<intel_npu::BATCH_MODE>() == ov::intel_npu::BatchMode::COMPILER) {
        return 1;
    }

    std::shared_ptr<ov::Model> testBatchModel = model->clone();
    // Find the batch dim
    ov::pass::Manager passManager;
    passManager.register_pass<ov::pass::InitNodeInfo>();
    passManager.register_pass<ov::pass::FindBatch>();
    passManager.run_passes(testBatchModel);
    // Do not reshape/re-batch originally batched networks and when there are no inputs with the N* layouts
    // input(s) should have the batch dim as the first dim (current limitation of the auto-batching impl)
    const auto& params = testBatchModel->get_parameters();
    for (size_t input_id = 0; input_id < params.size(); input_id++) {
        const auto& input = params[input_id];
        const auto& shape = input->get_partial_shape();
        if (shape.is_dynamic()) {
            logger.info("Dynamic networks are not supported when batching is handled by the plugin");
            return std::nullopt;
        }
        // Batching on plugin is working only when batching is found on 0th dimension
        if (shape.size() && shape[0].has_symbol()) {
            const auto& static_shape = input->get_shape();
            batchedInputs.insert(params[input_id]->output(0));
            sBatchSize.insert(static_shape[0]);
        } else {
            logger.info("Only networks with inputs batched by 0th dimension are supported");
            return std::nullopt;
        }
    }
    for (const auto& output : testBatchModel->get_results()) {
        const auto& shape = output->get_output_partial_shape(0);
        if (shape.is_dynamic()) {
            logger.info("Dynamic networks are not supported when batching is handled by the plugin");
            return std::nullopt;
        }
        // Batching on plugin is working only when batching is found on 0th dimension
        if (shape.size() && shape[0].has_symbol()) {
            const auto& node = output->input_value(0);
            const auto& static_shape = output->get_shape();
            batchedOutputs.insert(ov::Output<const ov::Node>(node.get_node(), node.get_index()));
            sBatchSize.insert(static_shape[0]);
        } else {
            logger.info("Only networks with outputs batched by 0th dimension are supported");
            return std::nullopt;
        }
    }
    if (!batchedInputs.size() || !batchedOutputs.size()) {
        logger.info("Only networks with inputs/outputs featuring batched dim are supported!");
        return std::nullopt;
    }

    if (sBatchSize.size() != 1) {
        logger.info("Batching size shall have same value for all tensors!");
        return std::nullopt;
    }

    auto it = sBatchSize.begin();
    return *it;
}

#ifdef BACKGROUND_FOLDING_ENABLED
struct ConstantFoldingConfig {
    bool foldingInBackgroundEnabled;
    int64_t maxConcurrentTasks;
    bool collectStatistics;
    int64_t memoryUsageLimit;
    double cacheCleanThreshold;
};

template <typename Options>
ConstantFoldingConfig getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    VPUX_THROW_UNLESS(options != nullptr, "failed to parse COMPILATION_MODE_PARAMS");
    return ConstantFoldingConfig{options->constantFoldingInBackground, options->constantFoldingInBackgroundNumThreads,
                                 options->constantFoldingInBackgroundCollectStatistics,
                                 options->constantFoldingInBackgroundMemoryUsageLimit,
                                 options->constantFoldingInBackgroundCacheCleanThreshold};
}

template <typename ReferenceSWOptions, typename ReferenceHWOptions, typename DefaultHWOptions>
ConstantFoldingConfig getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getConstantFoldingInBackground<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ReferenceHW) {
        return getConstantFoldingInBackground<ReferenceHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getConstantFoldingInBackground<DefaultHWOptions>(config);
    } else {
        VPUX_THROW("Unsupported compilation mode: {0}", compilationMode);
    }
}

ConstantFoldingConfig getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getConstantFoldingInBackground<ReferenceSWOptions37XX, ReferenceHWOptions37XX, DefaultHWOptions37XX>(
                config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getConstantFoldingInBackground<ReferenceSWOptions40XX, ReferenceHWOptions40XX, DefaultHWOptions40XX>(
                config);
    } else {
        VPUX_THROW("Unsupported device type: {0}", arch);
    }
}

#endif

void setSafeOptions(intel_npu::Config& config) {
    auto backendConfig = config.get<intel_npu::BACKEND_COMPILATION_PARAMS>();
    std::regex reg("enable-partial-workload-management=true");
    backendConfig = std::regex_replace(backendConfig, reg, "");
    // In case WLM is enabled by default explicitly disable it.
    backendConfig.append(" ").append("enable-partial-workload-management=false");
    config.update({{intel_npu::BACKEND_COMPILATION_PARAMS::key().data(), backendConfig}});
}

mlir::OwningOpRef<mlir::ModuleOp> compileModel(mlir::MLIRContext& ctx, const std::shared_ptr<ov::Model>& model,
                                               DeveloperConfig& devConf, mlir::TimingScope& rootTiming,
                                               bool enableDummyOpReplacement, const intel_npu::Config& config,
                                               vpux::Logger& log) {
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "compileModel");
    const auto arch = getArchKind(config);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "importNetwork");

    const auto dynamicShapeToStatic = config.get<intel_npu::DYNAMIC_SHAPE_TO_STATIC>();
    mlir::OwningOpRef<mlir::ModuleOp> module =
            importNetwork(&ctx, model, devConf, rootTiming, config.get<intel_npu::PERF_COUNT>(),
                          enableDummyOpReplacement, dynamicShapeToStatic, arch, log);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "PassManager");

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    addLogging(pm, log);
    devConf.setup(pm);

    auto pipelineFactory = createPipelineStrategy(arch);

    // TODO: somehow protect non-target cases
    pipelineFactory->buildPipeline(pm, config, rootTiming, log);

#ifdef BACKGROUND_FOLDING_ENABLED
    const auto foldingConfig = getConstantFoldingInBackground(config);

    std::unique_ptr<vpux::Const::BackgroundConstantFolding> foldingManager;
    if (foldingConfig.foldingInBackgroundEnabled) {
        foldingManager = std::make_unique<vpux::Const::BackgroundConstantFolding>(
                &ctx, foldingConfig.maxConcurrentTasks, foldingConfig.collectStatistics, foldingConfig.memoryUsageLimit,
                foldingConfig.cacheCleanThreshold, log);
    }
#endif

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "compileNetwork");

    compileNetwork(module.get(), pm, rootTiming);  // applies each pass in the pipeline

    if (isELFEnabled(config)) {
        mlir::PassManager elfPm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
        addLogging(elfPm, log);
        devConf.setup(elfPm);
        pipelineFactory->buildELFPipeline(elfPm, config, rootTiming, log);
        if (getWlmRollback(config).value_or(false)) {
            auto backup_module = mlir::OwningOpRef<mlir::ModuleOp>(module.get().clone());
            try {
                compileNetwork(module.get(), elfPm, rootTiming);
            } catch (WlmRollbackException&) {
                log.warning("Failed to export to ELF with current config, reverting to simple ELF pipeline");
                module = std::move(backup_module);
                auto safeConfig = config;
                setSafeOptions(safeConfig);
                mlir::PassManager simpleElfPm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
                addLogging(simpleElfPm, log);
                devConf.setup(simpleElfPm);
                pipelineFactory->buildELFPipeline(simpleElfPm, safeConfig, rootTiming, log);
                compileNetwork(module.get(), simpleElfPm, rootTiming);
            }
        } else {
            compileNetwork(module.get(), elfPm, rootTiming);
        }
    }

    devConf.dump(pm);

    return module;
}

}  // namespace

uint32_t CompilerImpl::getSupportedOpsetVersion() const {
    return SUPPORTED_OPSET;
}

NetworkDescription CompilerImpl::compile(const std::shared_ptr<ov::Model>& model,
                                         const intel_npu::Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    checkPlaformSupportedForCompilation(config.get<intel_npu::PLATFORM>());

    Logger log("vpux-compiler", getLogLevel(config));

    auto peakMemStart = getPeakMemoryUsage();

    DeveloperConfig devConf(log);

    mlir::DefaultTimingManager tm;
    devConf.setup(tm);

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "MLIRContext");

    const auto arch = getArchKind(config);

    mlir::DialectRegistry registry;
    registerDialects(registry);

    // TODO: needs refactoring. Ticket: E#50937
    // Dummy op interfaces will end up being deleted if we properly refactor this dummy op feature
    bool enableDummyOpReplacement = getDummyOpReplacement(config);
    registerCommonInterfaces(registry, enableDummyOpReplacement);

    auto interfacesRegistry = createInterfacesRegistry(arch);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry, mlir::MLIRContext::Threading::DISABLED);

    // If user didn't specify number of threads default to 8 threads. By default MLIR
    // will attempt to use all of the threads available on the system which might cause
    // large peak memory usage during constant-related passes such as constant-folding
    const bool hasThreadLimit = config.has<intel_npu::COMPILATION_NUM_THREADS>();
    int threadCount = 8;
    if (hasThreadLimit) {
        threadCount = config.get<intel_npu::COMPILATION_NUM_THREADS>();
    }
    llvm::ThreadPoolStrategy tpStr;
    std::unique_ptr<llvm::ThreadPool> threadPoolPtr;
    tpStr.ThreadsRequested = threadCount;
    tpStr.Limit = true;  // limits number of threads to the number of physical threads
    threadPoolPtr.reset(new llvm::ThreadPool(tpStr));
    ctx.setThreadPool(*threadPoolPtr);

    addLogging(ctx, log);
    auto rootTiming = tm.getRootScope();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    bool useCompilerBatching = true;
    try {
        auto batchSize = getBatchSize(model, config);

        if (batchSize.has_value()) {
            if (*batchSize > 1) {
                // When batching is handled by the plugin we need to modify performance_mode property to Throughput mode
                intel_npu::Config config_performance_mode = config;
                if (config_performance_mode.get<intel_npu::PERFORMANCE_HINT>() == ov::hint::PerformanceMode::LATENCY) {
                    std::stringstream strStream;
                    strStream << ov::hint::PerformanceMode::THROUGHPUT;
                    config_performance_mode.update({{ov::hint::performance_mode.name(), strStream.str()}});
                }

                // If fallback and handle batching on the compiler is needed we will use the original model
                std::shared_ptr<ov::Model> batch_model = model->clone();

                ov::set_batch(batch_model, 1);
                module = compileModel(ctx, batch_model, devConf, rootTiming, enableDummyOpReplacement,
                                      config_performance_mode, log);

                useCompilerBatching = false;
            }
        } else {
            const auto& batchType = config.get<intel_npu::BATCH_MODE>();
            if (batchType == ov::intel_npu::BatchMode::AUTO) {
                log.info("Batching is handled by the compiler");
            } else if (batchType == ov::intel_npu::BatchMode::PLUGIN) {
                VPUX_THROW("This model is not supported when handling batching on the plugin.");
            }
        }
    } catch (const std::exception& ex) {
        const auto& batchType = config.get<intel_npu::BATCH_MODE>();
        if (batchType == ov::intel_npu::BatchMode::AUTO) {
            log.info("An error occurred during network compilation so fallback on compiler batch mode {0}", ex.what());
        } else {
            VPUX_THROW(ex.what());
        }
    }

    if (useCompilerBatching) {
        module = compileModel(ctx, model, devConf, rootTiming, enableDummyOpReplacement, config, log);
    }

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "exportNetwork");
    auto networkDescription = exportNetwork(module.get(), rootTiming, log, model, config);
    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    auto peakMemEnd = getPeakMemoryUsage();

    log.debug("Start of compilation memory usage: Peak {0} KB", peakMemStart.count());
    // Note: Following log is parsed by CI. Take care when modifying it.
    log.info("End of compilation memory usage: Peak {0} KB", peakMemEnd.count());

    return networkDescription;
}

NetworkDescription CompilerImpl::compile(const std::shared_ptr<const ov::Model>& origModel,
                                         const intel_npu::Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "clone_model");

    // NGraph pipeline modifies the model so need to clone here
    auto model = origModel->clone();

    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    return compile(std::move(model), config);
}

//
// CompilerImpl::parse
//

NetworkMetadata CompilerImpl::parse(const std::vector<uint8_t>& compiledNetwork,
                                    const intel_npu::Config& config) const {
    if (isELFEnabled(config)) {
        return VPUMI37XX::getNetworkMetadata(compiledNetwork);
    } else {
        return VPUIP::getNetworkMetadata(compiledNetwork);
    }
}

//
// CompilerImpl::process_profiling_output
//

std::vector<ov::ProfilingInfo> CompilerImpl::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                      const std::vector<uint8_t>& network,
                                                                      const intel_npu::Config&) const {
    auto layerInfo = profiling::getLayerProfilingInfoHook(profData, network);
    return intel_npu::profiling::convertLayersToIeProfilingInfo(layerInfo);
}

//
// CreateNPUCompiler
//

#ifndef OPENVINO_STATIC_LIBRARY
OPENVINO_PLUGIN_API void CreateNPUCompiler(std::shared_ptr<ICompiler>& obj) {
    obj = std::make_shared<CompilerImpl>();
}
#endif

bool vpux::isELFEnabled(const intel_npu::Config& configuration) {
    const auto isVPUX37XX = getArchKind(configuration) == vpux::VPU::ArchKind::NPU37XX;
    const auto isVPUX40XX = getArchKind(configuration) == vpux::VPU::ArchKind::NPU40XX;

    const auto optionValue = configuration.get<intel_npu::USE_ELF_COMPILER_BACKEND>();
    using ov::intel_npu::ElfCompilerBackend;

    return optionValue == ElfCompilerBackend::YES ||
           (optionValue == ElfCompilerBackend::AUTO && (isVPUX37XX || isVPUX40XX));
}
