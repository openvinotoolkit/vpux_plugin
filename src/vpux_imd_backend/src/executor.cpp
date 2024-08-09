//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/executor.hpp"

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "vpux/IMD/parsed_properties.hpp"
#include "vpux/IMD/platform_helpers.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <openvino/util/file_util.hpp>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>

#include <fstream>

using vpux::printToString;

namespace Platform = ov::intel_npu::Platform;

namespace intel_npu {

//
// setWorkloadType
//
void IMDExecutor::setWorkloadType(const ov::WorkloadType /*workloadType*/) const {
    VPUX_THROW("IMDExecutor does not support WorkloadType");
}

//
// getMoviToolsPath
//

std::string IMDExecutor::getMoviToolsPath(const Config& config) {
    if (config.has<MV_TOOLS_PATH>()) {
        return printToString("{0}/linux64/bin", config.get<MV_TOOLS_PATH>());
    } else {
        const auto* rootDir = std::getenv("MV_TOOLS_DIR");
        const auto* version = std::getenv("MV_TOOLS_VERSION");

        if (rootDir != nullptr && version != nullptr) {
            return printToString("{0}/{1}/linux64/bin", rootDir, version);
        } else {
            VPUX_THROW("Can't locate MOVI tools directory, please provide VPUX_IMD_MV_TOOLS_PATH config option or "
                       "MV_TOOLS_DIR/MV_TOOLS_VERSION env vars");
        }
    }
}

//
// isValidElfSignature
//

bool IMDExecutor::isValidElfSignature(llvm::StringRef filePath) {
    std::ifstream in(std::string(filePath), std::ios_base::binary);

    VPUX_THROW_UNLESS(in.is_open(), "Could not open {0}", filePath);

    char buffer[4];
    in.read(buffer, 4);

    if (!in || buffer[0] != 0x7f || buffer[1] != 0x45 || buffer[2] != 0x4c || buffer[3] != 0x46) {
        return false;
    }

    return true;
}

//
// getSimicsPath
//

std::string IMDExecutor::getSimicsPath(const Config& config) {
    if (config.has<VPU4_SIMICS_DIR>()) {
        return config.get<VPU4_SIMICS_DIR>();
    } else {
        VPUX_THROW("Can't locate simics directory, please provide VPUX_IMD_VPU4_SIMICS_DIR config option or "
                   "VPU4_SIMICS_DIR env var");
    }
}

//
// setElfFile
//

void IMDExecutor::setElfFile(const std::string& bin) {
    if (char* custom = std::getenv("NPU_IMD_BIN")) {
        _app.elfFile = custom;
    } else {
        _app.elfFile = bin;
    }
}

//
// setMoviSimRunArgs
//

void IMDExecutor::setMoviSimRunArgs(const std::string_view platform, const Config& config) {
    const auto appName = getAppName(platform);
    const auto pathToTools = getMoviToolsPath(config);

    _app.runProgram = printToString("{0}/moviSim", pathToTools);

    if (platform == Platform::NPU3720) {
        // For some reason, -cv:3720xx doesn't work, while -cv:3700xx works OK for NPU3720
        _app.chipsetArg = "-cv:3700xx";
        setElfFile(printToString("{0}/vpux/{1}", ov::util::get_ov_lib_path(), appName));
        _app.imdElfArg = printToString("-l:LRT:{0}", _app.elfFile);
    } else {
        _app.chipsetArg = "-cv:ma2490";
        setElfFile(printToString("{0}/vpux/simulator/{1}", ov::util::get_ov_lib_path(), appName));
        _app.imdElfArg = printToString("-l:LRT0:{0}", _app.elfFile);
    }

    _app.runArgs = {_app.runProgram, _app.chipsetArg, "-nodasm", "-q", _app.imdElfArg, "-simLevel:fast"};
}

//
// setMoviDebugRunArgs
//

void IMDExecutor::setMoviDebugRunArgs(const std::string_view platform, const Config& config) {
    const auto appName = getAppName(platform);
    const auto pathToTools = getMoviToolsPath(config);

    const auto* vpuElfPlatform = std::getenv("NPU_ELF_PLATFORM");
    const auto* vpuFirmwareDir = std::getenv("NPU_FIRMWARE_SOURCES_PATH");
    const auto* srvIP = std::getenv("NPU_SRV_IP");
    const auto* srvPort = std::getenv("NPU_SRV_PORT");

    if (vpuFirmwareDir == nullptr) {
        VPUX_THROW("Can't locate vpu firmware directory, please provide NPU_FIRMWARE_SOURCES_PATH env var");
    }
    if (vpuElfPlatform == nullptr) {
        vpuElfPlatform = "silicon";
        _log.warning("'NPU_ELF_PLATFORM' env variable is unset, using the default value: 'silicon'");
    } else {
        auto vpuElfPlatformStr = std::string(vpuElfPlatform);
        if (vpuElfPlatformStr != "silicon" && vpuElfPlatformStr != "fpga")
            VPUX_THROW("Unsupported value for moviDebug run on NPU_ELF_PLATFORM env var, expected: 'silicon' or "
                       "'fpga', got '{0}'",
                       vpuElfPlatformStr);
    }

    _app.runProgram = printToString("{0}/moviDebug2", pathToTools);
    setElfFile(printToString("{0}/vpux/{1}", ov::util::get_ov_lib_path(), appName));
    _app.imdElfArg = printToString("-D:elf={0}", _app.elfFile);

    std::string default_targetArg;

    if (platform == Platform::NPU4000) {
        _app.chipsetArg = "-cv:4000";
        default_targetArg = "-D:default_target=H0";
    } else if (platform == Platform::NPU3720) {
        _app.chipsetArg = "-cv:3700xx";
        default_targetArg = "-D:default_target=LRT";
    } else {
        VPUX_THROW("Platform '{0}' is not supported", platform);
    }

    _app.runArgs = {_app.runProgram, _app.imdElfArg, _app.chipsetArg, default_targetArg, "--no-uart"};

    if (srvIP != nullptr) {
        auto srvIPArg = printToString("-srvIP:{0}", srvIP);
        _app.runArgs.append({srvIPArg});
    } else {
        _log.warning("'NPU_SRV_IP' env variable is unset, moviDebug2 will try to connect to localhost");
    }

    if (srvPort != nullptr) {
        auto srvPortArg = printToString("-serverPort:{0}", srvPort);
        _app.runArgs.append({srvPortArg});
    } else {
        _log.warning("'NPU_SRV_PORT' env variable is unset, moviDebug2 will try to connect to 30000 or 30001 port");
    }

    // Debug scripts
    if (platform == Platform::NPU4000) {
        auto test_run_templateArg = printToString("{0}/../make/test-run-template.scr", vpuFirmwareDir);
        auto test_run_reset_script = printToString("-D:lnl_reset={0}/../make/lnl_reset.tcl", vpuFirmwareDir);
        _app.runArgs.append({"--script", test_run_templateArg});
        _app.runArgs.append({test_run_reset_script});
    } else if (platform == Platform::NPU3720) {
        auto default_mdbg2Arg = printToString("{0}/build/buildSupport/scripts/debug/default_mdbg2.scr", vpuFirmwareDir);
        auto default_pipe_mdbg2Arg =
                printToString("{0}/build/buildSupport/scripts/debug/default_pipe_mdbg2.scr", vpuFirmwareDir);
        auto default_run_mdbg2Arg =
                printToString("{0}/build/buildSupport/scripts/debug/default_run_mdbg2.scr", vpuFirmwareDir);

        _app.runArgs.append({"--init", default_mdbg2Arg});
        _app.runArgs.append({"--init", default_pipe_mdbg2Arg});
        _app.runArgs.append({"--script", default_run_mdbg2Arg});
    } else {
        VPUX_THROW("Platform '{0}' is not supported", platform);
    }

    _app.runArgs.append({"-D:run_opt=runw", "-D:exit_opt=exit"});
}

//
// setSimicsRunArgs
//

void IMDExecutor::setSimicsRunArgs(const std::string_view platform, const Config& config) {
    const auto appName = getAppName(platform);
    const auto simicsDir = getSimicsPath(config);

    _app.runProgram = printToString("{0}/simics", simicsDir);
    setElfFile(printToString("{0}/vpux/{1}", ov::util::get_ov_lib_path(), appName));

    auto binaryFile = "$binary=" + _app.elfFile;

    // common params
    _app.runArgs = {
            _app.runProgram,
            "-batch-mode",
            "-e",
            binaryFile,
    };

    if (platform == Platform::NPU4000) {
        _app.runArgs.insert(_app.runArgs.end(), {"-e", "$VPU_GEN=4", "-e", "$VPU_GENSKU=4000", "-e",
                                                 "run-command-file \"%simics%/targets/vpu/vpu.simics\""});
    } else
        VPUX_THROW("Unsupported launch mode '{0}'", platform);

    std::string args("");
    for (auto& arg : _app.runArgs)
        args += arg + " ";
    printf("%s\n", args.c_str());
}

//
// parseAppConfig
//

void IMDExecutor::parseAppConfig(const std::string_view platform, const Config& config) {
    VPUX_THROW_UNLESS(platformSupported(platform), "Platform '{0}' is not supported", platform);

    const auto mode = config.get<LAUNCH_MODE>();

    switch (mode) {
    case LaunchMode::Simulator: {
        if (platform == Platform::NPU4000) {
            setSimicsRunArgs(platform, config);
        } else {
            setMoviSimRunArgs(platform, config);
        }
        break;
    }
    case LaunchMode::MoviDebug: {
        setMoviDebugRunArgs(platform, config);
        break;
    }
    default:
        VPUX_THROW("Unsupported launch mode '{0}'", stringifyEnum(mode));
    }

    VPUX_THROW_UNLESS(isValidElfSignature(_app.elfFile),
                      "Elf signature check failed for {0}. Please fetch the file using `git lfs pull`, then rebuild "
                      "the project or the `npu_imd_backend_copy_app` cmake target.",
                      _app.elfFile);

    _app.timeoutSec = config.get<MV_RUN_TIMEOUT>().count();
}

//
// Base interface API implementation
//

IMDExecutor::IMDExecutor(const std::string_view platform, const std::shared_ptr<const NetworkDescription>& network,
                         const Config& config)
        : _network(network), _log("InferenceManagerDemo", vpux::getLogLevel(config)) {
    parseAppConfig(ov::intel_npu::Platform::standardize(platform), config);
}

}  // namespace intel_npu
