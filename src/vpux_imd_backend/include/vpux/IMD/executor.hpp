//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "npu.hpp"
#include "npu_private_properties.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <string>

namespace vpux {

class IMDExecutor final : public intel_npu::IExecutor {
public:
    struct InferenceManagerDemo;

    IMDExecutor(const std::string_view, const std::shared_ptr<const intel_npu::NetworkDescription>& network,
                const intel_npu::Config& config);

    std::shared_ptr<const intel_npu::NetworkDescription>& getNetworkDesc() {
        return _network;
    }

    InferenceManagerDemo& getApp() {
        return _app;
    }

    struct InferenceManagerDemo final {
        std::string elfFile;
        std::string runProgram;
        SmallVector<std::string> runArgs;
        int64_t timeoutSec;
        std::string chipsetArg;
        std::string imdElfArg;
    };

private:
    std::string getMoviToolsPath(const intel_npu::Config& config);
    std::string getSimicsPath(const intel_npu::Config& config);
    void setElfFile(const std::string& bin);
    void setMoviSimRunArgs(const std::string_view platform, const intel_npu::Config& config);
    void setMoviDebugRunArgs(const std::string_view platform, const intel_npu::Config& config);
    void setSimicsRunArgs(const std::string_view platform, const intel_npu::Config& config);

    static bool isValidElfSignature(StringRef filePath);
    void parseAppConfig(const std::string_view platform, const intel_npu::Config& config);

    std::shared_ptr<const intel_npu::NetworkDescription> _network;
    Logger _log;

    InferenceManagerDemo _app;
};

}  // namespace vpux
