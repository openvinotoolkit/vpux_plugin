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

namespace intel_npu {

class IMDExecutor final : public IExecutor {
public:
    struct InferenceManagerDemo;

    IMDExecutor(const std::string_view, const std::shared_ptr<const NetworkDescription>& network, const Config& config);

    std::shared_ptr<const NetworkDescription>& getNetworkDesc() {
        return _network;
    }

    InferenceManagerDemo& getApp() {
        return _app;
    }

    struct InferenceManagerDemo final {
        std::string elfFile;
        std::string runProgram;
        llvm::SmallVector<std::string> runArgs;
        int64_t timeoutSec;
        std::string chipsetArg;
        std::string imdElfArg;
    };

    void setWorkloadType(const ov::WorkloadType workloadType) const override;

private:
    std::string getMoviToolsPath(const Config& config);
    std::string getSimicsPath(const Config& config);
    void setElfFile(const std::string& bin);
    void setMoviSimRunArgs(const std::string_view platform, const Config& config);
    void setMoviDebugRunArgs(const std::string_view platform, const Config& config);
    void setSimicsRunArgs(const std::string_view platform, const Config& config);

    static bool isValidElfSignature(llvm::StringRef filePath);
    void parseAppConfig(const std::string_view platform, const Config& config);

    std::shared_ptr<const NetworkDescription> _network;
    vpux::Logger _log;

    InferenceManagerDemo _app;
};

}  // namespace intel_npu
