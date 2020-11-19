// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include "kmb_test_env_cfg.hpp"
#include "kmb_test_tool.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsUtils {

class KmbLayerTestsCommon : virtual public LayerTestsCommon {
protected:
    static const KmbTestEnvConfig envConfig;
    KmbTestTool kmbTestTool;
public:
    explicit KmbLayerTestsCommon();
    void Run() override;
protected:
    void BuildNetworkWithoutCompile();
    virtual void ImportNetwork();
    void ExportNetwork();

    void Validate() override;
    void Compare(const std::vector<std::vector<std::uint8_t>>& expectedOutputs, const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) override;
    std::vector<std::vector<std::uint8_t>> CalculateRefs() override;

    virtual void SkipBeforeLoad() {}
    virtual void SkipBeforeImport() {}
    virtual void SkipBeforeInfer() {}
    virtual void SkipBeforeValidate() {}
};

class KmbSkipTestException: public std::runtime_error {
public:
    KmbSkipTestException(const std::string& what_arg): runtime_error(what_arg) {};
};

extern const TargetDevice testPlatformTargetDevice;

}  // namespace LayerTestsUtils
