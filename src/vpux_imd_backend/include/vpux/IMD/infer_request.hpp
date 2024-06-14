//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "intel_npu/al/icompiled_model.hpp"
#include "npu.hpp"
#include "npu_private_properties.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <openvino/runtime/profiling_info.hpp>

namespace vpux {

class IMDInferRequest final : public intel_npu::SyncInferRequest {
public:
    explicit IMDInferRequest(const std::shared_ptr<const intel_npu::ICompiledModel>& compiledModel,
                             const std::shared_ptr<intel_npu::IExecutor>& executor, const intel_npu::Config& config);

    void infer() override;
    void infer_async() override;

    void get_result() override;

private:
    void check_network_precision(const ov::element::Type_t precision) override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<uint8_t> get_raw_profiling_data() const;

    SmallString create_temporary_work_directory();
    void store_compiled_model();
    void store_network_inputs();
    void run_app();
    void read_from_file(const std::string& path, const std::shared_ptr<ov::ITensor>& tensor,
                        const bool isDynamic = false);
    void load_network_outputs();

    SmallString _workDirectory;
    const std::shared_ptr<intel_npu::IExecutor> _executorPtr;
    const intel_npu::Config _config;
    Logger _logger;

    std::unordered_map<std::string, size_t> _inputOrder;
    std::unordered_map<std::string, size_t> _outputOrder;

    std::shared_ptr<ov::ITensor> _rawProfilingData;
};

namespace profiling {

using LayerStatistics = std::vector<ov::ProfilingInfo>;
LayerStatistics getLayerStatistics(const uint8_t* rawData, size_t dataSize, const std::vector<char>& blob);

}  //  namespace profiling

}  //  namespace vpux
