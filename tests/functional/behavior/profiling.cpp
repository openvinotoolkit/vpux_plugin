//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/core/env.hpp"
#include "vpux/utils/core/logger.hpp"

#include "npu_private_properties.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <algorithm>
#include <cstdio>
#include <optional>
#include <sstream>
#include <utility>

class ProfilingTempReportEnv {
public:
    ProfilingTempReportEnv(const char* type = "JSON") {
        auto filename = std::tmpnam(tempName);
        EXPECT_FALSE(filename == nullptr);
        EXPECT_TRUE(vpux::env::getEnvVar("NPU_PRINT_PROFILING") == std::nullopt);
        vpux::env::setEnvVar("NPU_PRINT_PROFILING", type);
        vpux::env::setEnvVar("NPU_PROFILING_OUTPUT_FILE", filename);
    }
    void cleanup() {
        std::remove(tempName);
        vpux::env::unsetEnvVar("NPU_PRINT_PROFILING");
        vpux::env::unsetEnvVar("NPU_PROFILING_OUTPUT_FILE");
    }
    std::string readAsString() {
        std::ifstream input(tempName);
        input.exceptions(std::ios_base::badbit | std::ios_base::failbit);
        return std::string(std::istreambuf_iterator<std::string::value_type>(input), {});
    }

private:
    ProfilingTempReportEnv(const ProfilingTempReportEnv&) = delete;
    void operator=(const ProfilingTempReportEnv&) = delete;

    char tempName[L_tmpnam] = "";
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, std::optional<T> opt) {
    if (opt.has_value()) {
        return stream << *opt;
    } else {
        return stream << "<nullopt>";
    }
}

template <typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

class ProfilingSubgraphTestBase : public ov::test::VpuOv2LayerTest {
protected:
    void SetUp() override {
        createSubgraphFunction();
    }

    void createSubgraphFunction() {
        const auto type = ov::element::f32;

        ov::Shape inputShape = {1, 3, 128, 128};
        auto param = std::make_shared<ov::op::v0::Parameter>(type, inputShape);

        const std::vector<float> exponent({1.0f});
        const auto power_const = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{1}, exponent);
        auto pow = std::make_shared<ov::op::v1::Power>(param, power_const);
        pow->set_friendly_name("Power");

        auto add = std::make_shared<ov::op::v1::Add>(param, pow->output(0));
        add->set_friendly_name("Add");

        auto softmax = std::make_shared<ov::op::v1::Softmax>(add, /*axis*/ 1);
        softmax->set_friendly_name("Softmax");

        auto result = std::make_shared<ov::op::v0::Result>(softmax->output(0));

        ov::ParameterVector params{param};
        ov::ResultVector results{result};
        function = std::make_shared<ov::Model>(results, params, "ProfSubgraph");

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));
    }
};

typedef std::string PlatformId;

class ProfilingSubgraphSanityTest : public ProfilingSubgraphTestBase, public testing::WithParamInterface<PlatformId> {
protected:
    void runTest() {
        const auto platform = GetParam();
        run(platform);
    }
};

// expected minimal execution cpu time and expected maximal execution cpu and real times in us
using LayerTimingMap = std::map<std::string, std::tuple<unsigned, unsigned, unsigned>>;

// use tuples to get pretty printing for free
using ProfilingTestConfig = std::tuple<std::string, std::optional<unsigned>, LayerTimingMap>;
using ProfilingTestParams = std::tuple<ProfilingTestConfig, std::string>;

class ProfilingSubgraphTest :
        public ProfilingSubgraphTestBase,
        public testing::WithParamInterface<ProfilingTestParams> {
protected:
    ProfilingSubgraphTest(): log(vpux::Logger::global().nest("prof", 0)) {
    }

    void runTest() {
        const auto& config = std::get<0>(GetParam());
        const auto platform = std::get<0>(config);
        run(platform);
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ProfilingSubgraphTest::ParamType>& info) {
        const auto& [config, compilerType] = info.param;
        const auto& [platform, nTiles, timing] = config;
        std::stringstream name;
        name << platform << '_';
        if (nTiles.has_value()) {
            name << *nTiles << "T_";
        }
        name << compilerType;
        return name.str();
    }

private:
    void SetUp() override {
        const auto& [config, compilerType] = GetParam();
        const auto& [platform, nTiles, timing] = config;

        configuration.emplace(ov::enable_profiling.name(), true);
        configuration.emplace(ov::intel_npu::compiler_type.name(), compilerType);
        if (nTiles.has_value()) {
            configuration.emplace(ov::intel_npu::dpu_groups.name(), nTiles.value());
        }

        ProfilingSubgraphTestBase::SetUp();
    }

    virtual void infer() override {
        ov::test::VpuOv2LayerTest::infer();
        checkProfilingOutput();
    }

    void checkProfilingOutput() {
        ProfilingTempReportEnv tempReport;
        auto profData = inferRequest.get_profiling_info();
        auto jsonReport = tempReport.readAsString();
        tempReport.cleanup();

        ASSERT_TRUE(profData.size() > 0);

        for (const auto& profInfo : profData) {
            checkLayer(profInfo, jsonReport);
        }
    }

    void checkLayer(ov::ProfilingInfo profInfo, const std::string& jsonReport) {
        std::set<std::string> layerExecTypes = {"DPU", "Shave", "DMA"};

        auto inTheRange = [](long long val, long long min, long long max) {
            return val >= min && val <= max;
        };
        auto inAllowedExecTypes = [&](std::string execType) {
            return layerExecTypes.find(execType) != layerExecTypes.end();
        };

        const auto& layerName = profInfo.node_name;
        const auto cpuTime = profInfo.cpu_time.count();
        const auto realTime = profInfo.real_time.count();
        log.info("Layer {0} '{1}' ({2}) cpu: {3} us, real: {4} us", profInfo.node_type, layerName, profInfo.exec_type,
                 cpuTime, realTime);

        ASSERT_PRED1(inAllowedExecTypes, profInfo.exec_type);

        const auto& config = std::get<0>(GetParam());
        const auto& layerExecTimes = std::get<2>(config);
        auto execTime = layerExecTimes.find(layerName);
        if (execTime == layerExecTimes.end()) {
            log.warning("Unexpected layer: {0}", layerName);
            return;
        }
        const auto& expectedExecTimes = execTime->second;

        ASSERT_TRUE(jsonReport.find(layerName) != std::string::npos)
                << "Could not find the expected layer name: " << layerName << " in the json report.";

        auto [expectedMinCpuTimeUs, expectedMaxCpuTimeUs, expectedMaxRealTimeUs] = expectedExecTimes;

        ASSERT_PRED3(inTheRange, cpuTime, expectedMinCpuTimeUs, expectedMaxCpuTimeUs)
                << "CPU time " << cpuTime << "us is out of range.";
        ASSERT_PRED3(inTheRange, realTime, expectedMinCpuTimeUs, expectedMaxRealTimeUs)
                << "real time " << realTime << "us is out of range.";
        // real time can be smaller than the cpu time depending on number of tiles. We use the
        // expectedMinCpuTimeUs bound, which accounts for the dispersion due to the number of tiles.
    }

    vpux::Logger log;
};

// Profiling disabled sanity test-case

TEST_P(ProfilingSubgraphSanityTest, ProfilingDisabledTest) {
    runTest();
}

INSTANTIATE_TEST_SUITE_P(precommit_BehaviorTest_ProfilingDisabledTest, ProfilingSubgraphSanityTest,
                         testing::Values(ov::intel_npu::Platform::NPU3700, ov::intel_npu::Platform::NPU3720,
                                         ov::intel_npu::Platform::NPU4000),
                         [](const testing::TestParamInfo<PlatformId>& info) {
                             return to_string(info.param);
                         });

// Profiling enabled test cases

// For given layer types the experimental values of cpu/real time typically fall into range [us]:
//      Add (SW): [556, 558] / [766, 767]
//      SoftMax (SW): [3502, 3505] / [3509, 3510]
// For given layer types the experimental values of cpu/real time typically fall into range [us]:
//      Add (SW): [556, 558] / [766, 767]
//      SoftMax (SW): [3502, 3505] / [3509, 3510]
const static LayerTimingMap NPU3700_Timing = {{"Add", std::make_tuple(50, 8000, 8000)},
                                              {"Softmax", std::make_tuple(300, 35000, 35000)}};

// Empirical values depend on layer and timing measurement type
// (realTime vs cpu time), NPU HW used, compiler/driver task scheduling strategy, and network structure and are
// subject to intrinsic performance variations. For given layer types the experimentally observed values for
// cpu/real time for MLIR case are [us]:
//      Power (SW): [3012, 8949] / [3020, 7084]
//      Add (DPU): [24, 39] / [2994, 5125]
//      SoftMax (SW): [782, 2045] / [777, 1811]
// for for DRIVER case:
//      Power (SW): [3013, 5153] / [3021, 5167]
//      Add (DPU): [24, 40] / [17, 28]
//      SoftMax (SW): [786, 2044] / [781, 1810]
const static LayerTimingMap NPU3720_Timing = {{"Power", std::make_tuple(300, 90000, 90000)},
                                              {"Add", std::make_tuple(2, 400, 50000)},
                                              {"Softmax", std::make_tuple(80, 20000, 20000)}};

// For 1-tile case cpu and real execution times are similar. For given layer types the experimentally observed
// values for MLIR case are [us]:
//      Power (SW): [17601, 18093]
//      Add (DPU): [19, 21]
//      SoftMax (SW): [8718, 9045]
const static LayerTimingMap NPU4000_T1_Timing = {{"Power", std::make_tuple(1467, 181620, 181620)},
                                                 {"Add", std::make_tuple(2, 230, 230)},
                                                 {"Softmax", std::make_tuple(872, 90760, 90760)}};

// For 2-tile case cpu and real execution times are similar. For given layer types the experimentally observed
// values for cpu/real time for MLIR case are [us]:
//      Power (SW): [17591, 18094]
//      Add (DPU): [10, 10] / [14, 16]
//      SoftMax (SW): [4370, 4625]
const static LayerTimingMap NPU4000_T2_Timing = {{"Power", std::make_tuple(700, 180940, 180940)},
                                                 {"Add", std::make_tuple(1, 100, 160)},
                                                 {"Softmax", std::make_tuple(437, 46250, 46250)}};

TEST_P(ProfilingSubgraphTest, ProfilingTest) {
    const auto& [config, compilerType] = GetParam();
    const auto platform = std::get<0>(config);
    if (platform == ov::intel_npu::Platform::NPU3700 && compilerType == "DRIVER") {
        GTEST_SKIP();
    }
    runTest();
}

INSTANTIATE_TEST_SUITE_P(
        precommit_BehaviorTest_ProfilingTest, ProfilingSubgraphTest,
        testing::Combine(
                testing::Values(ProfilingTestConfig{ov::intel_npu::Platform::NPU3700, std::nullopt, NPU3700_Timing},
                                ProfilingTestConfig{ov::intel_npu::Platform::NPU3720, std::nullopt, NPU3720_Timing},
                                ProfilingTestConfig{ov::intel_npu::Platform::NPU4000, 1, NPU4000_T1_Timing},
                                ProfilingTestConfig{ov::intel_npu::Platform::NPU4000, 2, NPU4000_T2_Timing}),
                testing::Values("MLIR", "DRIVER")),
        ProfilingSubgraphTest::getTestCaseName);
