//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <vpu/kmb_plugin_config.hpp>

#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"

#include <ie_icnn_network_stats.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <ie_util_internal.hpp>
#include <conv_ref.hpp>

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;
using namespace details;

#ifdef ENABLE_MCM_COMPILER
TEST_F(kmbLayersTests_nightly, DISABLED_TestsConvolutionAfterScaleShift) {
    // TODO: tests fails. mcmCompiler compilation (Convolution with bias): Segmentation fault. Jira: VPUNND-1474
    const std::string model = conv_after_scale_shift;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 6 + 18816;
    std::size_t biasSize = 6 + 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t >(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsConvolutionAfterScaleShiftNoBias) {
    std::string model = conv_after_scale_shift;
    REPLACE_WITH_STR(model, "<biases offset=\"6\" size=\"6\"/>", " ");
    REPLACE_WITH_STR(model, "<biases offset=\"18828\" size=\"128\"/>", " ");

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 6 + 18816;
    std::size_t biasSize = 6 + 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t >(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsQuantizedConvolutionAfterScaleShift) {
    // TODO: Test fails. mcmCompiler can not compile the network (Convolution with bias). Jira: VPUNND-1474
    const std::string model = full_quant_model;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::map<std::string, std::string> config;
    details::CNNNetworkImplPtr clonedNetwork;

    setCommonConfig(config);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    std::size_t weightSize = 147456 + 65536;
    std::size_t biasSize = 256 + 1024;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t >(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP32);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv2"]->setPrecision(Precision::FP32);

    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)network).getStats(&pstats, nullptr);

    ASSERT_EQ(StatusCode::OK, s);

    if (!pstats->isEmpty()) {
        clonedNetwork = cloneNet(network);
        details::CNNNetworkInt8Normalizer::NormalizeNetwork(*clonedNetwork, *pstats);

        ASSERT_NO_THROW(ie.LoadNetwork(CNNNetwork(clonedNetwork), "kmb", config));
    }
}

//  TODO: mcmCompiler assert: 'extendToK parameters dimensions doesn't match size of output_channels or 1'
//  JIRA Bug: VPUNND-1494
TEST_F(kmbLayersTests_nightly, DISABLED_TestsQuantizedConvolutionAfterScaleShiftNoBias) {
    std::string model = full_quant_model;

    REPLACE_WITH_STR(model, "<biases offset=\"147456\" size=\"256\"/>", " ");
    REPLACE_WITH_STR(model, "<biases offset=\"213248\" size=\"1024\"/>", " ");

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::map<std::string, std::string> config;
    details::CNNNetworkImplPtr clonedNetwork;

    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    std::size_t weightSize = 147456 + 65536;
    std::size_t biasSize = 256 + 1024;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t >(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP32);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv2"]->setPrecision(Precision::FP32);

    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)network).getStats(&pstats, nullptr);
    ASSERT_EQ(StatusCode::OK, s);

    if (!pstats->isEmpty()) {
        clonedNetwork = cloneNet(network);
        details::CNNNetworkInt8Normalizer::NormalizeNetwork(*clonedNetwork, *pstats);

        ASSERT_NO_THROW(ie.LoadNetwork(CNNNetwork(clonedNetwork), "kmb", config));
    }

}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsConvolutionOnly) {
    const std::string model = convolution_only;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 18816;
    std::size_t biasSize = 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t >(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    // LoadNetwork results in the following message when MCM_PARSING_ONLY is set to 'NO':
    // The maximum peak memory requirment of the graph exceeds CMX and the partial serialisation algorithm is unable
    // to reduce parallelism, exiting now, this is normal behaviour
    // TODO disable 'parse only' and find out why it happens
    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsConvolutionOnlyNoBias) {
    std::string model = convolution_only;
    REPLACE_WITH_STR(model, "<biases offset=\"18816\" size=\"128\"/>", " ");

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 18816;
    std::size_t biasSize = 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t >(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    // LoadNetwork results in the following message when MCM_PARSING_ONLY is set to 'NO':
    // The maximum peak memory requirment of the graph exceeds CMX and the partial serialisation algorithm is unable
    // to reduce parallelism, exiting now, this is normal behaviour
    // TODO disable 'parse only' and find out why it happens
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}
#endif

struct convolution_test_params {
    SizeVector input_dim;
    conv_common_params conv_params;
};

size_t getConvWeightsByteSize(const std::vector<size_t>& inShape,
                          const conv_common_params& params,
                          const std::string& precision) {
    if (params.group != 0lu && params.out_c != 0lu && params.kernel.size() != 0lu) {
        size_t type_size = 1lu;
        if (precision == "FP32")
            type_size = sizeof(float);
        else if (precision == "FP16") {
            type_size = sizeof(ie_fp16);
        }

        int weights_size = type_size * inShape[1] * params.out_c / params.group;
        for (size_t i = 0lu; i < params.kernel.size(); i++) {
            weights_size *= params.kernel[i];
        }

        return weights_size;
    }

    return 0lu;
}

static void setCommonConfig(std::map<std::string, std::string>& config)
{
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)]  = CONFIG_VALUE(NO);

    const ::testing::TestInfo* const test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();

    config[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH)] = test_info->test_case_name();
    config[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS)] = test_info->name();
}

class ConvolutionTest : public ::testing::Test, public testing::WithParamInterface<convolution_test_params> {};

// Crash in mcmCompiler during parsing of scaleshift layer
// Disabled until issue will be resolved
TEST_P(ConvolutionTest, DISABLED_convolution_only) {
    // Besides weights and biases we need to store FQ blobs as well
    size_t weightsBufferOffset = 48;
    auto input_dims = GetParam().input_dim;
    auto conv_params = GetParam().conv_params;
    SizeVector output_dims;
    getConvOutShape(input_dims, conv_params, output_dims);

    size_t weightsByteSize = getConvWeightsByteSize(input_dims, conv_params, "FP32");
    size_t weightsSize = weightsByteSize / sizeof(float);
    size_t biasByteSize = conv_params.out_c * sizeof(float);
    size_t biasSize = biasByteSize / sizeof(float);

    auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize + weightsBufferOffset}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<float*>();
    std::fill(weightsBufferData, weightsBufferData + (weightsSize + biasSize + weightsBufferOffset / sizeof(float)), 1.0f);

    Core ie;

    std::string blob_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    blob_name += ".blob";
    std::replace(blob_name.begin(), blob_name.end(), '/', '_');

#ifndef __arm__
    std::string model = fq_convolution_only_slim;
    REPLACE_WITH_NUM(model, "_INPUT_BATCH_", input_dims[0]);
    REPLACE_WITH_NUM(model, "_INPUT_CHANNEL_", input_dims[1]);
    REPLACE_WITH_NUM(model, "_INPUT_HEIGHT_", input_dims[2]);
    REPLACE_WITH_NUM(model, "_INPUT_WIDTH_", input_dims[3]);

    REPLACE_WITH_NUM(model, "_WEIGHTS_OFFSET_", weightsBufferOffset);
    REPLACE_WITH_NUM(model, "_WEIGHTS_BYTE_SIZE_", weightsByteSize);

    REPLACE_WITH_NUM(model, "_BIAS_OFFSET_", weightsBufferOffset + weightsByteSize);
    REPLACE_WITH_NUM(model, "_BIAS_BYTE_SIZE_", biasByteSize);

    // Assuming kernel is a square
    REPLACE_WITH_NUM(model, "_KERNEL_SIZE_", conv_params.kernel[0]);
    REPLACE_WITH_NUM_VECTOR(model, "_KERNEL_", conv_params.kernel);
    REPLACE_WITH_NUM_VECTOR(model, "_STRIDE_", conv_params.stride);

    REPLACE_WITH_NUM(model, "_OUTPUT_BATCH_", output_dims[0]);
    REPLACE_WITH_NUM(model, "_OUTPUT_CHANNEL_", output_dims[1]);
    REPLACE_WITH_NUM(model, "_OUTPUT_HEIGHT_", output_dims[2]);
    REPLACE_WITH_NUM(model, "_OUTPUT_WIDTH_", output_dims[3]);

    CNNNetReader reader;
    ASSERT_NO_THROW(reader.ReadNetwork(model.data(), model.length()));
    ASSERT_NO_THROW(reader.SetWeights(weightsBuffer));
    ASSERT_TRUE(reader.isParseSuccess());

    CNNNetwork network = reader.getNetwork();

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);

    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, "kmb", config));
    ASSERT_NO_THROW( executableNetwork.Export(blob_name));

#else

#endif
}

// Disabled due to bug in mcmCompiler with Release build type.
// Corresponding Jira ticket VPUNND-1910
TEST_P(ConvolutionTest, DISABLED_u8_convolution_only) {
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

    Core ie;
    auto input_dims = GetParam().input_dim;
    auto conv_params = GetParam().conv_params;
    SizeVector output_dims;
    getConvOutShape(input_dims, conv_params, output_dims);

    size_t weightsByteSize = getConvWeightsByteSize(input_dims, conv_params, "U8");
    size_t weightsSize = weightsByteSize / sizeof(uint8_t);

    size_t biasByteSize = output_dims[1] * sizeof(int32_t);
    size_t biasSize = biasByteSize / sizeof(int32_t);

    auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<uint8_t*>();
    for (size_t i = 0; i < weightsSize; ++i) {
        weightsBufferData[i] = 1;
    }

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    for (size_t i = 0; i < biasSize; ++i) {
        biasData[i] = 1lu;
    }

    std::string blob_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    blob_name += ".blob";
    std::replace(blob_name.begin(), blob_name.end(), '/', '_');

#ifndef __arm__
    std::string model = fq_convolution_only_u8_slim;
    REPLACE_WITH_NUM(model, "_INPUT_BATCH_", input_dims[0]);
    REPLACE_WITH_NUM(model, "_INPUT_CHANNEL_", input_dims[1]);
    REPLACE_WITH_NUM(model, "_INPUT_HEIGHT_", input_dims[2]);
    REPLACE_WITH_NUM(model, "_INPUT_WIDTH_", input_dims[3]);

    REPLACE_WITH_NUM(model, "_WEIGHTS_BYTE_SIZE_", weightsByteSize);
    REPLACE_WITH_NUM(model, "_BIAS_BYTE_SIZE_", biasByteSize);
    REPLACE_WITH_NUM(model, "_BIAS_OFFSET_", weightsByteSize);

    // Assuming kernel is a square
    REPLACE_WITH_NUM(model, "_KERNEL_SIZE_", conv_params.kernel[0]);
    REPLACE_WITH_NUM_VECTOR(model, "_KERNEL_", conv_params.kernel);
    REPLACE_WITH_NUM_VECTOR(model, "_STRIDE_", conv_params.stride);

    REPLACE_WITH_NUM(model, "_OUTPUT_BATCH_", output_dims[0]);
    REPLACE_WITH_NUM(model, "_OUTPUT_CHANNEL_", output_dims[1]);
    REPLACE_WITH_NUM(model, "_OUTPUT_HEIGHT_", output_dims[2]);
    REPLACE_WITH_NUM(model, "_OUTPUT_WIDTH_", output_dims[3]);

    CNNNetReader reader;
    ASSERT_NO_THROW(reader.ReadNetwork(model.data(), model.length()));
    ASSERT_NO_THROW(reader.SetWeights(weightsBuffer));
    ASSERT_TRUE(reader.isParseSuccess());

    CNNNetwork network = reader.getNetwork();

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv2"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);

    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, "kmb", config));
    ASSERT_NO_THROW( executableNetwork.Export(blob_name));

#else

#endif
}

// Assuming input and output are in NCHW layout
// {input_dim}, {stride}, {kernel}, {pads_begin}, {pads_end}, {dilation}, "", group, out_c, with_bias, with_weights, quantization_level};
std::vector<convolution_test_params> test_params = {
        {{1, 64, 16, 16}, {{1, 1}, {2, 2}, {0, 0}, {0, 0}, {0, 0}, "", 1, 256, false, true, ""}},
        {{1, 64, 16, 16}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {0, 0}, "", 1, 64, false, true, ""}},
        {{1, 64, 16, 16}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {0, 0}, "", 1, 256, false, true, ""}},
};

INSTANTIATE_TEST_CASE_P(accuracy, ConvolutionTest, ::testing::ValuesIn(test_params));
