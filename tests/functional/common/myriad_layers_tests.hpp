// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <ie_version.hpp>
#include <ie_device.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <ie_plugin_dispatcher.hpp>
#include <inference_engine.hpp>
#include "tests_common.hpp"
#include <algorithm>
#include <cstddef>
#include <inference_engine/precision_utils.h>
#include <tuple>
#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include <vpu/vpu_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>
#include "myriad_layers_reference_functions.hpp"
#include "vpu_layers_tests.hpp"

/* Function to calculate CHW dimensions for the blob generated by */
/* Myriad/HDDL plugin.                                            */

class myriadLayersTests_nightly : public vpuLayersTests {
public:
    void NetworkInit(const std::string& layer_type,
                std::map<std::string, std::string>* params = nullptr,
                int weights_size = 0,
                int biases_size = 0,
                InferenceEngine::TBlob<uint8_t>::Ptr weights = nullptr,
                InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32,
                InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP16,
                bool useHWOpt = false);

private:
    void doNetworkInit(const std::string& layer_type,
            std::map<std::string, std::string>* params = nullptr,
            int weights_size = 0,
            int biases_size = 0,
            InferenceEngine::TBlob<uint8_t>::Ptr weights = nullptr,
            InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32,
            InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP16,
            bool useHWOpt = false);
};

template<class T>
class myriadLayerTestBaseWithParam: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<T> {
};

/* common classes for different basic tests */
extern const char POOLING_MAX[];
extern const char POOLING_AVG[];

struct pooling_layer_params {
    param_size kernel;
    param_size stride;
    param_size pad;
};

template <const char* poolType, typename... Types>
class PoolingTest : public myriadLayersTests_nightly,
                    public testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, const char*, Types...>>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, const char*, Types...>>::GetParam();
        _input_tensor = std::get<0>(p);
        _kernel_val   = std::get<1>(p).kernel;
        _stride_val   = std::get<1>(p).stride;
        _pad_val      = std::get<1>(p).pad;
        _layout       = std::get<2>(p);
        _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = _layout;
        if (_pad_val.x >= _kernel_val.x) {
            _pad_val.x = _kernel_val.x - 1;
        }
        if (_pad_val.y >= _kernel_val.y) {
            _pad_val.y = _kernel_val.y - 1;
        }
        _params["kernel-x"] = std::to_string(_kernel_val.x);
        _params["kernel-y"] = std::to_string(_kernel_val.y);
        _params["stride-x"] = std::to_string(_stride_val.x);
        _params["stride-y"] = std::to_string(_stride_val.y);
        _params["pad-x"] = std::to_string(_pad_val.x);
        _params["pad-y"] = std::to_string(_pad_val.y);
        _params["pool-method"] = poolType;
        _output_tensor.resize(4);
        _output_tensor[3] = std::ceil((_input_tensor[3] + 2. * _pad_val.x - _kernel_val.x) / _stride_val.x + 1);
        _output_tensor[2] = std::ceil((_input_tensor[2] + 2. * _pad_val.y - _kernel_val.y) / _stride_val.y + 1);
        _output_tensor[1] = _input_tensor[1];
        _output_tensor[0] = 1;
        ASSERT_EQ(_input_tensor.size(), 4);
        AddLayer("Pooling",
             &_params,
             {_input_tensor},
             {_output_tensor},
             ref_pooling_wrap);
    }

    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    param_size _kernel_val;
    param_size _stride_val;
    param_size _pad_val;
    const char* _layout;
    std::map<std::string, std::string> _params;
};

template <const char* poolType/*, typename... Types*/>
class GlobalPoolingTest : public myriadLayersTests_nightly,
                    public testing::WithParamInterface</*std::tuple<*/InferenceEngine::SizeVector/*, param_size, Types...>*/>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        auto p = ::testing::WithParamInterface<InferenceEngine::SizeVector/*, param_size, Types...>*/>::GetParam();
        _input_tensor = p/*std::get<0>(p)*/;

        _kernel_val   = {_input_tensor[3], _input_tensor[2]};
        _stride_val   = {1, 1};
        _pad_val      = {0, 0};

#if 0 // 4DGP
        // TODO: make it the test argument
        _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = VPU_CONFIG_VALUE(NCHW);
//        _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = VPU_CONFIG_VALUE(NHWC);
#endif

        _params["kernel-x"] = std::to_string(_kernel_val.x);
        _params["kernel-y"] = std::to_string(_kernel_val.y);
        _params["stride-x"] = std::to_string(_stride_val.x);
        _params["stride-y"] = std::to_string(_stride_val.y);
        _params["pad-x"] = std::to_string(_pad_val.x);
        _params["pad-y"] = std::to_string(_pad_val.y);
        _params["pool-method"] = poolType;
        _output_tensor.resize(4);
        _output_tensor[3] = std::ceil((_input_tensor[3] + 2. * _pad_val.x - _kernel_val.x) / _stride_val.x + 1);
        _output_tensor[2] = std::ceil((_input_tensor[2] + 2. * _pad_val.y - _kernel_val.y) / _stride_val.y + 1);
        _output_tensor[1] = _input_tensor[1];
        _output_tensor[0] = _input_tensor[0];
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
        ASSERT_EQ(_input_tensor.size(), 4);
        AddLayer("Pooling",
             &_params,
             {_input_tensor},
             {_output_tensor},
             ref_pooling_wrap);
    }

    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    param_size _kernel_val;
    param_size _stride_val;
    param_size _pad_val;
    std::map<std::string, std::string> _params;
};

template <const char* poolType, typename... Types>
class PoolingTestPad4 : public myriadLayersTests_nightly,
                    public testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, paddings4, const char*, Types...>>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, paddings4, const char*, Types...>>::GetParam();
        _input_tensor = std::get<0>(p);
        _kernel_val   = std::get<1>(p);
        _stride_val   = std::get<2>(p);
        _pad_val      = std::get<3>(p);
        _layout       = std::get<4>(p);
        _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = _layout;
        if (_pad_val.left >= _kernel_val.x) {
            _pad_val.left = _kernel_val.x - 1;
        }
        if (_pad_val.right >= _kernel_val.x) {
            _pad_val.right = _kernel_val.x - 1;
        }
        if (_pad_val.top >= _kernel_val.y) {
            _pad_val.top = _kernel_val.y - 1;
        }
        if (_pad_val.bottom >= _kernel_val.y) {
            _pad_val.bottom = _kernel_val.y - 1;
        }
        _params["kernel-x"] = std::to_string(_kernel_val.x);
        _params["kernel-y"] = std::to_string(_kernel_val.y);
        _params["stride-x"] = std::to_string(_stride_val.x);
        _params["stride-y"] = std::to_string(_stride_val.y);
        _params["pad-x"] = std::to_string(_pad_val.left);
        _params["pad-y"] = std::to_string(_pad_val.top);
        _params["pool-method"] = poolType;
        _output_tensor.resize(4);
        _output_tensor[3] = std::ceil((_input_tensor[3] + _pad_val.left + _pad_val.right  - _kernel_val.x) / _stride_val.x + 1);
        _output_tensor[2] = std::ceil((_input_tensor[2] + _pad_val.top  + _pad_val.bottom - _kernel_val.y) / _stride_val.y + 1);
        _output_tensor[1] = _input_tensor[1];
        _output_tensor[0] = 1;
        ASSERT_EQ(_input_tensor.size(), 4);
        AddLayer("Pooling",
             &_params,
             {_input_tensor},
             {_output_tensor},
             ref_pooling_wrap);
    }

    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    param_size _kernel_val;
    param_size _stride_val;
    paddings4 _pad_val;
    const char* _layout;
    std::map<std::string, std::string> _params;
};

template <typename... Types>
class ConvolutionTest : public myriadLayersTests_nightly,
                        public testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, Types...>>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, Types...>>::GetParam();
        _input_tensor = std::get<0>(p);
        kernel = std::get<1>(p);
        param_size stride = std::get<2>(p);
        param_size pad = std::get<3>(p);
        size_t out_channels = std::get<4>(p);
        group = std::get<5>(p);
        get_dims(_input_tensor, IW, IH,IC);
        size_t out_w = (IW + 2 * pad.x - kernel.x + stride.x) / stride.x;
        size_t out_h = (IH + 2 * pad.y - kernel.y + stride.y) / stride.y;

        gen_dims(_output_tensor, _input_tensor.size(), out_w, out_h, out_channels);

        size_t num_weights = kernel.x * kernel.y * (IC / group) * out_channels;
        size_t num_bias    = out_channels;

        std::map<std::string, std::string> layer_params = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"output", std::to_string(out_channels)}
                , {"group", std::to_string(group)}
        };
        AddLayer("Convolution",
                 &layer_params,
                 num_weights,
                 num_bias,
                 defaultWeightsRange,
                 {_input_tensor},
                 {_output_tensor},
                 ref_convolution_wrap);
    }
    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    size_t  group = 0;
    param_size kernel;
};

template <typename... Types>
class FCTest : public myriadLayersTests_nightly,
               public testing::WithParamInterface<std::tuple<fcon_test_params, int32_t, int32_t, Types...>>
{
public:
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        auto p = ::testing::WithParamInterface<std::tuple<fcon_test_params, int32_t, int32_t, Types...>>::GetParam();
        _par = std::get<0>(p);
        int32_t input_dim = std::get<1>(p);
        int32_t add_bias = std::get<2>(p);
        std::map<std::string, std::string> params;
        params["out-size"] = std::to_string(_par.out_c);
        int32_t IW = _par.in.w;
        int32_t IH = _par.in.h;
        int32_t IC = _par.in.c;
        gen_dims(_input_tensor, input_dim, IW, IH, IC);

        _output_tensor.push_back(1);
        _output_tensor.push_back(_par.out_c);

        size_t sz_weights = IC * IH * IW * _par.out_c;
        size_t sz_bias = 0;
        if (add_bias) {
            sz_bias = _par.out_c;
        }
        size_t sz = sz_weights + sz_bias;
        AddLayer("FullyConnected",
                 &params,
                 sz_weights,
                 sz_bias,
                 defaultWeightsRange,
                 {_input_tensor},
                 {_output_tensor},
                 ref_innerproduct_wrap);
    }
    InferenceEngine::SizeVector _input_tensor;
    InferenceEngine::SizeVector _output_tensor;
    fcon_test_params _par;
};

/* parameters definitions for the tests with several layers within the NET */
extern const std::vector<InferenceEngine::SizeVector> g_poolingInput;
extern const std::vector<InferenceEngine::SizeVector> g_poolingInput_postOp;
extern const std::vector<pooling_layer_params> g_poolingLayerParamsFull;
extern const std::vector<pooling_layer_params> g_poolingLayerParamsLite;
extern const std::vector<const char*> g_poolingLayout;
extern const std::vector<InferenceEngine::SizeVector> g_convolutionTensors;
extern const std::vector<InferenceEngine::SizeVector> g_convolutionTensors_postOp;
extern const std::vector<fcon_test_params> g_fcTestParamsSubset;
extern const std::vector<int32_t> g_dimensionsFC;
extern const std::vector<int32_t> g_addBiasFC;
