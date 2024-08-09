//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/hwtest/test_case_json_parser.hpp"

#include <gtest/gtest.h>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

void createCaseGeneratorHeaderJson(llvm::json::OStream& j) {
    j.attribute("architecture", "NPU37XX");
    j.attribute("backend_flow", "Default");
    j.attribute("case_type", "ZMajorConvolution");
    j.attribute("network", "");
    j.attribute("layer_name", "conv2d_u8_to_u8_unit_test");
}

void createCaseGeneratorInputJson(llvm::json::OStream& j) {
    std::array<unsigned long int, 4> shape = {1, 256, 16, 16};
    std::string dtype = "uint8";
    std::vector<double> scale = {0.01};
    unsigned long int zeropoint = 127;
    unsigned long int low_range = 0;
    unsigned long int high_range = 255;

    j.attributeBegin("input");
    {
        j.arrayBegin();
        j.objectBegin();
        j.attributeBegin("shape");
        {
            j.arrayBegin();
            for (auto s : shape)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attribute("dtype", dtype);
        j.attributeBegin("quantization");
        {
            j.objectBegin();
            j.attribute("low_range", low_range);
            j.attribute("high_range", high_range);
            j.attributeBegin("scale");
            {
                j.arrayBegin();
                for (auto s : scale)
                    j.value(s);
                j.arrayEnd();
            }
            j.attributeEnd();
            j.attribute("zeropoint", zeropoint);
            j.objectEnd();
        }
        j.attributeEnd();
        j.objectEnd();
        j.arrayEnd();
    }
    j.attributeEnd();
}

void createCaseGeneratorWeightsJson(llvm::json::OStream& j) {
    std::array<unsigned long int, 4> shape = {64, 256, 1, 1};
    std::string dtype = "uint8";
    std::vector<double> scale = {0.01};
    unsigned long int zeropoint = 0;
    unsigned long int low_range = 1;
    unsigned long int high_range = 1;

    j.attributeBegin("weight");
    {
        j.arrayBegin();
        j.objectBegin();
        j.attributeBegin("shape");
        {
            j.arrayBegin();
            for (auto s : shape)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attribute("dtype", dtype);
        j.attributeBegin("quantization");
        {
            j.objectBegin();
            j.attribute("low_range", low_range);
            j.attribute("high_range", high_range);
            j.attributeBegin("scale");
            {
                j.arrayBegin();
                for (auto s : scale)
                    j.value(s);
                j.arrayEnd();
            }
            j.attributeEnd();
            j.attribute("zeropoint", zeropoint);
            j.objectEnd();
        }
        j.attributeEnd();
        j.objectEnd();
        j.arrayEnd();
    }
    j.attributeEnd();
}

void createCaseGeneratorOutputJson(llvm::json::OStream& j) {
    std::array<unsigned long int, 4> shape = {1, 64, 16, 16};
    std::string dtype = "uint8";
    std::vector<double> scale = {0.01};
    unsigned long int zeropoint = 0;
    unsigned long int low_range = 1;
    unsigned long int high_range = 1;

    j.attributeBegin("output");
    {
        j.objectBegin();
        j.attributeBegin("shape");
        {
            j.arrayBegin();
            for (auto s : shape)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attribute("dtype", dtype);
        j.attributeBegin("quantization");
        {
            j.objectBegin();
            j.attribute("low_range", low_range);
            j.attribute("high_range", high_range);
            j.attributeBegin("scale");
            {
                j.arrayBegin();
                for (auto s : scale)
                    j.value(s);
                j.arrayEnd();
            }
            j.attributeEnd();
            j.attribute("zeropoint", zeropoint);
            j.objectEnd();
        }
        j.attributeEnd();
        j.objectEnd();
    }
    j.attributeEnd();
}

void createCaseGeneratorConvJson(llvm::json::OStream& j) {
    std::array<unsigned long int, 2> stride = {1, 1};
    std::array<unsigned long int, 2> pad = {0, 0};
    unsigned long int group = 1;
    unsigned dilation = 1;

    j.attributeBegin("conv_op");
    {
        j.objectBegin();

        j.attributeBegin("stride");
        {
            j.arrayBegin();
            for (auto s : stride)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attributeBegin("pad");
        {
            j.arrayBegin();
            for (auto s : pad)
                j.value(s);
            j.arrayEnd();
        }
        j.attributeEnd();

        j.attribute("group", group);
        j.attribute("dilation", dilation);

        j.objectEnd();
    }
    j.attributeEnd();
}

void createCaseGeneratorODUPermutationJson(llvm::json::OStream& j) {
    std::string order("nhwc");
    j.attribute("output_order", order);
}

void createAndRunConvTest() {
    auto testConfigFile = "conv_test.json";

    auto createNumericsBenchJsonSpec = [&]() {
        std::error_code ec;
        llvm::raw_fd_ostream jsonFd(testConfigFile, ec);
        llvm::json::OStream j(jsonFd);

        j.objectBegin();

        createCaseGeneratorHeaderJson(j);
        createCaseGeneratorInputJson(j);
        createCaseGeneratorWeightsJson(j);
        createCaseGeneratorConvJson(j);
        createCaseGeneratorOutputJson(j);
        createCaseGeneratorODUPermutationJson(j);

        j.objectEnd();
    };

    createNumericsBenchJsonSpec();

    std::ifstream in_file(testConfigFile);
    std::stringstream in_file_buffer;
    in_file_buffer << in_file.rdbuf();

    nb::TestCaseJsonDescriptor desc(in_file_buffer.str());

    nb::InputLayer input = desc.getInputLayerList().front();
    ASSERT_EQ(nb::to_string(input.dtype), "uint8");

    nb::WeightLayer weight = desc.getWeightLayers().front();
    ASSERT_EQ(weight.qp.scale, std::vector<double>{0.01});
}

TEST(VPUX37XX_JSON_Parser, conv_test) {
    createAndRunConvTest();
}
