//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vector>

#include <random>
#include "common_test_utils/data_utils.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "shared_tests_instances/vpu_ov2_layer_test.hpp"
#include "single_op_tests/non_max_suppression.hpp"

namespace ov {
namespace test {

class NmsLayerTestCommon : public NmsLayerTest, virtual public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::generate_inputs(targetInputStaticShapes);
        size_t it = 0;
        for (const auto& input : VpuOv2LayerTest::inputs) {
            if ((it == 0) || (it == 1)) {
                auto tensor = input.second;
                uint32_t range = 1;
                uint32_t resolution = 1000;
                if (it == 0) {  // default GenerateInput parameters
                    range = 10;
                    resolution = 1;
                }
                if (tensor.get_element_type() == ov::element::f32) {
                    fillDataRandomFloatWithFp16Type(tensor, range, 0, resolution);
                } else {
                    ov::test::utils::fill_random_unique_sequence<ov::fundamental_type_for<ov::element::f16>>(
                            tensor.data<ov::fundamental_type_for<ov::element::f16>>(), tensor.get_byte_size(), range, 0,
                            resolution);
                }
                if (it == 0) {
                    sortCorner(tensor);
                }
            } else {
                break;
            }
            it++;
        }
    }

protected:
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
    void run() override {
    }

    void run(const std::string_view platform) {
        VpuOv2LayerTest::run(platform);
    }

private:
    void fillDataRandomFloatWithFp16Type(const ov::Tensor& tensor, const uint32_t range, int32_t start_from,
                                         const int32_t k) {
        std::default_random_engine random(1);
        std::uniform_int_distribution<int32_t> distribution(k * start_from, k * (start_from + range));
        auto* rawData = tensor.data<float>();
        for (size_t i = 0; i < tensor.get_byte_size() / sizeof(float); i++) {
            auto value = static_cast<float>(distribution(random));
            value /= static_cast<float>(k);
            ov::float16 fp16Val = ov::float16(value);
            rawData[i] = static_cast<float>(fp16Val);
        }
    }
    void sortCorner(const ov::Tensor& tensor) {
        auto* rawBlobDataPtr = tensor.data<float>();
        for (size_t i = 0; i < tensor.get_byte_size() / sizeof(float); i += 4) {
            float y1 = rawBlobDataPtr[i + 0];
            float x1 = rawBlobDataPtr[i + 1];
            float y2 = rawBlobDataPtr[i + 2];
            float x2 = rawBlobDataPtr[i + 3];

            float ymin = std::min(y1, y2);
            float ymax = std::max(y1, y2);
            float xmin = std::min(x1, x2);
            float xmax = std::max(x1, x2);

            rawBlobDataPtr[i + 0] = ymin;
            rawBlobDataPtr[i + 1] = xmin;
            rawBlobDataPtr[i + 2] = ymax;
            rawBlobDataPtr[i + 3] = xmax;
        }
    }

protected:
    void SetUp() override {
        InputShapeParams inShapeParams;
        InputTypes inputTypes;
        int maxOutBoxesPerClass;
        float iouThr, scoreThr, softNmsSigma;
        op::v5::NonMaxSuppression::BoxEncodingType boxEncoding;
        op::v9::NonMaxSuppression::BoxEncodingType boxEncoding_v9;
        bool sortResDescend;
        element::Type outType;
        std::tie(inShapeParams, inputTypes, maxOutBoxesPerClass, iouThr, scoreThr, softNmsSigma, boxEncoding,
                 sortResDescend, outType, VpuOv2LayerTest::targetDevice) = this->GetParam();

        boxEncoding_v9 = boxEncoding == op::v5::NonMaxSuppression::BoxEncodingType::CENTER
                                 ? op::v9::NonMaxSuppression::BoxEncodingType::CENTER
                                 : op::v9::NonMaxSuppression::BoxEncodingType::CORNER;

        size_t numBatches, numBoxes, numClasses;
        std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

        ov::element::Type paramsType, maxBoxType, thrType;
        std::tie(paramsType, maxBoxType, thrType) = inputTypes;

        const std::vector<size_t> boxesShape{numBatches, numBoxes, 4}, scoresShape{numBatches, numClasses, numBoxes};
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(paramsType, ov::Shape(boxesShape)),
                                   std::make_shared<ov::op::v0::Parameter>(paramsType, ov::Shape(scoresShape))};

        auto maxOutBoxesPerClassNode = std::make_shared<ov::op::v0::Constant>(
                maxBoxType, ov::Shape{}, std::vector<int32_t>{static_cast<int32_t>(maxOutBoxesPerClass)});
        auto iouThrNode = std::make_shared<ov::op::v0::Constant>(thrType, ov::Shape{}, std::vector<float>{iouThr});
        auto scoreThrNode = std::make_shared<ov::op::v0::Constant>(thrType, ov::Shape{}, std::vector<float>{scoreThr});
        auto softNmsSigmaNode =
                std::make_shared<ov::op::v0::Constant>(thrType, ov::Shape{}, std::vector<float>{softNmsSigma});

        auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(params[0], params[1], maxOutBoxesPerClassNode,
                                                                   iouThrNode, scoreThrNode, softNmsSigmaNode,
                                                                   boxEncoding_v9, sortResDescend, outType);
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(nms, params, "NMS");
    }
};

TEST_P(NmsLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(NmsLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;
namespace {

const std::vector<ov::test::InputShapeParams> inShapeParams = {
        ov::test::InputShapeParams{1, 80, 1},   // standard params usage 90% of conformance tests
        ov::test::InputShapeParams{1, 40, 20},  // 1 usage style
        ov::test::InputShapeParams{3, 30, 18},  // for check remain posibility
};

const std::vector<int32_t> maxOutBoxPerClass = {5, 15};
const std::vector<float> iouThreshold = {0.3f, 0.7f};
const std::vector<float> scoreThreshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f, 0.5f};
const std::vector<ov::op::v5::NonMaxSuppression::BoxEncodingType> encodType = {
        ov::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
        ov::op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
};
const std::vector<bool> sortResDesc = {false, true};
const std::vector<ov::element::Type> outType = {ov::element::i32};
std::vector<ov::element::Type> paramsType = {
        ov::element::f32,
};
std::vector<ov::element::Type> maxBoxType = {
        ov::element::i32,
};
std::vector<ov::element::Type> thrType = {
        ov::element::f16,
};

const auto nmsParams = ::testing::Combine(
        ::testing::ValuesIn(inShapeParams),
        ::testing::Combine(::testing::ValuesIn(paramsType), ::testing::ValuesIn(maxBoxType),
                           ::testing::ValuesIn(thrType)),
        ::testing::ValuesIn(maxOutBoxPerClass), ::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold),
        ::testing::ValuesIn(sigmaThreshold), ::testing::ValuesIn(encodType), ::testing::ValuesIn(sortResDesc),
        ::testing::ValuesIn(outType), ::testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_NmsLayerTest, NmsLayerTestCommon, nmsParams,
                        NmsLayerTestCommon::getTestCaseName);

const std::vector<ov::test::InputShapeParams> inShapeParamsSmoke = {ov::test::InputShapeParams{2, 9, 12}};
const std::vector<int32_t> maxOutBoxPerClassSmoke = {5};
const std::vector<float> iouThresholdSmoke = {0.3f};
const std::vector<float> scoreThresholdSmoke = {0.3f};
const std::vector<float> sigmaThresholdSmoke = {0.0f, 0.5f};
const std::vector<ov::op::v5::NonMaxSuppression::BoxEncodingType> encodTypeSmoke = {
        ov::op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
const auto nmsParamsSmoke =
        testing::Combine(testing::ValuesIn(inShapeParamsSmoke),
                         ::testing::Combine(::testing::ValuesIn(paramsType), ::testing::ValuesIn(maxBoxType),
                                            ::testing::ValuesIn(thrType)),
                         ::testing::ValuesIn(maxOutBoxPerClassSmoke), ::testing::ValuesIn(iouThresholdSmoke),
                         ::testing::ValuesIn(scoreThresholdSmoke), ::testing::ValuesIn(sigmaThresholdSmoke),
                         ::testing::ValuesIn(encodTypeSmoke), ::testing::ValuesIn(sortResDesc),
                         ::testing::ValuesIn(outType), ::testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_NmsLayerTest, NmsLayerTestCommon, nmsParamsSmoke,
                         NmsLayerTestCommon::getTestCaseName);
}  // namespace
