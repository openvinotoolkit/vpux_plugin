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

#include "tests_timeout.hpp"
#include "kmb_layers_tests.hpp"
#include "kmb_regression_target.hpp"

#include <gtest/gtest.h>
#include <regression_tests.hpp>
#include <inference_engine/precision_utils.h>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>

#include <ie_icnn_network_stats.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <ie_util_internal.hpp>
#include "low_precision_transformations/transformer.hpp"

#include <vpu_layers_tests.hpp>

#include <mutex>
#include <condition_variable>
#include <ie_layers.h>

#ifdef ENABLE_VPUAL

using namespace ::testing;
using namespace InferenceEngine;
using namespace Regression::Matchers;
using namespace InferenceEngine::details;
using namespace TestsTimeout;
using namespace KmbRegressionTarget;

struct TestingNetworkParameters : public CompilationParameter {
    TestingNetworkParameters() = default;
    TestingNetworkParameters(
            std::string name,
            std::string pathToNetwork,
            std::string pathToWeights,
            std::string pathToInput) : CompilationParameter(name, pathToNetwork, pathToWeights),
                    path_to_input(pathToInput)
    {
    };

    std::string path_to_input;
};

using VpuInferAndCompareTestParam = WithParamInterface<TestingNetworkParameters>;

class VpuInferAndCompareTests : public vpuLayersTests,
                                public VpuInferAndCompareTestParam {
public:
    using TestParam = VpuInferAndCompareTestParam;

    // Operations
    static std::string getTestCaseName(TestParamInfo <VpuInferAndCompareTestParam::ParamType> param);
};

std::string VpuInferAndCompareTests::getTestCaseName(
        TestParamInfo <VpuInferAndCompareTestParam::ParamType> param) {
    auto inputPath = (param.param).path_to_input;
    std::replace(inputPath.begin(), inputPath.end(), '/', '_');
    std::replace(inputPath.begin(), inputPath.end(), '-', '_');
    return (param.param).name + "_" + inputPath;
}

TEST_P(VpuInferAndCompareTests, DISABLED_NQA) {  // To be run in manual mode when device is available
    TestingNetworkParameters path_to_files = TestParam::GetParam();
    std::string irXmlPath = ModelsPath() + path_to_files.path_to_network;
    std::string weightsPath = ModelsPath() + path_to_files.path_to_weights;
    std::string inputPath = get_data_path() + path_to_files.path_to_input;

    CNNNetReader netReader;
    netReader.ReadNetwork(irXmlPath);
    netReader.ReadWeights(weightsPath);

    CNNNetwork network = netReader.getNetwork();

    InputsDataMap inputInfo = network.getInputsInfo();
    for (auto & item : inputInfo) {
        item.second->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    }

    Core ie;
    int batch = 1;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "kmb", {});

    Blob::Ptr input;
    Blob::Ptr result;
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    // TODO: infer part and input/output processing should be corrected
    // depending to actual inputs/outputs of testing network
    ASSERT_NO_THROW(input = inferRequest.GetBlob("data"));
    ASSERT_NO_THROW(result = inferRequest.GetBlob("prob"));

    std::shared_ptr<unsigned char> imageData;
    FormatReader::ReaderPtr pictureReader(inputPath.c_str());
    imageData = pictureReader->getData();
    std::vector<unsigned char> imageDataBatched;
    for(int i = 0; i != batch; i++) {
        std::copy(imageData.get(), imageData.get() + pictureReader->size(), std::back_inserter(imageDataBatched));
    }

    IE_SUPPRESS_DEPRECATED_START
    ConvertImageToInput(&imageDataBatched.front(), imageDataBatched.size(), *input.get());
    IE_SUPPRESS_DEPRECATED_END

    ASSERT_NO_THROW(inferRequest.Infer());

    auto out1 = result.get();
    for (int i=0; i != batch; i++) {
        // TODO: offsets and thresholds should be corrected depending on the actual testing network
        auto result_checked_value = out1->cbuffer().as<const float *>()[283 + i*out1->size() / batch];
        std::cout << result_checked_value << std::endl;
        EXPECT_NEAR(result_checked_value, 0.697f,  0.01) << "Value out of threshold for batch: " << i;
    }
}

std::vector<TestingNetworkParameters> vpuInferAndCompareTestsNQA = {
        TestingNetworkParameters{"ResNet_50_v1_tf_int8_dense",
                        "/KMB_models/NQA/ResNet-50-tf/resnet50-int8.xml",
                        "/KMB_models/NQA/ResNet-50-tf/resnet50-int8.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"ResNet_50_v1_tf_int8_sparse",
                        "/KMB_models/NQA/ResNet-50-tf/resnetv1-int8-sparse-v2-tf-0001.xml",
                        "/KMB_models/NQA/ResNet-50-tf/resnetv1-int8-sparse-v2-tf-0001.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"ResNet_50_v1_onnx_int8_dense",
                        "/KMB_models/NQA/ResNet-50-onnx/resnet50-v1-int8.xml",
                        "/KMB_models/NQA/ResNet-50-onnxf/resnet50-v1-int8.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"ResNet_50_v1_onnx_int8_sparse",
                        "/KMB_models/NQA/ResNet-50-onnx/resnet50-int8-sparse-v2.xml",
                        "/KMB_models/NQA/ResNet-50-onnx/resnet50-int8-sparse-v2.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"GoogLeNet_v1_tf_int8",
                        "/KMB_models/NQA/GoogLeNet-v1-tf/inceptionv1-int8-tf-0001.xml",
                        "/KMB_models/NQA/GoogLeNet-v1-tf/inceptionv1-int8-tf-0001.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"MobileNet_v2_tf_int8_dense",
                        "/KMB_models/NQA/MoblieNet-v2-tf/mobilenetv2-int8.xml",
                        "/KMB_models/NQA/Moblie Net-v2-tf/mobilenetv2-int8.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"MobileNet_v2_tf_int8_sparse",
                        "/KMB_models/NQA/MoblieNet-v2-onnx/mobilenetv2-int8-sparse-v2-tf-0001.xml",
                        "/KMB_models/NQA/MoblieNet-v2-onnx/mobilenetv2-int8-sparse-v2-tf-0001.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"MobileNet_v2_onnx_int8_dense",
                        "/KMB_models/NQA/MoblieNet-v2-onnx/mobilenetv2-int8.xml",
                        "/KMB_models/NQA/MoblieNet-v2-onnx/mobilenetv2-int8.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"MobileNet_v2_onnx_int8_sparse",
                        "/KMB_models/NQA/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2.xml",
                        "/KMB_models/NQA/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"YoloTiny_v2_tf_int8",
                        "/KMB_models/NQA/YoloTiny-v2-tf/tiny_yolo_v2.xml",
                        "/KMB_models/NQA/YoloTiny-v2-tf/tiny_yolo_v2.bin",
                        "/416x416/person.bmp"},
        TestingNetworkParameters{"Inceptionv3_onnx_int8",
                        "/KMB_models/NQA/inceptionv3-onnx/inceptionv3-int8.xml",
                        "/KMB_models/NQA/inceptionv3-onnx/inceptionv3-int8.bin",
                        "/299x299/lassy_googlenet_big.bmp"},
        TestingNetworkParameters{"SqueezeNetv1.1_onnx_int8",
                        "/KMB_models/NQA/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8.xml",
                        "/KMB_models/NQA/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"SSD512_onnx_int8",
                        "/KMB_models/INT8/SSD512-int8-onnx-0001/SSD512-int8-onnx-0001.xml",
                        "/KMB_models/INT8/SSD512-int8-onnx-0001/SSD512-int8-onnx-0001.bin",
                        "/512x512/dog_croped512.bmp"},
        TestingNetworkParameters{"Yolo_v2_tf_int8",
                        "/KMB_models/NQA/yolo_v2_tf/yolo_v2.xml",
                        "/KMB_models/NQA/yolo_v2_tf/yolo_v2.bin",
                        "/416x416/person.bmp"},

        //  IRs from DL_benchmarking_models
        TestingNetworkParameters{"FasterRcnnResnet101_tf_fp16",
                        "/KMB_models/FP16/faster_rcnn_resnet101_coco/tf/tf_frozen/FP16/1/dldt/faster_rcnn_resnet101_coco.xml",
                        "/KMB_models/FP16/faster_rcnn_resnet101_coco/tf/tf_frozen/FP16/1/dldt/faster_rcnn_resnet101_coco.bin",
                        // TODO: Add and use 600x600 picture the input size of network
                        "/512x512/dog_croped512.bmp"},
        TestingNetworkParameters{"ICNet_caffe_fp16",
                        "/KMB_models/FP16/icnet/caffe/caffe/FP16/1/dldt/icnet.xml",
                        "/KMB_models/FP16/icnet/caffe/caffe/FP16/1/dldt/icnet.bin",
                        "/1024x2048/frankfurt_001016.bmp"},

        // u8_asymmetric models
        TestingNetworkParameters{"YoloTiny_v2_u8_asymmetric",
                        "/KMB_models/NQA/u8_asymmetric/YoloTiny-v2/tiny_yolo_v2_asymmetric.xml",
                        "/KMB_models/NQA/u8_asymmetric/YoloTiny-v2/tiny_yolo_v2_asymmetric.bin",
                        "/416x416/person.bmp"},
        TestingNetworkParameters{"YoloTiny_v2_u8_asymmetric_cut",
                        "/KMB_models/NQA/u8_asymmetric/YoloTiny-v2/tiny_yolo_v2_asymmetric_cut.xml",
                        "/KMB_models/NQA/u8_asymmetric/YoloTiny-v2/tiny_yolo_v2_asymmetric.bin",
                        "/416x416/person.bmp"},
        TestingNetworkParameters{"MobileNet_v2_u8_asymmetric",
                        "/KMB_models/NQA/u8_asymmetric/MobileNet-v2/mobilenetv2_asymmetric.xml",
                        "/KMB_models/NQA/u8_asymmetric/MobileNet-v2/mobilenetv2_asymmetric.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"MobileNet_v2_u8_asymmetric_cut",
                        "/KMB_models/NQA/u8_asymmetric/MobileNet-v2/mobilenetv2_asymmetric_cut.xml",
                        "/KMB_models/NQA/u8_asymmetric/MobileNet-v2/mobilenetv2_asymmetric.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"Resnet_50_u8_asymmetric",
                        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric.xml",
                        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"Resnet_50_u8_asymmetric_cut",
                        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric_cut.xml",
                        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"Resnet_50_u8_asymmetric_cutfc",
                        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric_cutfc.xml",
                        "/KMB_models/NQA/u8_asymmetric/ResNet-50/resnet-50-pytorch_asymmetric_cutfc.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"Inception_v1_tf_asymmetric",
                        "/KMB_models/NQA/u8_asymmetric/inception-v1_tf/inceptionv1.xml",
                        "/KMB_models/NQA/u8_asymmetric/inception-v1_tf/inceptionv1.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"SqueezeNet1_1_asymmetric",
                        "/KMB_models/NQA/u8_asymmetric/squeezenet1_1/squeezenet1_1.xml",
                        "/KMB_models/NQA/u8_asymmetric/squeezenet1_1/squeezenet1_1.bin",
                        "/224x224/cat3.bmp"},
        TestingNetworkParameters{"Yolo_v2_asymmetric",
                        "/KMB_models/NQA/u8_asymmetric/Yolo-v2/yolo_v2_asymmetric.xml",
                        "/KMB_models/NQA/u8_asymmetric/Yolo-v2/yolo_v2_asymmetric.bin",
                        "/416x416/person.bmp"},
};

INSTANTIATE_TEST_CASE_P(InferAndCompareTestsNQA, VpuInferAndCompareTests,
    ::testing::ValuesIn(vpuInferAndCompareTestsNQA),
    VpuInferAndCompareTests::getTestCaseName
);

#endif
