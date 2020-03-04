//
// Copyright 2020 Intel Corporation.
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

#pragma once

#include <cpp/ie_cnn_net_reader.h>
#include <ie_api.h>
#include "test_model_path.hpp"

struct modelBlobInfo {
    std::string graphName, graphPath, inputPath, outputPath;
};

struct modelTensors {
    InferenceEngine::TensorDesc inputTensor, outputTensor;
};

namespace PrecompiledResNet_Helper {
    // Old version, u8 output
    // TODO Remote after adding fp16 output support to HDDL2 Plugin
    static const modelBlobInfo resnet50_dpu =
            {
                    .graphName = "resnet-50-dpu",
                    .graphPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50-dpu/resnet-50-dpu.blob",
                    .inputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50-dpu/input.bin",
                    .outputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50-dpu/output.bin"
            };

    static const modelTensors resnet50_dpu_tensors =
            {
                    .inputTensor = InferenceEngine::TensorDesc(InferenceEngine::Precision::U8,
                                                               {1, 3, 224, 224},
                                                               InferenceEngine::Layout::NCHW
                    ),
                    .outputTensor = InferenceEngine::TensorDesc(InferenceEngine::Precision::U8,
                                                                {1, 1000, 1, 1},
                                                                InferenceEngine::Layout::NCHW
                    )
            };

    // Actual version, fp16 output
    static const modelBlobInfo resnet50 =
            {
                    .graphName = "resnet-50-dpu",
                    .graphPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob",
                    .inputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input.bin",
                    .outputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output.bin",
            };

    static const modelTensors resnet50_tensors =
            {
                    .inputTensor = InferenceEngine::TensorDesc(InferenceEngine::Precision::U8,
                                                               {1, 3, 224, 224},
                                                               InferenceEngine::Layout::NCHW
                    ),
                    .outputTensor = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP16,
                                                                {1, 1000, 1, 1},
                                                                InferenceEngine::Layout::NCHW
                    )
            };
};
