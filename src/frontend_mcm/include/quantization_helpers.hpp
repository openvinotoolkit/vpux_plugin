//
// Copyright 2019-2020 Intel Corporation.
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
#include <ie_layers_internal.hpp>
#include <low_precision_transformations/network_helper.hpp>
#include <low_precision_transformations/quantization_details.hpp>
#include <string>
#include <vector>

#ifdef ENABLE_MCM_COMPILER
#include <mcm/tensor/quantization_params.hpp>

namespace vpu {

namespace QuantizationHelpers {

bool isPostOp(const InferenceEngine::CNNLayerPtr& layer);
std::vector<float> getBlobValue(const InferenceEngine::CNNLayerPtr& constantLayer);
bool isWeightableLayerQuantized(const InferenceEngine::CNNLayerPtr& weightableLayer);
bool isRealQuantizeLayer(const InferenceEngine::CNNLayerPtr& layer);

mv::QuantizationParams calculateOutputScalesAndZeroPoint(
    const InferenceEngine::CNNLayerPtr& fakeQuantizeLayer, bool mergeInOne = false);

void fillQuntizationActivationParams(
    const InferenceEngine::CNNLayerPtr& quantizedLayer, mv::QuantizationParams& outputQuantParams);

// for symmetric case only, using mcm logic
int64_t calculateZeroPoint(float high, float low, int levels, InferenceEngine::Precision precision);

InferenceEngine::Blob::Ptr calculateQuntizationWeights(
    const InferenceEngine::CNNLayerPtr& weightableLayer, mv::QuantizationParams& weightsQuantParams);

mv::QuantizationParams fillQuantizeParamsForU8orI8weights(
    const InferenceEngine::CNNLayerPtr& weightableLayer, int levels, InferenceEngine::Precision precision);

std::vector<int64_t> quantizeBiases(const std::vector<double>& activationScales,
    const std::vector<double>& weightsScales, const InferenceEngine::Blob::Ptr biasBlob,
    mv::QuantizationParams& outputQuantParam);

}  // namespace QuantizationHelpers
}  // namespace vpu

#endif
