//
// Copyright 2017-2018 Intel Corporation.
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

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <tuple>
#include <set>

#include <ie_layers_internal.hpp>

#include <vpu/stub_stage.hpp>

namespace vpu {

void FrontEnd::parseDeconvolution(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    //
    // Extract parameters
    //

    auto deconvLayer = std::dynamic_pointer_cast<ie::DeconvolutionLayer>(layer);
    IE_ASSERT(deconvLayer != nullptr);

    int kernelSizeX = deconvLayer->_kernel_x;
    int kernelSizeY = deconvLayer->_kernel_y;

    int kernelStrideX = deconvLayer->_stride_x;
    int kernelStrideY = deconvLayer->_stride_y;

    auto paddings = getPaddings(*deconvLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : 0;

    int dilationX = deconvLayer->_dilation_x;
    int dilationY = deconvLayer->_dilation_y;

    int groupSize = deconvLayer->_group;

    //
    // Create const datas
    //

    auto input = inputs[0];
    auto output = outputs[0];

    if ((groupSize == 0) ||
        (groupSize > input->desc().dim(Dim::C)) ||
        (input->desc().dim(Dim::C) % groupSize != 0) ||
        (groupSize > output->desc().dim(Dim::C)) ||
        (output->desc().dim(Dim::C) % groupSize != 0)) {
        VPU_THROW_EXCEPTION << "DeconvolutionLayer has invalid group value";
    }

    Data weights, biases;
    std::tie(weights, biases) = getWeightsAndBiases(model, layer);

    IE_ASSERT(weights->desc().totalDimSize() >=
              kernelSizeX * kernelSizeY * (input->desc().dim(Dim::C) / groupSize) * output->desc().dim(Dim::C));
    weights = model->duplicateData(
        weights,
        "@deconv",
        DataDesc({
            kernelSizeX,
            kernelSizeY,
            input->desc().dim(Dim::C) / groupSize,
            output->desc().dim(Dim::C)}));

    if (biases->usage() != DataUsage::Fake) {
        IE_ASSERT(biases->desc().totalDimSize() >= output->desc().dim(Dim::C));
        biases = model->duplicateData(
            biases,
            "@deconv",
            DataDesc({output->desc().dim(Dim::C)}));
    }

    //
    // Create stub stage
    //

    auto stage = model->addNewStage<StubStage>(
        layer->name,
        StageType::StubDeconv,
        layer,
        {input, weights, biases},
        {output});

    stage->attrs().set<int>("kernelSizeX", kernelSizeX);
    stage->attrs().set<int>("kernelSizeY", kernelSizeY);

    stage->attrs().set<int>("kernelStrideX", kernelStrideX);
    stage->attrs().set<int>("kernelStrideY", kernelStrideY);

    stage->attrs().set<int>("padLeft", padLeft);
    stage->attrs().set<int>("padRight", padRight);
    stage->attrs().set<int>("padTop", padTop);
    stage->attrs().set<int>("padBottom", padBottom);

    stage->attrs().set<int>("dilationX", dilationX);
    stage->attrs().set<int>("dilationY", dilationY);

    stage->attrs().set<int>("groupSize", groupSize);
    stage->attrs().set<bool>("tryHW", true);
}

}  // namespace vpu
