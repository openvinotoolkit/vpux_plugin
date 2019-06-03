//
// Copyright 2018 Intel Corporation.
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

#include <frontend_mcm.hpp>

#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <set>
#include <vector>

#include <vpu/compile_env.hpp>

namespace vpu {

namespace KmbPlugin {

namespace {

typedef void (FrontEndMcm::*parser_t)(
        const mv::OpModel& modelMcm,
        const ie::CNNLayerPtr& layer,
        const McmNodeVector& inputs);

ie::details::caseless_map<std::string, parser_t> g_mcm_parsers = {
    {"Convolution",        &FrontEndMcm::parseConvolution},
    {"Pooling",            &FrontEndMcm::parsePooling},
    {"ReLU",               &FrontEndMcm::parseReLU},
    {"Clamp",              &FrontEndMcm::parseClamp},
    {"FullyConnected",     &FrontEndMcm::parseFullyConnected},
    {"SoftMax",            &FrontEndMcm::parseSoftMax},
    {"GRN",                &FrontEndMcm::parseGRN},
    {"MVN",                &FrontEndMcm::parseMVN},
    {"Norm",               &FrontEndMcm::parseNorm},
    {"Concat",             &FrontEndMcm::parseConcat},
    {"Eltwise",            &FrontEndMcm::parseEltwise},
    {"Split",              &FrontEndMcm::parseSplit},
    {"Sigmoid",            &FrontEndMcm::parseSigmoid},
    {"TanH",               &FrontEndMcm::parseTanH},
    {"PReLU",              &FrontEndMcm::parsePReLU},
    {"Bias",               &FrontEndMcm::parseBias},
    // Caffe Slice is transformed to Split by IE
    {"Slice",              &FrontEndMcm::parseSplit},
    {"BatchNormalization", &FrontEndMcm::parseBatchNorm},
    {"ScaleShift",         &FrontEndMcm::parseScale},
    {"Deconvolution",      &FrontEndMcm::parseDeconvolution},
    {"Power",              &FrontEndMcm::parsePower},
    {"Copy",               &FrontEndMcm::parseCopy},
    {"Reshape",            &FrontEndMcm::parseReshape},
    {"ELU",                &FrontEndMcm::parseELU},
    // Flatten is represented as Reshape in KMB model
    {"Flatten",            &FrontEndMcm::parseReshape},
    {"Crop",               &FrontEndMcm::parseCrop},
    {"Tile",               &FrontEndMcm::parseTile},
    {"Normalize",          &FrontEndMcm::parseNormalize},
    {"PriorBox",           &FrontEndMcm::parsePriorBox},
    {"PriorBoxClustered",  &FrontEndMcm::parsePriorBoxClustered},
    {"Permute",            &FrontEndMcm::parsePermute},
    {"DetectionOutput",    &FrontEndMcm::parseDetectionOutput},
    {"RegionYolo",         &FrontEndMcm::parseRegionYolo},
    {"ReorgYolo",          &FrontEndMcm::parseReorgYolo},
    {"CTCGreedyDecoder",   &FrontEndMcm::parseCTCDecoder},
    {"Proposal",           &FrontEndMcm::parseProposal},
    {"ROIPooling",         &FrontEndMcm::parseROIPooling},
    {"PSROIPooling",       &FrontEndMcm::parsePSROIPooling},
    {"Interp",             &FrontEndMcm::parseInterp},
    {"Custom",             &FrontEndMcm::parseCustom},
    {"MTCNN",              &FrontEndMcm::parseMTCNN},
    {"LSTMCell",           &FrontEndMcm::parseLSTMCell},
    {"Pad",                &FrontEndMcm::parsePad},
    {"Resample",           &FrontEndMcm::parseResample},
    {"ArgMax",             &FrontEndMcm::parseArgMax},
};

std::atomic<int> g_counter(0);

}  // namespace

void FrontEndMcm::buildInitialModel(const ie::ICNNNetwork& network) {
    const auto& env = CompileEnv::get();

    runCommonPasses(network, LayersOrder::DFS);

    for (const auto& layer : _ieNetworkParser.orderedLayers) {
        IE_ASSERT(layer != nullptr);

        env.log->debug("try to parse layer %s", layer->name);

        McmNodeVector inputs;
        getInputData(layer, inputs);
        auto it =
                (_customLayers.count(layer->type) > 0) ?
                    g_mcm_parsers.find("Custom") :
                    g_mcm_parsers.find(layer->type);
        if (it == g_mcm_parsers.end()) {
            VPU_THROW_EXCEPTION
                    << "Cannot convert layer \""
                    << layer->name
                    << "\" due to unsupported layer type \""
                    << layer->type
                    << "\"";
        }

        auto parser = it->second;
        IE_ASSERT(parser != nullptr);

        (this->*parser)(_modelMcm, layer, inputs);
    }

    parseOutputData();
}

std::set<std::string> FrontEndMcm::checkSupportedLayers(const ie::ICNNNetwork& network) {
    const auto& env = CompileEnv::get();

    runCommonPasses(network, LayersOrder::BFS);

    std::set<std::string> layerNames;

    for (const auto& layer : _ieNetworkParser.orderedLayers) {
        IE_ASSERT(layer != nullptr);

        env.log->debug("Try to parse layer %s", layer->name);

        McmNodeVector inputs;
        getInputData(layer, inputs);

        auto it =
                (_customLayers.count(layer->type) > 0) ?
                    g_mcm_parsers.find("Custom") :
                    g_mcm_parsers.find(layer->type);
        if (it != g_mcm_parsers.end()) {
            try {
                // If we can create and have not thrown exception, then layer is supported.
                auto parser = it->second;
                IE_ASSERT(parser != nullptr);

                (this->*parser)(_modelMcm, layer, inputs);

                layerNames.insert(layer->name);
            } catch (const ie::details::InferenceEngineException&) {
                // TODO: continue instead?
                break;
            }
        }
    }

    return layerNames;
}

ie::CNNNetwork FrontEndMcm::detectNetworkBatch(
        const ie::ICNNNetwork& origNetwork) {
    VPU_PROFILE(detectNetworkBatch);

    // Keep original network.
    IE_SUPPRESS_DEPRECATED_START
    return ie::CNNNetwork(const_cast<ie::ICNNNetwork*>(&origNetwork));
    IE_SUPPRESS_DEPRECATED_END
}

void FrontEndMcm::parseInputData() {
    VPU_PROFILE(parseInputData);

    const auto& env = CompileEnv::get();
    env.log->debug("Try to parse network input");

    //
    // Parse network inputs
    //

    IE_ASSERT(_ieNetworkParser.networkInputs.size() == 1);

    for (const auto& inputInfo : _ieNetworkParser.networkInputs) {
        auto netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        DataDesc kmbDesc(ieData->getTensorDesc());
        if (kmbDesc.numDims() >= 3) {
            if (env.config.hwOptimization || env.config.forceLayout == ComputeLayout::NCHW) {
                kmbDesc.moveDim(Dim::C, 2);
            } else {
                kmbDesc.moveDim(Dim::C, 0);
            }
        }
        mv::Shape inputShape = {static_cast<std::size_t>(kmbDesc.dim(Dim::W)),
                                static_cast<std::size_t>(kmbDesc.dim(Dim::H)),
                                static_cast<std::size_t>(kmbDesc.dim(Dim::C)),
                                static_cast<std::size_t>(kmbDesc.dim(Dim::N))};
        auto mvInput = _modelMcm.input(inputShape, mv::DType("Float16"), mv::Order("NCHW"), {{}, {}, {}, {}}, netInput->name());
        auto input = std::make_shared<McmNodeObject>(mvInput, kmbDesc);
        _nodes.push_back(input);
        bindData(input, ieData);
        env.log->debug("Network input '%s'(orig: '%s') parsed to mcmModel", mvInput->getName(), netInput->name());
    }

#if 0
    //
    // Parse constant data
    //

        auto kmbData = model->addInputData(ieData->getName(), kmbDesc);
        bindData(kmbData, ieData);

        auto kmbData = model->addConstData(
            ieData->getName(),
            kmbDesc,
            ieBlobContent(ieBlob));

        // User might ask to return the output from Const layer.
        if (auto kmbOutData = getVpuData(ieData)) {
            IE_ASSERT(kmbOutData->usage() == DataUsage::Output);

            _stageBuilder->addCopyStage(
                model,
                formatString("%s@return-const", kmbData->name()),
                nullptr,
                kmbData,
                kmbOutData);
        }

        bindData(kmbData, ieData);
#endif
}

void FrontEndMcm::parseOutputData() {
    VPU_PROFILE(parseOutputData);

    const auto& env = CompileEnv::get();
    env.log->debug("Try to parse network output");

    //
    // Parse network outputs
    //

    for (const auto& outputInfo : _ieNetworkParser.networkOutputs) {
        auto ieData = outputInfo.second;

        IE_ASSERT(ieData != nullptr);

        auto lastLayerOut = getMcmData(ieData);
        IE_ASSERT(lastLayerOut != nullptr);
        auto name = lastLayerOut->getMcmNode()->getName();

        auto mvOutput = _modelMcm.output(lastLayerOut->getMcmNode());
        _output = std::make_shared<McmNodeObject>(mvOutput, lastLayerOut->desc());
        _nodes.push_back(_output);
    }
}

void FrontEndMcm::runCommonPasses(
        const ie::ICNNNetwork& network,
        LayersOrder order) {
    _ieNetworkParser.clear();
    auto reshapedNetwork = detectNetworkBatch(network);  // , model);

    //
    // Get IE layers in topological order
    //

    if (order == LayersOrder::DFS) {
        _ieNetworkParser.parseNetworkDFS(reshapedNetwork);
    } else {
        _ieNetworkParser.parseNetworkBFS(reshapedNetwork);
    }

    //
    // Parse network inputs/outputs/const datas
    //

    parseInputData();
}

McmNode FrontEndMcm::getMcmData(const ie::DataPtr& ieData) {
    IE_ASSERT(ieData != nullptr);

    auto it = _ieToMcmMap.find(ieData);
    if (it == _ieToMcmMap.end()) {
        return nullptr;
    }

    return it->second;
}

void FrontEndMcm::bindData(const McmNode& data, const ie::DataPtr& ieData) {
    IE_ASSERT(_modelMcm.isValid(data->getMcmNode()));
    IE_ASSERT(_modelMcm.isValid(_modelMcm.getSourceOp(data->getMcmNode())));

    _ieToMcmMap[ieData] = data;
    data->setOrigData(ieData);
}

void FrontEndMcm::getInputData(
        const ie::CNNLayerPtr& layer,
        McmNodeVector& inputs) {
    IE_ASSERT(layer != nullptr);

    inputs.resize(layer->insData.size());
    for (size_t i = 0; i < layer->insData.size(); ++i) {
        auto layerInput = layer->insData[i].lock();
        IE_ASSERT(layerInput != nullptr);

        inputs[i] = getMcmData(layerInput);
        IE_ASSERT(inputs[i] != nullptr);
    }
}

}  // namespace KmbPlugin

}  // namespace vpu
