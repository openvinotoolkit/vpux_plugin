//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/network_description.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <algorithm>

using namespace vpux;

using intel_npu::IONodeDescriptor;
using intel_npu::IONodeDescriptorMap;
using intel_npu::NetworkMetadata;

namespace {
const uint32_t DIM_N = 0, DIM_C = 1, DIM_H = 2, DIM_W = 3, DIM_D = 4;

static const std::unordered_map<DimsOrder, std::vector<float>> orderMapping = {
        {DimsOrder::NCHW, {DIM_N, DIM_C, DIM_H, DIM_W}},
        {DimsOrder::NHWC, {DIM_N, DIM_H, DIM_W, DIM_C}},
        {DimsOrder::NCDHW, {DIM_N, DIM_C, DIM_D, DIM_H, DIM_W}},
        {DimsOrder::NDHWC, {DIM_N, DIM_D, DIM_H, DIM_W, DIM_C}},
        {DimsOrder::C, {DIM_C}},
        {DimsOrder::CHW, {DIM_C, DIM_H, DIM_W}},
        {DimsOrder::NC, {DIM_N, DIM_C}},
};

ov::element::Type_t extractPrecisionFromDType(MVCNN::DType dtype) {
    static const EnumMap<MVCNN::DType, ov::element::Type_t> dataTypeMapping = {
            {MVCNN::DType_FP32, ov::element::Type_t::f32}, {MVCNN::DType_FP16, ov::element::Type_t::f16},
            {MVCNN::DType_U64, ov::element::Type_t::u64},  {MVCNN::DType_U32, ov::element::Type_t::u32},
            {MVCNN::DType_U16, ov::element::Type_t::u16},  {MVCNN::DType_U8, ov::element::Type_t::u8},
            {MVCNN::DType_I64, ov::element::Type_t::i64},  {MVCNN::DType_I32, ov::element::Type_t::i32},
            {MVCNN::DType_I16, ov::element::Type_t::i16},  {MVCNN::DType_I8, ov::element::Type_t::i8},
            {MVCNN::DType_BIN, ov::element::Type_t::u1},
    };

    return dataTypeMapping.at(dtype);
}

/**
 * @brief Deserializes a tensor descriptor and stores it using OpenVINO specific structures.
 * @param tensor The object whose values shall be converted.
 * @return The same tensor but in a structure which makes use of the OpenVINO API.
 */
IONodeDescriptor deserializeTensor(const MVCNN::TensorReference* tensor) {
    const std::string& tensorName = tensor->name()->str();
    const auto* dims = tensor->dimensions();

    std::vector<size_t> dataDims;
    dataDims.resize(dims->size());
    std::copy_n(dims->data(), dims->size(), dataDims.data());

    const ov::Shape& shape = ov::Shape(dataDims);
    const ov::element::Type_t precision = extractPrecisionFromDType(tensor->data_dtype());

    return {tensorName, "", {}, precision, shape, shape};
}

using TensorReferenceVector = flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>>;

/**
 * @brief Extracts the state descriptors in a format interpretable by the OpenVINO API.
 * @param tensors The vector from which the descriptors shall be extracted.
 * @param stateDescriptors The structure in which the result shall be stored.
 * @param stateNames The names shall be stored here in the order in which the state descriptors are found.
 */
void deserializeStateTensors(const TensorReferenceVector* tensors, IONodeDescriptorMap& stateDescriptors,
                             std::vector<std::string>& stateNames) {
    if (tensors == nullptr) {
        return;
    }

    for (auto ind : irange(tensors->size())) {
        const auto* tensor = tensors->Get(ind);
        std::string tensorName = tensor->name()->str();

        // The inputs and outputs of the state nodes share the same metadata, thus we'll consider only the the inputs
        // here
        if (isStateInputName(tensorName)) {
            tensorName = tensorName.substr(READVALUE_PREFIX.length());
            stateNames.push_back(tensorName);
            stateDescriptors[tensorName] = deserializeTensor(tensor);
            stateDescriptors[tensorName].outputTensorNames = {tensorName};
            stateDescriptors[tensorName].legacyName = tensorName;
        }
    }
}

/**
 * @brief Extracts the order in which the inputs/outputs are found within the compiled model.
 * @details The order is a requirement only when running inferences using the IMD backend.
 * @param mvcnnOVNode The vector from which the order of inputs/outputs shall be extracted.
 * @param tensors The vector from which the order of states shall be extracted
 * @return A mapping between the names of the inputs/outputs and their order indices.
 */
std::unordered_map<std::string, size_t> extractIOOrder(
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::OVNode>>* mvcnnOVNode,
        const TensorReferenceVector* tensors) {
    std::unordered_map<std::string, size_t> order;

    uint32_t tensorIndex = 0;

    for (auto ind : irange(mvcnnOVNode->size())) {
        if (const auto* node = mvcnnOVNode->Get(ind)) {
            order.emplace(node->friendly_name()->str(), tensorIndex);
            ++tensorIndex;
        }
    }

    for (auto ind : irange(tensors->size())) {
        const MVCNN::TensorReference* tensor = tensors->Get(ind);
        const std::string tensorName = tensor->name()->str();
        if (isStateInputName(tensorName) || isStateOutputName(tensorName) || isShapeTensorName(tensorName)) {
            order.emplace(tensorName, tensorIndex);
            ++tensorIndex;
        }
    }

    return order;
}

/**
 * @brief Extracts the profiling output descriptors in a format interpretable by the OpenVINO API.
 * @param tensors The vector from which the descriptors shall be extracted.
 * @return The profiling output descriptors
 */
IONodeDescriptorMap deserializeProfilingOutputTensors(const TensorReferenceVector* tensors) {
    IONodeDescriptorMap tensorDescriptors;

    if (tensors == nullptr) {
        return tensorDescriptors;
    }

    for (auto ind : irange(tensors->size())) {
        const auto* tensor = tensors->Get(ind);
        const std::string& tensorName = tensor->name()->str();

        tensorDescriptors[tensorName] = deserializeTensor(tensor);
    }

    return tensorDescriptors;
}

const EnumMap<MVCNN::OVNodeType, ov::element::Type_t> mapElementTypeOV = {
        {MVCNN::OVNodeType::OVNodeType_UNDEFINED, ov::element::Type_t::undefined},
        {MVCNN::OVNodeType::OVNodeType_DYNAMIC, ov::element::Type_t::dynamic},
        {MVCNN::OVNodeType::OVNodeType_BOOLEAN, ov::element::Type_t::boolean},
        {MVCNN::OVNodeType::OVNodeType_BF16, ov::element::Type_t::bf16},
        {MVCNN::OVNodeType::OVNodeType_F16, ov::element::Type_t::f16},
        {MVCNN::OVNodeType::OVNodeType_F32, ov::element::Type_t::f32},
        {MVCNN::OVNodeType::OVNodeType_F64, ov::element::Type_t::f64},
        {MVCNN::OVNodeType::OVNodeType_I4, ov::element::Type_t::i4},
        {MVCNN::OVNodeType::OVNodeType_I8, ov::element::Type_t::i8},
        {MVCNN::OVNodeType::OVNodeType_I16, ov::element::Type_t::i16},
        {MVCNN::OVNodeType::OVNodeType_I32, ov::element::Type_t::i32},
        {MVCNN::OVNodeType::OVNodeType_I64, ov::element::Type_t::i64},
        {MVCNN::OVNodeType::OVNodeType_U1, ov::element::Type_t::u1},
        {MVCNN::OVNodeType::OVNodeType_U4, ov::element::Type_t::u4},
        {MVCNN::OVNodeType::OVNodeType_U8, ov::element::Type_t::u8},
        {MVCNN::OVNodeType::OVNodeType_U16, ov::element::Type_t::u16},
        {MVCNN::OVNodeType::OVNodeType_U32, ov::element::Type_t::u32},
        {MVCNN::OVNodeType::OVNodeType_U64, ov::element::Type_t::u64},
};

/**
 * @brief Extracts the parameter/result (i.e. input/output) node descriptors in a format interpretable by the OpenVINO
 * API.
 * @param mvcnnOVNode The vector from which the descriptors shall be extracted.
 * @param nodes The structure in which the result shall be stored.
 * @param names The names shall be stored here in the order in which the node descriptors are found.
 */
void deserializeIONodes(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::OVNode>>* mvcnnOVNode,
                        IONodeDescriptorMap& nodes, std::vector<std::string>& names) {
    // Check for the existence of a field in a blob. In older versions of the blob, this field may not exist
    if (mvcnnOVNode == nullptr) {
        return;
    }

    for (auto ind : irange(mvcnnOVNode->size())) {
        if (const auto* node = mvcnnOVNode->Get(ind)) {
            const auto nodeType = mapElementTypeOV.at(node->type());
            const auto currentNodeName = node->friendly_name()->str();

            const auto nodeShape = [&node]() {
                ov::Shape retShape;
                for (auto iter = node->shape()->cbegin(); iter != node->shape()->cend(); ++iter) {
                    retShape.push_back(*iter);
                }
                return retShape;
            }();

            const auto outputTensorNames = [&node]() {
                std::unordered_set<std::string> retTensorNames;
                for (auto iter = node->tensor_names()->cbegin(); iter != node->tensor_names()->cend(); ++iter) {
                    retTensorNames.insert(iter->str());
                }
                return retTensorNames;
            }();

            const auto legacyName = node->input_name()->str();

            names.push_back(currentNodeName);
            nodes[currentNodeName] = {legacyName, currentNodeName, outputTensorNames, nodeType, nodeShape, nodeShape};
        }
    }
}
}  // namespace

NetworkMetadata vpux::VPUIP::getNetworkMetadata(const std::vector<uint8_t>& blob) {
    NetworkMetadata network;

    OV_ITT_TASK_CHAIN(NETWORK_DESCRIPTION, itt::domains::VPUXPlugin, "NetworkDescription::NetworkDescription",
                      "VerifyGraphFileBuffer");
    VPUX_THROW_UNLESS(!blob.empty(), "Got NULL pointer");

    flatbuffers::Verifier verifier(blob.data(), blob.size(),
                                   /*max_depth=*/128, /*max_tables=*/UINT32_MAX);
    VPUX_THROW_UNLESS(MVCNN::VerifyGraphFileBuffer(verifier), "Got invalid VPUIP blob - network description");

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "GetGraphFile");
    const auto* graphFile = MVCNN::GetGraphFile(blob.data());
    const auto* header = graphFile->header();

    if (header->identifier() != nullptr) {
        network.name = header->identifier()->str();
    }

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializeIONodes");
    VPUX_THROW_UNLESS(header->ov_parameters() != nullptr && header->ov_results() != nullptr,
                      "VPUIP blob does not the parameter and result nodes");

    deserializeIONodes(header->ov_parameters(), network.parameters, network.inputNames);
    deserializeIONodes(header->ov_results(), network.results, network.outputNames);

    VPUX_THROW_UNLESS(!network.results.empty(), "VPUIP blob does not contain outputs");

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializeStatesAndProfilingOutputs");
    deserializeStateTensors(header->net_input(), network.states, network.stateNames);
    network.profilingOutputs = deserializeProfilingOutputTensors(header->profiling_output());

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "extractInputsOutputsOrder");
    network.inputOrder = extractIOOrder(header->ov_parameters(), header->net_input());
    network.outputOrder = extractIOOrder(header->ov_results(), header->net_output());

    return network;
}
