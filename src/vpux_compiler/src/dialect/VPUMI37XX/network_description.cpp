//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <openvino/core/partial_shape.hpp>

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include <vpux_headers/serial_metadata.hpp>

#include "vpux/compiler/dialect/VPUMI37XX/network_description.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
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

ov::element::Type_t extractPrecisionFromDType(elf::DType dtype) {
    static const EnumMap<elf::DType, ov::element::Type_t> dataTypeMapping = {
            {elf::DType::DType_FP32, ov::element::Type_t::f32}, {elf::DType::DType_FP16, ov::element::Type_t::f16},
            {elf::DType::DType_U64, ov::element::Type_t::u64},  {elf::DType::DType_U32, ov::element::Type_t::u32},
            {elf::DType::DType_U16, ov::element::Type_t::u16},  {elf::DType::DType_U8, ov::element::Type_t::u8},
            {elf::DType::DType_I64, ov::element::Type_t::i64},  {elf::DType::DType_I32, ov::element::Type_t::i32},
            {elf::DType::DType_I16, ov::element::Type_t::i16},  {elf::DType::DType_I8, ov::element::Type_t::i8},
            {elf::DType::DType_BIN, ov::element::Type_t::u1},
    };

    return dataTypeMapping.at(dtype);
}

/**
 * @brief Deserializez a tensor descriptor and stores it using OpenVINO specific structures.
 * @param tensor The object whose values shall be converted.
 * @return The same tensor but in a structure which makes use of the OpenVINO API.
 */
IONodeDescriptor deserializeTensor(const elf::TensorRef* tensor) {
    const std::string& tensorName = tensor->name;
    const auto* dims = tensor->dimensions;

    std::vector<size_t> dataDims;
    dataDims.resize(tensor->dimensions_size);
    std::copy_n(dims, tensor->dimensions_size, dataDims.data());

    const ov::Shape& shape = ov::Shape(dataDims);
    const ov::element::Type_t precision = extractPrecisionFromDType(tensor->data_type);

    return {tensorName, "", {}, precision, shape, shape};
}

using TensorReferenceVector = flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>>;

/**
 * @brief Extracts the state descriptors in a format interpretable by the OpenVINO API.
 * @param tensors The array from which the descriptors shall be extracted.
 * @param count The number of tensors in the array
 * @param stateDescriptors The structure in which the result shall be stored.
 * @param stateNames The names shall be stored here in the order in which the state descriptors are found.
 */
void deserializeStateTensors(elf::TensorRef* tensors, uint32_t count, IONodeDescriptorMap& stateDescriptors,
                             std::vector<std::string>& stateNames) {
    for (uint32_t i = 0; i < count; ++i) {
        auto tensor = tensors + i;
        std::string tensorName = tensor->name;

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
 * @brief Extracts the shape descriptors in a format interpretable by the OpenVINO API.
 * @param inTensors An array from which the descriptors of input tensors shall be extracted.
 * @param outTensors An array from which the descriptors of output tensors shall be extracted.
 * @param inCount, outCount The number of tensors in the arrays
 * @param shapeDescriptors The structure in which the result shall be stored.
 * @param shapeNames The names shall be stored here in the order in which the state descriptors are found.
 */
void deserializeShapeTensors(elf::TensorRef* inTensors, uint32_t inCount, elf::TensorRef* outTensors, uint32_t outCount,
                             IONodeDescriptorMap& shapeDescriptors, std::vector<std::string>& shapeNames) {
    for (uint32_t i = 0; i < inCount; ++i) {
        auto tensor = inTensors + i;
        std::string tensorName = tensor->name;

        if (isShapeTensorName(tensorName)) {
            tensorName = tensorName.substr(SHAPE_TENSOR_PREFIX.length());
            shapeNames.push_back(tensorName);
            shapeDescriptors[tensorName] = deserializeTensor(tensor);
            shapeDescriptors[tensorName].outputTensorNames = {tensorName};
            shapeDescriptors[tensorName].legacyName = tensorName;
        }
    }
    for (uint32_t i = 0; i < outCount; ++i) {
        auto tensor = outTensors + i;
        std::string tensorName = tensor->name;

        if (isShapeTensorName(tensorName)) {
            tensorName = tensorName.substr(SHAPE_TENSOR_PREFIX.length());
            shapeNames.push_back(tensorName);
            shapeDescriptors[tensorName] = deserializeTensor(tensor);
            shapeDescriptors[tensorName].outputTensorNames = {tensorName};
            shapeDescriptors[tensorName].legacyName = tensorName;
        }
    }
}

/**
 * @brief Extracts the order in which the inputs/outputs are found within the compiled model.
 * @details The order is a requirement only when running inferences using the IMD backend.
 * @param OVNode The array from which the order of inputs/outputs shall be extracted.
 * @param count The number of inputs/outputs in the array
 * @param tensors The array from which the order of states and shape tensors shall be extracted
 * @param count The number of tensors in the array
 * @return A mapping between the names of the inputs/outputs and their order indices.
 */
std::unordered_map<std::string, size_t> extractIOOrder(const elf::OVNode* OVNode, uint32_t countNodes,
                                                       const elf::TensorRef* tensors, uint32_t countTensors) {
    std::unordered_map<std::string, size_t> order;

    uint32_t tensorIndex = 0;

    for (uint32_t ind = 0; ind < countNodes; ++ind) {
        if (auto* node = OVNode + ind) {
            order.emplace(node->friendly_name, tensorIndex);
            ++tensorIndex;
        }
    }

    for (uint32_t ind = 0; ind < countTensors; ++ind) {
        const elf::TensorRef* tensor = tensors + ind;
        if (isStateInputName(tensor->name) || isStateOutputName(tensor->name) || isShapeTensorName(tensor->name)) {
            order.emplace(tensor->name, tensorIndex);
            ++tensorIndex;
        }
    }

    return order;
}

/**
 * @brief Extracts the profiling output descriptors in a format interpretable by the OpenVINO API.
 * @param tensors The array from which the descriptors shall be extracted.
 * @param count The number of tensors in the array
 * @return The profiling output descriptors
 */
IONodeDescriptorMap deserializeProfilingOutputTensors(elf::TensorRef* tensors, uint32_t count) {
    IONodeDescriptorMap tensorDescriptors;

    for (uint32_t i = 0; i < count; ++i) {
        auto tensor = tensors + i;
        const std::string& tensorName = tensor->name;

        tensorDescriptors[tensorName] = deserializeTensor(tensor);
    }

    return tensorDescriptors;
}

const EnumMap<elf::OVNodeType, ov::element::Type_t> mapElementTypeOV = {
        {elf::OVNodeType::OVNodeType_UNDEFINED, ov::element::Type_t::undefined},
        {elf::OVNodeType::OVNodeType_DYNAMIC, ov::element::Type_t::dynamic},
        {elf::OVNodeType::OVNodeType_BOOLEAN, ov::element::Type_t::boolean},
        {elf::OVNodeType::OVNodeType_BF16, ov::element::Type_t::bf16},
        {elf::OVNodeType::OVNodeType_F16, ov::element::Type_t::f16},
        {elf::OVNodeType::OVNodeType_F32, ov::element::Type_t::f32},
        {elf::OVNodeType::OVNodeType_F64, ov::element::Type_t::f64},
        {elf::OVNodeType::OVNodeType_I4, ov::element::Type_t::i4},
        {elf::OVNodeType::OVNodeType_I8, ov::element::Type_t::i8},
        {elf::OVNodeType::OVNodeType_I16, ov::element::Type_t::i16},
        {elf::OVNodeType::OVNodeType_I32, ov::element::Type_t::i32},
        {elf::OVNodeType::OVNodeType_I64, ov::element::Type_t::i64},
        {elf::OVNodeType::OVNodeType_U1, ov::element::Type_t::u1},
        {elf::OVNodeType::OVNodeType_U4, ov::element::Type_t::u4},
        {elf::OVNodeType::OVNodeType_U8, ov::element::Type_t::u8},
        {elf::OVNodeType::OVNodeType_U16, ov::element::Type_t::u16},
        {elf::OVNodeType::OVNodeType_U32, ov::element::Type_t::u32},
        {elf::OVNodeType::OVNodeType_U64, ov::element::Type_t::u64},
};

/**
 * @brief Extracts the parameter/result (i.e. input/output) node descriptors in a format interpretable by the OpenVINO
 * API.
 * @param OVNode The array from which the descriptors shall be extracted.
 * @param count The number of tensors in the array
 * @param nodes The structure in which the result shall be stored.
 * @param names The names shall be stored here in the order in which the node descriptors are found.
 */
void deserializeIONodes(elf::OVNode* OVNode, uint32_t count, IONodeDescriptorMap& nodes,
                        std::vector<std::string>& names) {
    // Check for the existence of a field in a blob. In older versions of the blob, this field may not exist
    if (count == 0) {
        return;
    }

    for (uint32_t ind = 0; ind < count; ind++) {
        if (auto* node = OVNode + ind) {
            const auto nodeType = mapElementTypeOV.at(node->type);
            const auto currentNodeName = std::string(node->friendly_name);

            const auto nodeShape = [&node]() {
                const auto dynamicDim = std::numeric_limits<uint64_t>::max();
                ov::PartialShape retShape;
                for (size_t i = 0; i < node->shape_size; i++) {
                    if (node->shape[i] != dynamicDim) {
                        retShape.push_back(checked_cast<int64_t>(node->shape[i]));
                    } else {
                        retShape.push_back(-1);
                    }
                }
                return retShape;
            }();

            const auto outputTensorNames = [&node]() {
                std::unordered_set<std::string> retTensorNames;
                for (size_t i = 0; i < node->tensor_names_count; i++) {
                    retTensorNames.insert(std::string(node->tensor_names[i]));
                }
                return retTensorNames;
            }();

            const auto legacyName = std::string(node->input_name);

            names.push_back(currentNodeName);
            nodes[currentNodeName] = {legacyName, currentNodeName, outputTensorNames, nodeType, nodeShape, nodeShape};
        }
    }
}

}  // namespace

NetworkMetadata vpux::VPUMI37XX::getNetworkMetadata(const std::vector<uint8_t>& blob) {
    NetworkMetadata network;

    OV_ITT_TASK_CHAIN(NETWORK_DESCRIPTION, itt::domains::VPUXPlugin, "NetworkDescription::NetworkDescription",
                      "elfReader");
    VPUX_THROW_UNLESS(!blob.empty(), "Got NULL pointer");

    auto accessor = elf::ElfDDRAccessManager(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);

    std::shared_ptr<elf::NetworkMetadata> metadata;

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "getSection&getHeader");
    for (size_t secIndex = 0; secIndex < reader.getSectionsNum(); secIndex++) {
        const auto& section = reader.getSection(secIndex);

        const auto secHeader = section.getHeader();
        if (secHeader->sh_type == static_cast<elf::Elf_Word>(vpux::ELFNPU37XX::SectionTypeAttr::VPU_SHT_NETDESC)) {
            metadata = elf::MetadataSerialization::deserialize(section.getData<uint8_t>(), secHeader->sh_size);
            break;
        }
    }

    VPUX_THROW_UNLESS(metadata != nullptr, "METADATA NOT FOUND IN ELF");
    network.name = metadata->mIdentification.blob_name;

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializeIONodes");

    deserializeIONodes(metadata->mOVParameters.data(), metadata->mOVParameters.size(), network.parameters,
                       network.inputNames);
    deserializeIONodes(metadata->mOVResults.data(), metadata->mOVResults.size(), network.results, network.outputNames);

    VPUX_THROW_UNLESS(!network.results.empty(), "Metadata structure does not contain info on outputs");

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializeStatesAndProfilingOutputs");

    deserializeStateTensors(metadata->mNetInputs.data(), metadata->mNetInputs.size(), network.states,
                            network.stateNames);
    deserializeShapeTensors(metadata->mNetInputs.data(), metadata->mNetInputs.size(), metadata->mNetOutputs.data(),
                            metadata->mNetOutputs.size(), network.shapes, network.shapeNames);
    network.profilingOutputs =
            deserializeProfilingOutputTensors(metadata->mProfilingOutputs.data(), metadata->mProfilingOutputs.size());

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "extractInputsOutputsOrder");

    network.inputOrder = extractIOOrder(metadata->mOVParameters.data(), metadata->mOVParameters.size(),
                                        metadata->mNetInputs.data(), metadata->mNetInputs.size());
    network.outputOrder = extractIOOrder(metadata->mOVResults.data(), metadata->mOVResults.size(),
                                         metadata->mNetOutputs.data(), metadata->mNetOutputs.size());

    network.numStreams = metadata->mResourceRequirements.nn_slice_count_;

    return network;
}
