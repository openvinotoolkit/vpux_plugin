//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

#include "vpux/compiler/utils/strings.hpp"

#include "vpux/compiler/core/profiling_metadata.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include "schema/profiling_generated.h"

using namespace vpux;

void copy_str(char* dst, const std::string& src, bool throwOnErr = false) {
    VPUX_THROW_WHEN(throwOnErr && (src.size() >= elf::MAX_STRING_LEN), "Target char array is too small");
    auto str_len = src.size() < elf::MAX_STRING_LEN ? src.size() : elf::MAX_STRING_LEN - 1;

    memcpy(dst, src.data(), str_len);
    dst[str_len] = '\0';
}

elf::DType ELFNPU37XX::createDType(mlir::Type type) {
    if (type.isF64()) {
        return elf::DType::DType_FP64;
    } else if (type.isF32()) {
        return elf::DType::DType_FP32;
    } else if (type.isF16()) {
        return elf::DType::DType_FP16;
    } else if (type.isBF16()) {
        return elf::DType::DType_BFP16;
    } else if (type.isFloat8E5M2()) {
        return elf::DType::DType_FP8;
    } else if (type.isFloat8E4M3FN()) {
        return elf::DType::DType_HF8;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return elf::DType::DType_I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return elf::DType::DType_I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return elf::DType::DType_I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return elf::DType::DType_I8;
    } else if (type.isSignedInteger(4)) {
        return elf::DType::DType_I4;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint64_t))) {
        return elf::DType::DType_U64;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint32_t))) {
        return elf::DType::DType_U32;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint16_t))) {
        return elf::DType::DType_U16;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return elf::DType::DType_U8;
    } else if (type.isInteger(4)) {
        return elf::DType::DType_U4;
    } else if (type.isInteger(2)) {
        return elf::DType::DType_I2;
    } else if (type.isInteger(1)) {
        return elf::DType::DType_BIN;
    } else if (type.isa<mlir::quant::QuantizedType>()) {
        return createDType(type.cast<mlir::quant::QuantizedType>().getStorageType());
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

elf::TensorRef ELFNPU37XX::createTensorRef(vpux::NDTypeInterface type, StringRef name) {
    elf::TensorRef out{};

    copy_str(out.name, name.str());

    // dtype
    out.data_type = ELFNPU37XX::createDType(type.getElementType());

    // dims
    const auto shape = type.getShape();
    out.dimensions_size = checked_cast<uint32_t>(shape.size());

    bool isStatic = true;
    for (const auto& sh_pair : shape | indexed) {
        const auto ind = checked_cast<uint32_t>(sh_pair.index());
        auto dim = sh_pair.value();
        if (dim > 0) {
            out.dimensions[ind] = checked_cast<uint32_t>(dim);
        } else if (dim == -1) {
            out.dimensions[ind] = std::numeric_limits<uint32_t>::max();
            isStatic = false;
        } else {
            VPUX_THROW_WHEN(shape.empty(),
                            "Unexpected dim value. It must be a positive number or -1 to represent a dynamic dim");
        }
    }

    // strides
    // TODO: we can resolve strides as we have bounds information in tensors: E#88898
    // this check can be removed after that
    if (isStatic) {
        auto strides = type.getStrides();
        out.strides_size = checked_cast<uint32_t>(strides.size());

        Strides temp;
        temp.push_back(type.getElemTypeSize());
        temp.append(strides.begin(), strides.end());

        for (auto iterator : temp | indexed) {
            auto val = iterator.value();
            auto index = iterator.index();

            out.strides[index] = checked_cast<uint64_t>(val.count());
        }
    }

    // dimsOrder
    out.order = type.getDimsOrder().code();

    return out;
}

elf::TensorRef ELFNPU37XX::createTensorRef(mlir::Value val, StringRef name) {
    return createTensorRef(val.getType().cast<vpux::NDTypeInterface>(), name);
}

std::unique_ptr<elf::NetworkMetadata> ELFNPU37XX::constructMetadata(
        mlir::ModuleOp module, IE::CNNNetworkOp netOp, mlir::func::FuncOp netFunc,
        const std::vector<std::shared_ptr<const ov::Node>>& parameters,
        const std::vector<std::shared_ptr<const ov::Node>>& results, vpux::Logger log) {
    log.setName("ELF - Construct Metadata");

    auto inputsInfo = netOp.getInputsDataInfo();
    auto outputsInfo = netOp.getOutputsDataInfo();
    auto profilingOutputsInfo = netOp.getProfilingOutputsDataInfo();

    // We are returning a unique_ptr to the heap allocated metadata due to its large size.
    // Returning the metadata struct by value can cause a stack overflow on certain systems.
    auto metadataPtr = std::make_unique<elf::NetworkMetadata>();
    auto& metadata = *metadataPtr.get();

    // Copy arch_name and throw if it doesn't fit into the buffer.
    // arch_name must not be truncated to ensure proper operation of the ELF loader.
    copy_str(metadata.mIdentification.arch_name, VPU::stringifyArchKind(VPU::getArch(module)).str(), true);
    // Copy blob_name and throw if it doesn't fit into the buffer.
    // blob_name must not be truncated to ensure proper operation of the driver.
    copy_str(metadata.mIdentification.blob_name, module.getName().value_or("network").str(), true);

    metadata.mNetInputs.resize(inputsInfo.size());
    metadata.mInTensorDescriptors.resize(inputsInfo.size());

    metadata.mNetOutputs.resize(outputsInfo.size());
    metadata.mOutTensorDescriptors.resize(outputsInfo.size());

    metadata.mProfilingOutputs.resize(profilingOutputsInfo.size());

    const auto architecture = VPU::getArch(module);
    if (architecture == VPU::ArchKind::NPU40XX) {
        auto ioBindings = VPUASM::IOBindingsOp::getFromModule(module);
        auto inputDeclarations =
                to_small_vector(ioBindings.getInputDeclarations().front().getOps<VPUASM::DeclareBufferOp>());

        for (const auto& p : inputsInfo | indexed) {
            const auto index = checked_cast<uint32_t>(p.index());
            auto userInfo = p.value();
            auto inputDeclaration = inputDeclarations[index];

            auto declaredInputType = inputDeclaration.getBufferType().getMemref().cast<vpux::NDTypeInterface>();
            const auto userType = userInfo.getUserType().cast<vpux::NDTypeInterface>();

            metadata.mNetInputs[index] = createTensorRef(declaredInputType, userInfo.getName());
            metadata.mInTensorDescriptors[index] = createTensorRef(userType, userInfo.getName());
        }

        auto outDeclarations =
                to_small_vector(ioBindings.getOutputDeclarations().front().getOps<VPUASM::DeclareBufferOp>());
        for (const auto& p : outputsInfo | indexed) {
            const auto index = p.index();
            auto userInfo = p.value();
            auto outDeclaration = outDeclarations[index];

            auto declaredOutType = outDeclaration.getBufferType().getMemref().cast<vpux::NDTypeInterface>();
            const auto userType = userInfo.getUserType().cast<vpux::NDTypeInterface>();
            metadata.mNetOutputs[index] = createTensorRef(declaredOutType, userInfo.getName());
            metadata.mOutTensorDescriptors[index] = createTensorRef(userType, userInfo.getName());
        }

        // profiling
        auto profilingDeclarations =
                to_small_vector(ioBindings.getProfilingBuffDeclarations().front().getOps<VPUASM::DeclareBufferOp>());
        for (const auto& p : profilingOutputsInfo | indexed) {
            const auto index = p.index();
            auto profilingDeclaration = profilingDeclarations[index];

            auto declaredPorfileBuffType =
                    profilingDeclaration.getBufferType().getMemref().cast<vpux::NDTypeInterface>();

            metadata.mProfilingOutputs[index] = createTensorRef(declaredPorfileBuffType, p.value().getName());
        }
    } else {
        // input
        for (const auto& p : inputsInfo | indexed) {
            const auto index = checked_cast<uint32_t>(p.index());
            auto userInfo = p.value();
            const auto val = netFunc.getArgument(index);

            const auto userType = userInfo.getUserType().cast<vpux::NDTypeInterface>();

            metadata.mNetInputs[index] = createTensorRef(val, userInfo.getName());
            metadata.mInTensorDescriptors[index] = createTensorRef(userType, userInfo.getName());
        }

        // output
        for (const auto& p : outputsInfo | indexed) {
            const auto index = p.index();
            const auto funcArgIndex = inputsInfo.size() + index;

            auto userInfo = p.value();
            const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgIndex));

            const auto userType = userInfo.getUserType().cast<vpux::NDTypeInterface>();

            metadata.mNetOutputs[index] = createTensorRef(val, userInfo.getName());
            metadata.mOutTensorDescriptors[index] = createTensorRef(userType, userInfo.getName());
        }

        // profiling
        for (const auto& p : profilingOutputsInfo | indexed) {
            const auto index = p.index();
            const auto funcArgInd = inputsInfo.size() + outputsInfo.size() + index;

            const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

            metadata.mProfilingOutputs[index] = createTensorRef(val, p.value().getName());
        }
    }

    // ov parameters
    metadata.mOVParameters.resize(parameters.size());
    for (const auto& node : parameters | indexed) {
        VPUX_THROW_WHEN(node.value() == nullptr, "Null OV node");
        auto node_val = node.value();
        auto index = node.index();

        elf::OVNode tmp_node{};
        tmp_node.type = ELFNPU37XX::mapElementType.at(node_val->get_element_type());

        // name strings
        copy_str(tmp_node.friendly_name, node_val->get_friendly_name());
        const auto tmpInputName = ov::op::util::create_ie_output_name(node_val->output(0));
        copy_str(tmp_node.input_name, tmpInputName);

        const auto tmpTensorNames = node_val->get_output_tensor(0).get_names();
        auto tensorNamesSize = tmpTensorNames.size();
        if (tensorNamesSize > elf::MAX_METADATA_IO) {
            log.warning("OV Parameter {0} has {1} tensor names. Trimming to the maximum limit of {2}", index,
                        tensorNamesSize, elf::MAX_METADATA_IO);
            tensorNamesSize = elf::MAX_METADATA_IO;
        }

        tmp_node.tensor_names_count = tensorNamesSize;
        for (auto tensor_name : tmpTensorNames | indexed) {
            if (tensor_name.index() >= tensorNamesSize) {
                break;
            }
            copy_str(tmp_node.tensor_names[tensor_name.index()], tensor_name.value());
        }

        // shape
        auto shape = node_val->get_output_partial_shape(0);
        tmp_node.shape_size = checked_cast<uint32_t>(shape.size());

        for (const auto& sh_iterator : shape | indexed) {
            auto value = sh_iterator.value();
            tmp_node.shape[sh_iterator.index()] =
                    value.is_static() ? value.get_length() : std::numeric_limits<uint64_t>::max();
        }

        metadata.mOVParameters[index] = tmp_node;
    }

    // ov results
    metadata.mOVResults.resize(results.size());
    for (const auto& node : results | indexed) {
        VPUX_THROW_WHEN(node.value() == nullptr, "Null OV node");
        auto node_val = node.value();
        auto index = node.index();

        elf::OVNode tmp_node{};
        tmp_node.type = ELFNPU37XX::mapElementType.at(node_val->get_element_type());

        // name strings
        copy_str(tmp_node.friendly_name, node_val->get_friendly_name());

        const auto tmpInputName = ov::op::util::create_ie_output_name(node_val->input_value(0));
        copy_str(tmp_node.input_name, tmpInputName);

        const auto tmpTensorNames = node_val->get_output_tensor(0).get_names();
        auto tensorNamesSize = tmpTensorNames.size();
        if (tensorNamesSize > elf::MAX_METADATA_IO) {
            log.warning("OV Result {0} has {1} tensor names. Trimming to the maximum limit of {2}", index,
                        tensorNamesSize, elf::MAX_METADATA_IO);
            tensorNamesSize = elf::MAX_METADATA_IO;
        }

        tmp_node.tensor_names_count = tensorNamesSize;
        for (auto tensor_name : tmpTensorNames | indexed) {
            if (tensor_name.index() >= tensorNamesSize) {
                break;
            }
            copy_str(tmp_node.tensor_names[tensor_name.index()], tensor_name.value());
        }

        auto shape = node_val->get_output_partial_shape(0);
        tmp_node.shape_size = checked_cast<uint32_t>(shape.size());

        for (const auto& sh_iterator : shape | indexed) {
            auto value = sh_iterator.value();
            tmp_node.shape[sh_iterator.index()] =
                    value.is_static() ? value.get_length() : std::numeric_limits<uint64_t>::max();
        }

        metadata.mOVResults[index] = tmp_node;
    }

    return metadataPtr;
}

using BarrierMap = DenseMap<mlir::Value, uint32_t>;

BarrierMap getBarriers(mlir::func::FuncOp funcOp) {
    BarrierMap barriersIds;
    for (VPUMI37XX::ConfigureBarrierOp barrierOp : funcOp.getOps<VPUMI37XX::ConfigureBarrierOp>()) {
        auto val = barrierOp.getBarrier();
        VPUX_THROW_UNLESS(barriersIds.count(val) == 0, "Value {0} was already serialized", val);
        barriersIds.insert({val, checked_cast<uint32_t>(barriersIds.size())});
    }
    return barriersIds;
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> getOpBarriers(const BarrierMap& virtBarriers,
                                                                      mlir::ValueRange waitBarriers,
                                                                      mlir::ValueRange updateBarriers) {
    const auto extractBarriersIDs = [&virtBarriers](mlir::ValueRange barriers) -> std::vector<uint32_t> {
        std::vector<uint32_t> ids;
        ids.reserve(barriers.size());
        for (const auto bar : barriers) {
            const auto it = virtBarriers.find(bar);
            VPUX_THROW_UNLESS(it != virtBarriers.end(), "Value {0} wasn't serialized yet", bar);
            ids.push_back(it->second);
        }
        return ids;
    };

    std::vector<uint32_t> waitIds = extractBarriersIDs(waitBarriers);
    std::vector<uint32_t> updateIds = extractBarriersIDs(updateBarriers);

    return std::make_pair(waitIds, updateIds);
}
