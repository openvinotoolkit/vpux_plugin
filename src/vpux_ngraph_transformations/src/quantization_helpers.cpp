//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/quantization_helpers.hpp"
#include <algorithm>
#include <numeric>
#include <openvino/core/node.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/reference/autobroadcast_binop.hpp>
#include <stack>
#include <vector>
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/type/float16.hpp"

using namespace vpux;

int64_t calculateZeroPoint(float low, float high, int levels, const ov::element::Type& elemType) {
    OPENVINO_ASSERT((low <= 0.f) && (high >= 0.f) && (low != high));
    OPENVINO_ASSERT(levels <= 256);

    int zeroPoint = 0;

    if (elemType == ov::element::u8) {
        float x = -static_cast<float>(levels - 1) * low / (high - low);
        zeroPoint = static_cast<int>(std::round(x));
    } else if (elemType == ov::element::i8) {
        float x = -static_cast<float>(levels - 1) * ((high + low) * 0.5f) / (high - low);
        zeroPoint = static_cast<int>(std::round(x));
    } else {
        OPENVINO_THROW("Unsupported element type ", elemType);
    }

    return zeroPoint;
}

std::vector<int64_t> calculateZeroPoints(const std::vector<double>& low, const std::vector<double>& high, int levels,
                                         const ov::element::Type& elemType) {
    OPENVINO_ASSERT(high.size() == low.size());

    std::vector<int64_t> out(high.size());

    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = calculateZeroPoint(checked_cast<float>(low[i]), checked_cast<float>(high[i]), levels, elemType);
    }
    return out;
}

double calculateScale(float low, float high, int levels) {
    OPENVINO_ASSERT(low != high);
    OPENVINO_ASSERT(levels <= 256);

    return static_cast<double>((high - low) / static_cast<float>(levels - 1));
}

std::vector<double> calculateScales(const std::vector<double>& low, const std::vector<double>& high, int levels) {
    OPENVINO_ASSERT(high.size() == low.size());

    std::vector<double> out(high.size());

    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = calculateScale(checked_cast<float>(low[i]), checked_cast<float>(high[i]), levels);
    }

    return out;
}

double clamp(double val, double low, double high) {
    OPENVINO_ASSERT(low <= high);
    return std::min(high, std::max(low, val));
}

void align_zp(float& min, float& max, const int max_levels) {
    auto zp = calculateZeroPoint(min, max, max_levels, ov::element::u8);
    auto scale = calculateScale(min, max, max_levels);
    min = static_cast<float>((0.0 - zp) * scale);
    max = static_cast<float>((max_levels - 1.0 - zp) * scale);
}

bool is_fq_agnostic(const std::shared_ptr<ov::Node>& node) {
    return (std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(node) != nullptr ||
            std::dynamic_pointer_cast<ov::op::v1::MaxPool>(node) != nullptr ||
            std::dynamic_pointer_cast<ov::op::v0::ReorgYolo>(node) != nullptr ||
            // [Track number: E#21478]
            // (std::dynamic_pointer_cast<ov::op::v1::Reshape>(node) != nullptr &&
            //  std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(node->output(0).get_target_inputs().begin()->get_node()->shared_from_this())
            //  == nullptr) ||
            std::dynamic_pointer_cast<ov::op::v1::Transpose>(node) != nullptr ||
            std::dynamic_pointer_cast<ov::op::v0::Squeeze>(node) != nullptr ||
            std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(node) != nullptr ||
            std::dynamic_pointer_cast<ov::op::v0::DepthToSpace>(node) != nullptr);
}

void replace_node_if_changed(const std::shared_ptr<ov::op::v0::Constant>& node, const std::vector<double>& data,
                             const std::string& name_postfix) {
    auto new_node = std::make_shared<ov::op::v0::Constant>(ov::element::f64, node->get_shape(), data.data());
    if (node->get_element_type() == ov::element::f32) {
        new_node = std::make_shared<ov::op::v0::Constant>(node->get_element_type(), node->get_shape(),
                                                          new_node->cast_vector<float>().data());
    }
    if (node->get_element_type() == ov::element::f16) {
        new_node = std::make_shared<ov::op::v0::Constant>(node->get_element_type(), node->get_shape(),
                                                          new_node->cast_vector<vpux::type::float16>().data());
    }
    if (node->get_element_type() == ov::element::u8) {
        new_node = std::make_shared<ov::op::v0::Constant>(node->get_element_type(), node->get_shape(),
                                                          new_node->cast_vector<unsigned char>().data());
    }

    if (node->get_friendly_name().find(name_postfix) == std::string::npos)
        new_node->set_friendly_name(node->get_friendly_name() + name_postfix);
    else
        new_node->set_friendly_name(node->get_friendly_name());

    bool changed = false;
    auto node_vals = node->cast_vector<double>();
    auto new_node_vals = new_node->cast_vector<double>();
    for (size_t i = 0; i < node_vals.size(); i++) {
        changed |= !isDoubleEqual(node_vals[i], new_node_vals[i]);
    }

    if (changed)
        ov::replace_node(node, new_node);
}

bool replace_node_if_changed(const std::shared_ptr<ov::op::v0::Constant>& node, const ov::element::Type_t type,
                             const std::vector<float>& data, const std::string& name_postfix) {
    auto new_node = std::make_shared<ov::op::v0::Constant>(
            type, data.size() == 1 ? ov::Shape(0) : ov::Shape({data.size(), 1, 1, 1}), data.data());
    if (node->get_friendly_name().find(name_postfix) == std::string::npos)
        new_node->set_friendly_name(node->get_friendly_name() + name_postfix);
    else
        new_node->set_friendly_name(node->get_friendly_name());

    bool changed = false;
    auto node_vals = node->cast_vector<double>();
    auto new_node_vals = new_node->cast_vector<double>();
    for (size_t i = 0; i < node_vals.size(); i++) {
        changed |= !isDoubleEqual(node_vals[i], new_node_vals[i]);
    }

    if (changed)
        ov::replace_node(node, new_node);

    return changed;
}

void replace_node_if_changed(const std::shared_ptr<ov::op::v0::Constant>& node, const ov::element::Type_t type,
                             const float data, const std::string& name_postfix) {
    auto new_node = std::make_shared<ov::op::v0::Constant>(type, ov::Shape(0), data);
    if (node->get_friendly_name().find(name_postfix) == std::string::npos)
        new_node->set_friendly_name(node->get_friendly_name() + name_postfix);
    else
        new_node->set_friendly_name(node->get_friendly_name());

    bool changed = false;
    auto node_vals = node->cast_vector<double>();
    auto new_node_vals = new_node->cast_vector<double>();
    for (size_t i = 0; i < node_vals.size(); i++) {
        changed |= !isDoubleEqual(node_vals[i], new_node_vals[i]);
    }

    if (changed)
        ov::replace_node(node, new_node);
}

int64_t quantizeVal(double val, double scale, int64_t zeroPoint, const ov::element::Type elemType) {
    int64_t qVal = 0;

    if (elemType == ov::element::u8) {
        qVal = static_cast<int64_t>(clamp(std::round(val / scale + zeroPoint), 0, 255));
    } else {
        OPENVINO_THROW("Unsupported element type ", elemType);
    }

    return qVal;
}

std::vector<int64_t> quantizeData(const ov::Shape& outShape, const ov::element::Type outElemType,
                                  const std::vector<double>& src, const ov::Shape& srcShape,
                                  const std::vector<double>& scales, const std::vector<int64_t>& zeroPoints,
                                  const ov::Shape& scalesShape) {
    const auto broadcast_spec = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY);

    std::vector<size_t> srcInds(ov::shape_size(srcShape));
    std::iota(srcInds.begin(), srcInds.end(), size_t(0));

    std::vector<size_t> scalesInds(ov::shape_size(scalesShape));
    std::iota(scalesInds.begin(), scalesInds.end(), size_t(0));

    std::vector<int64_t> out(ov::shape_size(outShape));

    ov::reference::autobroadcast_binop(srcInds.data(), scalesInds.data(), out.data(), srcShape, scalesShape,
                                       broadcast_spec, [&](size_t srcInd, size_t scaleInd) -> int64_t {
                                           const auto srcVal = src[srcInd];
                                           const auto scale = scales[scaleInd];
                                           const auto zeroPoint = zeroPoints[scaleInd];
                                           return quantizeVal(srcVal, scale, zeroPoint, outElemType);
                                       });

    return out;
}

std::vector<std::shared_ptr<ov::Node>> getParents(std::shared_ptr<ov::Node> node) {
    const auto input_values = node->input_values();
    std::vector<std::shared_ptr<ov::Node>> result;
    for (auto&& iv : input_values) {
        result.emplace_back(iv.get_node_shared_ptr());
    }
    return result;
}

std::vector<std::shared_ptr<ov::Node>> getAnyInputsFQ(std::shared_ptr<ov::Node> node) {
    std::vector<std::shared_ptr<ov::Node>> result;
    std::set<std::shared_ptr<ov::Node>> visited;
    std::stack<std::shared_ptr<ov::Node>> layers;

    layers.push(node);
    while (!layers.empty()) {
        auto input = layers.top();
        layers.pop();
        visited.insert(input);
        if ((dynamic_cast<ov::op::v0::FakeQuantize*>(input.get())) &&
            (nullptr == dynamic_cast<ov::op::v0::Constant*>(input->input_value(0).get_node()))) {
            result.push_back(input);
        } else {
            auto inputs = getParents(input);
            for (const auto& newInput : inputs) {
                if (!visited.count(newInput)) {
                    layers.push(newInput);
                }
            }
        }
    }
    return result;
}

std::vector<std::shared_ptr<ov::Node>> getInputsFQ(const std::shared_ptr<ov::Node>& node) {
    const auto parents = getParents(node);
    std::vector<std::shared_ptr<ov::Node>> result;
    std::vector<std::shared_ptr<ov::Node>> tmp;
    for (const auto& input : parents) {
        tmp = getAnyInputsFQ(input);
        if (tmp.size()) {
            std::move(tmp.begin(), tmp.end(), std::back_inserter(result));
            tmp.clear();
        } else
            return std::vector<std::shared_ptr<ov::Node>>();
    }
    return result;
}

bool all_fqs_have_same_io_params(std::set<std::shared_ptr<ov::Node>>& fqs) {
    for (auto fq_node : fqs) {
        auto fq_node1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
        auto fq_node2 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
        auto fq_node3 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(3).get_node_shared_ptr());
        auto fq_node4 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(4).get_node_shared_ptr());
        if (fq_node1 == nullptr || fq_node2 == nullptr || fq_node3 == nullptr || fq_node4 == nullptr)
            return false;

        auto fq_data1 = fq_node1->cast_vector<float>();
        auto fq_data2 = fq_node2->cast_vector<float>();
        auto fq_data3 = fq_node3->cast_vector<float>();
        auto fq_data4 = fq_node4->cast_vector<float>();

        auto fq_i_counts = fq_data1.size();

        if (fq_i_counts != fq_data2.size())
            OPENVINO_THROW("FQ ", fq_node->get_friendly_name(), " have different input low/high parameters count");
        auto fq_o_counts = fq_data3.size();
        if (fq_o_counts != fq_data4.size())
            OPENVINO_THROW("FQ ", fq_node->get_friendly_name(), " have different output low/high parameters count");

        if (fq_i_counts != fq_o_counts) {
            return false;
        }

        for (size_t i = 0; i < fq_o_counts; i++) {
            if (!isFloatEqual(fq_data1[i], fq_data3[i]) || !isFloatEqual(fq_data2[i], fq_data4[i])) {
                return false;
            }
        }
    }

    return true;
}

bool all_fqs_are_equal(std::vector<std::shared_ptr<ov::Node>>& fqs) {
    auto get_fq_param = [](int input_num, std::shared_ptr<ov::Node> fq_node) -> std::vector<float> {
        if (fq_node == nullptr) {
            OPENVINO_THROW("Unexpected nullptr value in fq_node");
        }

        auto fq_param =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(input_num).get_node_shared_ptr());
        if (fq_param == nullptr) {
            return std::vector<float>();
        }
        return fq_param->cast_vector<float>();
    };

    return std::all_of(fqs.begin() + 1, fqs.end(), [&](const std::shared_ptr<ov::Node> fq) {
        bool is_same = true;
        const int fq_constants_count = 4;
        for (int i = 1; i < 1 + fq_constants_count; ++i) {
            is_same &= get_fq_param(i, fq) == get_fq_param(i, fqs.front());
        }
        return is_same;
    });
}
