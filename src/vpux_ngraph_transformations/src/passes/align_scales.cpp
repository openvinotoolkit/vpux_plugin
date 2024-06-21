//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/align_scales.hpp"

#include <memory>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <openvino/op/ops.hpp>

#include "vpux/quantization_helpers.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using vpux::checked_cast;

namespace vpux {
namespace passes {

static bool node_is_add_or_concat(std::shared_ptr<ov::Node> node) {
    return (std::dynamic_pointer_cast<ov::op::v0::Concat>(node) != nullptr ||
            std::dynamic_pointer_cast<ov::op::v1::Add>(node) != nullptr);
}

static std::vector<std::shared_ptr<ov::Node>> gather_nodes_around(std::shared_ptr<ov::Node> node) {
    auto result = std::vector<std::shared_ptr<ov::Node>>();

    for (const auto& input : node->input_values()) {
        result.push_back(input.get_node()->shared_from_this());
    }
    for (const auto& node_output : node->outputs()) {
        for (auto consumer : node_output.get_target_inputs()) {
            result.push_back(consumer.get_node()->shared_from_this());
        }
    }

    return result;
}

static void gather_fqs(std::shared_ptr<ov::Node> node, std::set<std::shared_ptr<ov::Node>>& fqs_to_align) {
    for (const auto& input : node->input_values()) {
        const auto& input_node = input.get_node()->shared_from_this();
        if (std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(input_node) != nullptr) {
            if (fqs_to_align.find(input_node) == fqs_to_align.end()) {
                fqs_to_align.insert(input_node);
                auto nodes_around = gather_nodes_around(input_node);
                for (const auto& node_around : nodes_around) {
                    if (is_fq_agnostic(node_around)) {
                        gather_fqs(node_around, fqs_to_align);
                    }
                }
            }
        }
    }
    if (is_fq_agnostic(node)) {
        for (const auto& node_output : node->outputs()) {
            for (auto consumer : node_output.get_target_inputs()) {
                const auto& output_node = consumer.get_node()->shared_from_this();
                if (std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(output_node) != nullptr) {
                    if (fqs_to_align.find(output_node) == fqs_to_align.end()) {
                        fqs_to_align.insert(output_node);
                        auto nodes_around = gather_nodes_around(output_node);
                        for (const auto& node_around : nodes_around) {
                            if (is_fq_agnostic(node_around)) {
                                gather_fqs(node_around, fqs_to_align);
                            }
                        }
                    }
                }
            }
        }
    }
}

static bool no_concat_consumers_around_fqs(std::set<std::shared_ptr<ov::Node>>& fqs) {
    for (const auto& fq : fqs) {
        const auto& nodes_around = gather_nodes_around(fq);
        for (const auto& node_around : nodes_around) {
            if (std::dynamic_pointer_cast<ov::op::v0::Concat>(node_around) != nullptr) {
                return false;
            }
        }
    }

    return true;
}

static void find_min_max(std::set<std::shared_ptr<ov::Node>>& fqs, float& min, float& max, float& range,
                         int& max_levels) {
    for (auto fq_node : fqs) {
        auto fq = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(fq_node);
        OPENVINO_ASSERT(fq != nullptr);
        auto fq_node1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node1 != nullptr);
        auto fq_node2 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node2 != nullptr);
        auto fq_node3 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(3).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node3 != nullptr);
        auto fq_node4 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(4).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node4 != nullptr);

        auto fq_data1 = fq_node1->cast_vector<float>();
        auto fq_data2 = fq_node2->cast_vector<float>();
        auto fq_data3 = fq_node3->cast_vector<float>();
        auto fq_data4 = fq_node4->cast_vector<float>();

        if (max_levels < static_cast<int>(fq->get_levels())) {
            max_levels = static_cast<int>(fq->get_levels());
        }
        for (size_t c = 0; c < fq_data1.size(); c++) {
            if (min > fq_data1[c]) {
                min = fq_data1[c];
            }
            if (max < fq_data2[c]) {
                max = fq_data2[c];
            }
            if (range < fq_data2[c] - fq_data1[c]) {
                range = fq_data2[c] - fq_data1[c];
            }
        }
    }
}

static void broadcast_changes(const std::shared_ptr<ov::Node>& node);

static void align_fq(std::set<std::shared_ptr<ov::Node>>& fqs, const float min, const float max, const float range,
                     const int max_levels) {
    auto changed = std::vector<bool>(fqs.size());
    size_t i = 0;

    for (auto fq_node : fqs) {
        auto fq = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(fq_node);
        OPENVINO_ASSERT(fq != nullptr);
        auto fq_node1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node1 != nullptr);
        auto fq_node2 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node2 != nullptr);
        auto fq_node3 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(3).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node3 != nullptr);
        auto fq_node4 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(4).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node4 != nullptr);

        auto fq_data1 = fq_node1->cast_vector<float>();
        auto fq_data2 = fq_node2->cast_vector<float>();
        auto fq_data3 = fq_node3->cast_vector<float>();
        auto fq_data4 = fq_node4->cast_vector<float>();

        changed[i] = static_cast<int>(fq->get_levels()) != max_levels;
        fq->set_levels(max_levels);
        if (no_concat_consumers_around_fqs(fqs)) {
            // Can align for Eltwises only
            for (size_t c = 0; c < fq_data1.size(); c++) {
                auto zp =
                        checked_cast<double>(calculateZeroPoint(fq_data1[c], fq_data2[c], max_levels, ov::element::u8));
                double scale = range / (max_levels - 1.0);
                fq_data1[c] = static_cast<float>((0.0 - zp) * scale);
                fq_data2[c] = static_cast<float>((max_levels - 1.0 - zp) * scale);
                align_zp(fq_data1[c], fq_data2[c], max_levels);
                fq_data3[c] = fq_data1[c];
                fq_data4[c] = fq_data2[c];
            }
        } else {
            // At least one Concat - should use Concat alignment
            for (size_t c = 0; c < fq_data1.size(); c++) {
                fq_data1[c] = min;
                fq_data2[c] = max;
                fq_data3[c] = min;
                fq_data4[c] = max;
            }
        }

        changed[i] = replace_node_if_changed(fq_node1, ov::element::f32, fq_data1, "_scale_aligned") || changed[i];
        changed[i] = replace_node_if_changed(fq_node2, ov::element::f32, fq_data2, "_scale_aligned") || changed[i];
        changed[i] = replace_node_if_changed(fq_node3, ov::element::f32, fq_data3, "_scale_aligned") || changed[i];
        changed[i] = replace_node_if_changed(fq_node4, ov::element::f32, fq_data4, "_scale_aligned") || changed[i];

        if (changed[i]) {
            if (fq->get_friendly_name().find("_scale_aligned") == std::string::npos)
                fq->set_friendly_name(fq->get_friendly_name() + "_scale_aligned");
        }
        i++;
    }

    i = 0;
    for (const auto& fq_node : fqs) {
        if (changed[i]) {
            broadcast_changes(fq_node);
        }
        i++;
    }
}

static void adjust_fqs_to_align(std::set<std::shared_ptr<ov::Node>>& fqs) {
    float min_range = std::numeric_limits<float>::max();
    std::set<std::shared_ptr<ov::Node>> filtered_fqs;
    size_t child_node_num = 0;
    const float max_fq_range_ratio = 5.0;

    for (auto fq_node : fqs) {
        auto fq_node1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node1 != nullptr);
        auto fq_node2 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node2 != nullptr);

        auto fq_data1 = fq_node1->cast_vector<float>();
        auto fq_data2 = fq_node2->cast_vector<float>();

        for (size_t c = 0; c < fq_data1.size(); c++) {
            if (min_range > fq_data2[c] - fq_data1[c]) {
                min_range = fq_data2[c] - fq_data1[c];
            }
        }

        for (const auto& node_output : fq_node->outputs()) {
            child_node_num += node_output.get_target_inputs().size();
        }
    }

    if (child_node_num == fqs.size())
        return;

    for (auto fq_node : fqs) {
        auto fq_node1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node1 != nullptr);
        auto fq_node2 = std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(fq_node2 != nullptr);

        auto fq_data1 = fq_node1->cast_vector<float>();
        auto fq_data2 = fq_node2->cast_vector<float>();

        for (size_t c = 0; c < fq_data1.size(); c++) {
            if (fq_data2[c] - fq_data1[c] < max_fq_range_ratio * min_range) {
                filtered_fqs.insert(fq_node);
                break;
            }
        }
    }

    fqs = std::move(filtered_fqs);
}

static void broadcast_changes(const std::shared_ptr<ov::Node>& node) {
    auto nodes_to_align = std::vector<std::shared_ptr<ov::Node>>();

    for (const auto& input : node->input_values()) {
        if (node_is_add_or_concat(input.get_node()->shared_from_this()) ||
            is_fq_agnostic(input.get_node()->shared_from_this())) {
            nodes_to_align.push_back(input.get_node()->shared_from_this());
        }
    }
    for (const auto& node_output : node->outputs()) {
        for (auto consumer : node_output.get_target_inputs()) {
            if (node_is_add_or_concat(consumer.get_node()->shared_from_this()) ||
                is_fq_agnostic(consumer.get_node()->shared_from_this())) {
                nodes_to_align.push_back(consumer.get_node()->shared_from_this());
            }
        }
    }

    for (const auto& node_to_align : nodes_to_align) {
        std::set<std::shared_ptr<ov::Node>> fqs_to_align;
        gather_fqs(node_to_align, fqs_to_align);
        if (fqs_to_align.size() < 2) {
            continue;
        }
        if (!all_fqs_have_same_io_params(fqs_to_align)) {
            continue;
        }
        if (fqs_to_align.size() > 2)
            adjust_fqs_to_align(fqs_to_align);

        float min = 0;
        float max = 0;
        float range = 0;
        int max_levels = 0;
        find_min_max(fqs_to_align, min, max, range, max_levels);

        align_zp(min, max, max_levels);

        align_fq(fqs_to_align, min, max, range, max_levels);
    }
}

static bool can_broadcast(const std::set<std::shared_ptr<ov::Node>>& fqs_to_align) {
    std::vector<size_t> channels;
    for (const auto& fq : fqs_to_align) {
        const auto shape = fq->input_value(1).get_node_shared_ptr()->get_shape();
        if (shape.size() > 1) {
            channels.push_back(shape.at(1));
        }
    }

    if (channels.empty()) {
        return true;
    }

    return std::all_of(channels.cbegin(), channels.cend(), [channels](const size_t& chan) -> bool {
        return chan == channels.at(0);
    });
}

// for concat, concat output fq min / max range should be the minimal / maximal of all the inputs
static void update_concat_out_fq(std::shared_ptr<ov::Node> node, const std::set<std::shared_ptr<ov::Node>>& fqs) {
    if (std::dynamic_pointer_cast<ov::op::v0::Concat>(node) != nullptr) {
        std::set<std::shared_ptr<ov::Node>> out_fqs;
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        bool min_update = false;
        bool max_update = false;

        for (const auto& node_output : node->outputs()) {
            for (auto consumer : node_output.get_target_inputs()) {
                auto consumer_node = consumer.get_node()->shared_from_this();
                if (std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(consumer_node) != nullptr) {
                    out_fqs.insert(consumer_node);
                }
            }
        }

        for (auto fq_node : fqs) {
            if (out_fqs.find(fq_node) == out_fqs.end()) {
                auto fq_node1 =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
                OPENVINO_ASSERT(fq_node1 != nullptr);
                auto fq_node2 =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
                OPENVINO_ASSERT(fq_node2 != nullptr);
                auto fq_data1 = fq_node1->cast_vector<float>();
                auto fq_data2 = fq_node2->cast_vector<float>();

                if (fq_data1.size() > 0) {
                    min = std::min(min, *std::min_element(fq_data1.begin(), fq_data1.end()));
                    min_update = true;
                }
                if (fq_data2.size() > 0) {
                    max = std::max(max, *std::max_element(fq_data2.begin(), fq_data2.end()));
                    max_update = true;
                }
            }
        }

        if (min_update && max_update) {
            for (auto fq_node : out_fqs) {
                auto fq_node1 =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
                OPENVINO_ASSERT(fq_node1 != nullptr);
                auto fq_node2 =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
                OPENVINO_ASSERT(fq_node2 != nullptr);
                auto fq_node3 =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(3).get_node_shared_ptr());
                OPENVINO_ASSERT(fq_node3 != nullptr);
                auto fq_node4 =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(fq_node->input_value(4).get_node_shared_ptr());
                OPENVINO_ASSERT(fq_node4 != nullptr);
                auto fq_data1 = fq_node1->cast_vector<float>();
                auto fq_data2 = fq_node2->cast_vector<float>();
                auto fq_data3 = fq_node3->cast_vector<float>();
                auto fq_data4 = fq_node4->cast_vector<float>();

                std::fill_n(fq_data1.begin(), fq_data1.size(), min);
                std::fill_n(fq_data2.begin(), fq_data2.size(), max);
                std::fill_n(fq_data3.begin(), fq_data3.size(), min);
                std::fill_n(fq_data4.begin(), fq_data4.size(), max);

                bool changed = false;
                changed |= replace_node_if_changed(fq_node1, ov::element::f32, fq_data1, "_minmax_aligned");
                changed |= replace_node_if_changed(fq_node2, ov::element::f32, fq_data2, "_minmax_aligned");
                changed |= replace_node_if_changed(fq_node3, ov::element::f32, fq_data3, "_minmax_aligned");
                changed |= replace_node_if_changed(fq_node4, ov::element::f32, fq_data4, "_minmax_aligned");
                if (changed) {
                    fq_node->set_friendly_name(fq_node->get_friendly_name() + "_minmax_aligned");
                    broadcast_changes(fq_node);
                }
            }
        }
    }
}

bool AlignScales::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool pass_applied = false;

    for (const std::shared_ptr<ov::Node>& node : m->get_ops()) {
        if (!node_is_add_or_concat(node)) {
            continue;
        }

        std::set<std::shared_ptr<ov::Node>> fqs_to_align;
        gather_fqs(node, fqs_to_align);
        if (!can_broadcast(fqs_to_align)) {
            continue;
        }

        if (fqs_to_align.size() < 2) {
            continue;
        }
        if (!all_fqs_have_same_io_params(fqs_to_align)) {
            continue;
        }

        update_concat_out_fq(node, fqs_to_align);

        if (fqs_to_align.size() > 2) {
            adjust_fqs_to_align(fqs_to_align);
        }

        float min = 0;
        float max = 0;
        float range = 0;
        int max_levels = 0;
        find_min_max(fqs_to_align, min, max, range, max_levels);

        align_zp(min, max, max_levels);
        align_fq(fqs_to_align, min, max, range, max_levels);

        pass_applied = true;
    }

    return pass_applied;
}

}  // namespace passes
}  // namespace vpux
