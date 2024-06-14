//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/fuse_padding.hpp"
#include <memory>
#include <vector>

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/pattern/op/label.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace vpux {
namespace pass {

FusePadding::FusePadding() {
    auto input = ov::pass::pattern::any_input();
    const auto& pad_value = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    const auto& pad_begin = std::make_shared<ov::pass::pattern::op::Label>(ov::element::i64, ov::Shape{4});
    const auto& pad_end = std::make_shared<ov::pass::pattern::op::Label>(ov::element::i64, ov::Shape{4});

    const ov::Strides strides = {1, 1, 1, 1};
    const ov::CoordinateDiff pad_diff = {0, 0, 0, 0};
    const ov::Shape pool_shape = {1, 1, 1, 1};

    const auto pad = std::make_shared<ov::op::v1::Pad>(input, pad_begin, pad_end, pad_value, ov::op::PadMode::CONSTANT);
    const auto conv = std::make_shared<ov::op::v1::Convolution>(pad, ov::pass::pattern::any_input(), strides, pad_diff,
                                                                pad_diff, strides);

    const auto group_conv = std::make_shared<ov::op::v1::GroupConvolution>(pad, ov::pass::pattern::any_input(), strides,
                                                                           pad_diff, pad_diff, strides);

    const auto maxpool = std::make_shared<ov::op::v1::MaxPool>(pad, strides, pool_shape, pool_shape, pool_shape);

    const auto matcher_nodes = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{conv, group_conv, maxpool});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto pattern_to_output = m.get_pattern_map();
        auto pad_node = std::dynamic_pointer_cast<ov::op::v1::Pad>(pattern_to_output.at(pad));
        auto pad_begin_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_to_output.at(pad_begin));
        auto pad_end_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_to_output.at(pad_end));
        auto pad_value_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_to_output.at(pad_value));

        // apply pass only for constant padding with 0 value
        if (!pad_node || !pad_begin_node || !pad_end_node || !pad_value_node ||
            pad_node->get_pad_mode() != ov::op::PadMode::CONSTANT)
            return false;

        auto pad_value_element = pad_value_node->cast_vector<int64_t>();
        if (pad_value_element.size() != 1 || pad_value_element[0] != 0)
            return false;

        auto layer = m.get_match_root();

        if (!layer)
            return false;

        if (auto convolution = std::dynamic_pointer_cast<ov::op::v1::Convolution>(layer)) {
            if (setPadding<ov::CoordinateDiff>(
                        convolution->get_pads_begin().size(), pad_begin_node->get_coordinate_diff_val(),
                        pad_end_node->get_coordinate_diff_val(),
                        [&convolution](const ov::CoordinateDiff& begin, const ov::CoordinateDiff& end) {
                            convolution->set_pads_begin(begin);
                            convolution->set_pads_end(end);
                        })) {
                convolution->set_auto_pad(ov::op::PadType::EXPLICIT);
            } else {
                return false;
            }

        } else if (auto group_conv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(layer)) {
            if (setPadding<ov::CoordinateDiff>(
                        group_conv->get_pads_begin().size(), pad_begin_node->get_coordinate_diff_val(),
                        pad_end_node->get_coordinate_diff_val(),
                        [&group_conv](const ov::CoordinateDiff& begin, const ov::CoordinateDiff& end) {
                            group_conv->set_pads_begin(begin);
                            group_conv->set_pads_end(end);
                        })) {
                group_conv->set_auto_pad(ov::op::PadType::EXPLICIT);
            } else {
                return false;
            }
        } else if (auto maxpool = std::dynamic_pointer_cast<ov::op::v1::MaxPool>(layer)) {
            if (setPadding<ov::Shape>(maxpool->get_pads_begin().size(), pad_begin_node->get_shape_val(),
                                      pad_end_node->get_shape_val(),
                                      [&maxpool](const ov::Shape& begin, const ov::Shape& end) {
                                          maxpool->set_pads_begin(begin);
                                          maxpool->set_pads_end(end);
                                      })) {
                maxpool->set_auto_pad(ov::op::PadType::EXPLICIT);
            } else {
                return false;
            }
        } else {
            return false;
        }

        return ov::replace_output_update_name(pad_node->output(0), pad_node->input_value(0));
    };

    const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(matcher_nodes, "FusePadding");
    register_matcher(matcher, callback);
}

template <class T>
bool FusePadding::setPadding(const size_t rank, const T& pads_begin, const T& pads_end,
                             const std::function<void(const T&, const T&)>& setter) {
    if (rank < 1 || pads_begin.size() <= rank || pads_end.size() <= rank)
        return false;

    setter(T(pads_begin.begin() + rank, pads_begin.end()), T(pads_end.begin() + rank, pads_end.end()));

    return true;
}

}  // namespace pass
}  // namespace vpux
