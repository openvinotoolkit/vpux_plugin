//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/replace_onnx_pattern_to_reorg.hpp"
#include <openvino/core/rt_info.hpp>
#include <openvino/op/reorg_yolo.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace vpux {
namespace passes {

OnnxReorgPatternToDarkNetReorg::OnnxReorgPatternToDarkNetReorg() {
    const auto& input = ov::pass::pattern::any_input();
    const auto& reshape1 =
            std::make_shared<ov::opset1::Reshape>(input, ov::pass::pattern::wrap_type<ov::op::v0::Constant>(), true);
    const auto& transpose1 =
            std::make_shared<ov::opset1::Transpose>(reshape1, ov::pass::pattern::wrap_type<ov::op::v0::Constant>());
    const auto& reshape2 = std::make_shared<ov::opset1::Reshape>(
            transpose1, ov::pass::pattern::wrap_type<ov::op::v0::Constant>(), true);
    const auto& transpose2 =
            std::make_shared<ov::opset1::Transpose>(reshape2, ov::pass::pattern::wrap_type<ov::op::v0::Constant>());
    const auto& reshape3 = std::make_shared<ov::opset1::Reshape>(
            transpose2, ov::pass::pattern::wrap_type<ov::op::v0::Constant>(), true);
    const auto& transpose3 =
            std::make_shared<ov::opset1::Transpose>(reshape3, ov::pass::pattern::wrap_type<ov::op::v0::Constant>());
    const auto& reshape4 = std::make_shared<ov::opset1::Reshape>(
            transpose3, ov::pass::pattern::wrap_type<ov::op::v0::Constant>(), true);

    ov::graph_rewrite_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto reorg_input = pattern_to_output.at(input);

        auto first_reshape =
                std::dynamic_pointer_cast<ov::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr());
        if (first_reshape == nullptr) {
            return false;
        }

        auto last_reshape =
                std::dynamic_pointer_cast<ov::opset1::Reshape>(pattern_to_output.at(reshape4).get_node_shared_ptr());
        if (last_reshape == nullptr) {
            return false;
        }

        std::vector<size_t> first_reshape_in_dims = first_reshape->get_input_shape(0);
        std::vector<size_t> last_reshape_out_dims = last_reshape->get_output_shape(0);

        // Input\Output shapes for Reorg layer from original YoloV2 network
        // Expectations that this pass will only work on the "classic" version of the network
        std::vector<size_t> expected_in_shape = {1, 64, 26, 26};
        std::vector<size_t> expected_out_shape = {1, 256, 13, 13};

        if ((first_reshape_in_dims != expected_in_shape) || (last_reshape_out_dims != expected_out_shape)) {
            return false;
        }

        auto reorgYolo = std::make_shared<ov::op::v0::ReorgYolo>(reorg_input, ov::Strides{2});

        reorgYolo->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
                {
                        pattern_to_output.at(reshape1).get_node_shared_ptr(),
                        pattern_to_output.at(transpose1).get_node_shared_ptr(),
                        pattern_to_output.at(reshape2).get_node_shared_ptr(),
                        pattern_to_output.at(transpose2).get_node_shared_ptr(),
                        pattern_to_output.at(reshape3).get_node_shared_ptr(),
                        pattern_to_output.at(transpose3).get_node_shared_ptr(),
                        pattern_to_output.at(reshape4).get_node_shared_ptr(),
                },
                reorgYolo);
        ov::replace_node(m.get_match_root(), reorgYolo);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape4, "OnnxReorgPatternToDarkNetReorg");
    register_matcher(m, matcher_pass_callback);
}

}  // namespace passes
}  // namespace vpux
