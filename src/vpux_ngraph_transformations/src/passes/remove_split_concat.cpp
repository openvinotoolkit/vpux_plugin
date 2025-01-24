//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/remove_split_concat.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/split.hpp>

#include <memory>

namespace vpux {
namespace pass {

// Modnet Workaround: remove [split -> concat] subgraph if it is directly after input node
bool RemoveSplitConcat::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool pass_applied = false;

    for (const std::shared_ptr<ov::Node>& node : m->get_ops()) {
        const auto split_node = std::dynamic_pointer_cast<ov::op::v1::Split>(node);
        if (split_node != nullptr) {
            const auto parent_node = split_node->input_value(0).get_node_shared_ptr();
            const auto parameter_node = std::dynamic_pointer_cast<ov::op::v0::Parameter>(parent_node);
            if (parameter_node == nullptr) {
                continue;
            }

            for (const auto& node_output : split_node->outputs()) {
                const auto target_inputs = node_output.get_target_inputs();
                if (!target_inputs.empty()) {
                    const auto consumer_input = *target_inputs.begin();
                    const auto consumer = consumer_input.get_node();
                    if (consumer) {
                        const auto concat_node =
                                std::dynamic_pointer_cast<ov::op::v0::Concat>(consumer->shared_from_this());
                        if (concat_node != nullptr) {
                            if (replace_output_update_name(concat_node->output(0), split_node->input_value(0))) {
                                pass_applied = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    return pass_applied;
}

}  // namespace pass
}  // namespace vpux
