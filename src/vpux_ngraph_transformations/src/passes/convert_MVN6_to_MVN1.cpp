//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/convert_MVN6_to_MVN1.hpp"

#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/mvn.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include "openvino/core/node.hpp"
#include "openvino/util/log.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

namespace passes {

ConvertMVN6toMVN1::ConvertMVN6toMVN1() {
    auto mvn6 = ov::pass::pattern::wrap_type<ov::op::v6::MVN>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto mvn6 = std::dynamic_pointer_cast<ov::op::v6::MVN>(m.get_match_root());
        if (!mvn6) {
            return false;
        }
        const auto eps_mode = mvn6->get_eps_mode();

        const float eps = mvn6->get_eps();
        if (eps_mode != ov::op::MVNEpsMode::INSIDE_SQRT) {
            // MVN-1 does not support outside_sqrt eps mode, in this case we should do MVN6Decomposition pass
            // Disable temporarily to enable the BDK3 ModNet.
            OPENVINO_WARN << "MVN-1 does not support outside_sqrt eps mode.";

            // For small enough 'eps' values, can treat OUTSIDE_SQRT mode as INSIDE_SQRT
            const double EPS_THRESHOLD = 1e-3;
            if (eps > EPS_THRESHOLD) {
                return false;
            }
        }

        const auto input = mvn6->input_value(0);

        const bool normalize_variance = mvn6->get_normalize_variance();

        auto const_axes = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                mvn6->input(1).get_source_output().get_node_shared_ptr());
        OPENVINO_ASSERT(nullptr != const_axes);
        auto axes = const_axes->cast_vector<int32_t>();

        const auto dims_count = input.get_partial_shape().get_max_shape().size();
        if (!(static_cast<int32_t>(dims_count) >= 2 && static_cast<int32_t>(dims_count) <= 4)) {
            OPENVINO_WARN << "MVN6->MVN1 conversion supports only 2D, 3D or 4D cases";
            return false;
        }

        for (auto& it : axes) {
            it = it < 0 ? it + checked_cast<int32_t>(dims_count) : it;
        }

        std::sort(axes.begin(), axes.end());

        bool across_channels = false;
        auto inputShape = input.get_partial_shape().get_shape();
        std::vector<size_t> newInShape;
        if ((dims_count == 2 || dims_count == 3 || dims_count == 4) && axes.size() == 1 &&
            static_cast<uint32_t>(axes[0]) == (dims_count - 1)) {
            // clang-format off
            // For this case(calculate mean value on width), convert the 2D/3D MVN6 to MVN1 by the steps in below.
            // 1.Reshape input to 4D shape(CxHxW -> CxHx1xW or HxW -> 1xHx1xW).
            // 2.Create MVN-1 op with new 4D input shape, axes.size() == 1 && axes[0] == 2 means do not share mean values across channels.
            // 3.Reshape 4D result to original 2D/3D shape.
            // clang-format on
            across_channels = false;
            if (inputShape.size() == 4) {
                // Conversion from MVN6 -> MVN1 for 4D shape and axes.size() == 1, implicitly assumes H*W on the second
                // position, due to the fact that MVN1 definition is more restrictive
                newInShape.push_back(inputShape[0]);
                newInShape.push_back(inputShape[1] * inputShape[2]);
                newInShape.push_back(1);
                newInShape.push_back(inputShape[3]);

            } else if (inputShape.size() == 3) {
                // CxHxW -> CxHxWx1
                newInShape.push_back(inputShape[0]);
                newInShape.push_back(inputShape[1]);
                newInShape.push_back(inputShape[2]);
                newInShape.push_back(1);
            } else if (inputShape.size() == 2) {
                // HxW -> 1xHxWx1
                newInShape.push_back(1);
                newInShape.push_back(inputShape[0]);
                newInShape.push_back(inputShape[1]);
                newInShape.push_back(1);
            } else {
                VPUX_THROW("Unexpected input shape");
            }
            auto constNode = std::make_shared<ov::opset1::Constant>(ov::element::Type_t::i64,
                                                                    ov::Shape{newInShape.size()}, newInShape);
            auto reshapeInput = std::dynamic_pointer_cast<ov::opset1::Reshape>(
                    std::make_shared<ov::opset1::Reshape>(input, constNode, false));

            const auto Mvn1 =
                    std::make_shared<ov::op::v0::MVN>(reshapeInput, across_channels, normalize_variance, (double)(eps));
            Mvn1->set_friendly_name(mvn6->get_friendly_name());

            // Output shape is equal with input shape
            auto constNode2 = std::make_shared<ov::opset1::Constant>(ov::element::Type_t::i64,
                                                                     ov::Shape{inputShape.size()}, inputShape);
            auto reshapeOutput = std::dynamic_pointer_cast<ov::opset1::Reshape>(
                    std::make_shared<ov::opset1::Reshape>(Mvn1, constNode2, false));
            reshapeOutput->set_friendly_name(mvn6->get_friendly_name());

            ov::replace_node(mvn6, reshapeOutput);
            return true;
        } else if (dims_count == 4) {
            if (axes.size() == 3 && axes[0] == 1 && axes[1] == 2 && axes[2] == 3)
                across_channels = true;
            else if (axes.size() == 2 && axes[0] == 2 && axes[1] == 3)
                across_channels = false;
            else {
                // MVN-1 layer supports only normalization across channel or spatial dimension, in this case we
                // should do MVN6Decomposition pass
                return false;
            }

            const auto Mvn1 =
                    std::make_shared<ov::op::v0::MVN>(input, across_channels, normalize_variance, (double)(eps));
            Mvn1->set_friendly_name(mvn6->get_friendly_name());

            ov::replace_node(mvn6, Mvn1);
            return true;
        } else {
            // MVN6->MVN1 conversion failed, rely on MVN6 op
            return false;
        }
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mvn6, "ConvertMVN6toMVN1");
    register_matcher(m, callback);
}

}  // namespace passes
}  // namespace vpux
