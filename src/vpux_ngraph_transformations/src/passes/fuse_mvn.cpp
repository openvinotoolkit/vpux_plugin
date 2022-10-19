//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include "vpux/passes/fuse_mvn.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset6.hpp>
#include "ngraph/opsets/opset4.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/op/mvn.hpp>
#include <ngraph/op/constant.hpp>
#include <details/ie_exception.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "ngraph/node.hpp"
#include "ngraph/log.hpp"

#include "transformations/utils/utils.hpp"

#include "vpux/utils/core/error.hpp"

#include <algorithm>

namespace vpux {

namespace passes {

using namespace ngraph;

namespace {

enum class ReductionMode {
    ACROSS_CHANNELS, 
    SPATIAL,
    UNSUPPORTED
};

const double EPS_THRESHOLD = 1e-4;

ReductionMode getReductionMode(size_t inputOrder, std::vector<int64_t> &axes) {
    if (inputOrder < 2 || inputOrder > 4) {
        return ReductionMode::UNSUPPORTED;
    }
     
    int64_t referenceNumAxes = static_cast<int64_t>(inputOrder) - 1;
    const int64_t lastAxisNum = 3;
    int numMatched = 0;

    std::sort(axes.begin(), axes.end());
    for(int64_t i = 0; i < referenceNumAxes; ++i) {
        if (axes[axes.size() - i - 1] != lastAxisNum - i) {
            break;
        }
        ++numMatched;
    }
    if (numMatched == referenceNumAxes) {
        return ReductionMode::ACROSS_CHANNELS;
    } else if (numMatched + 1 == referenceNumAxes) {
        return ReductionMode::SPATIAL;
    }

    // Currently MVN-6 supported as extended by sqrt-inside mode MVN-1
    // This node couldnt be handled by MVN-1, so no fusion
    return ReductionMode::UNSUPPORTED;
}

template <class T>
std::function<bool(ngraph::Output<ngraph::Node>)> value_is_equal_to(const std::vector<T>& ref_values) {
    return [ref_values](ngraph::Output<ngraph::Node> output) -> bool {
        auto node = output.get_node_shared_ptr();
        if (auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(node)) {
            return const_node->template cast_vector<T>() == ref_values;
        }
        return false;
    };
}

};

ConvertLayerNormToMVN::ConvertLayerNormToMVN() {
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps)
    auto x = pattern::any_input();
    // (x - Reshape(ReduceMean(Unsqueeze(x), axes)))
    //     `-------------reshape1------------'

    auto unsqueeze1 = pattern::wrap_type<opset6::Reshape>({x, ngraph::pattern::any_input()});

    auto mean1Axes = pattern::wrap_type<opset6::Constant>();
    auto mean1 = pattern::wrap_type<opset6::ReduceMean>({ unsqueeze1, mean1Axes });

    auto shape1 = pattern::wrap_type<opset6::Constant>();
    auto reshape1 = pattern::wrap_type<opset6::Reshape>({mean1, shape1});

    auto hasReshape1OrNot = std::make_shared<pattern::op::Or>(OutputVector{mean1, reshape1 });

    // (x - ReduceMean(x, axes))
    // `-sub1------------------'
    auto sub1 = pattern::wrap_type<opset6::Subtract>({ x, hasReshape1OrNot });

    auto cast = pattern::wrap_type<opset6::Convert>({ sub1 });
    const auto hasConvertOrNot = std::make_shared<pattern::op::Or>(OutputVector{ cast, sub1 });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `---------------------power--'
    auto const_2 = pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({ 2.0 }));
    auto power = pattern::wrap_type<opset6::Power>({ hasConvertOrNot, const_2 });

    // Sqrt(Reshape(ReduceMean(Unsqueeze((x - ReduceMean(x, axes)) ^ 2))))
    //     `---mean2---------------------------------------------------'
    auto unsqueeze2 = pattern::wrap_type<opset6::Reshape>({power, ngraph::pattern::any_input()});

    auto mean2Axes = pattern::wrap_type<opset6::Constant>();
    auto mean2 = pattern::wrap_type<opset6::ReduceMean>({ unsqueeze2, mean2Axes });

    auto shape2 = pattern::wrap_type<opset6::Constant>();
    auto reshape2 = pattern::wrap_type<opset6::Reshape>({mean2, shape2});

    auto hasReshape2OrNot = std::make_shared<pattern::op::Or>(OutputVector{mean2, reshape2 });

    auto const_0_5 = pattern::wrap_type<ngraph::opset6::Constant>(value_is_equal_to<float>({0.5}));
    auto eps = pattern::wrap_type<opset6::Constant>();
    // ------------------- OUTSIDE_SQRT ----------------------

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto powerSqrtOs = pattern::wrap_type<opset6::Power>({ hasReshape2OrNot, const_0_5 });
    auto sqrtOs = pattern::wrap_type<opset6::Sqrt>({ hasReshape2OrNot });
    const auto powerOrSqrtOs = std::make_shared<pattern::op::Or>(OutputVector{ powerSqrtOs, sqrtOs });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps
    // `----------------------------------------------Add---'
    auto addEpsOs = pattern::wrap_type<opset6::Add>({ powerOrSqrtOs, eps });

    // // ------------------- INSIDE_SQRT ----------------------

    // (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps))
    // `-----------------------------------------------Add---'
    auto addEpsIs = pattern::wrap_type<opset6::Add>({ hasReshape2OrNot, eps });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto powerSqrtIs = pattern::wrap_type<opset6::Power>({ addEpsIs, const_0_5 });
    auto sqrtIs = pattern::wrap_type<opset6::Sqrt>({ addEpsIs });
    const auto powerOrSqrtIs = std::make_shared<pattern::op::Or>(OutputVector{ powerSqrtIs, sqrtIs });

    auto outsideOrInside = std::make_shared<pattern::op::Or>(OutputVector{ addEpsOs, powerOrSqrtIs });

    // Final Divide
    auto const_neg_1 = pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({ -1 }));
    auto powerDiv = pattern::wrap_type<opset6::Power>({ outsideOrInside, const_neg_1 });
    auto div = pattern::wrap_type<opset6::Multiply>({ sub1, powerDiv });

    auto divAlt = pattern::wrap_type<opset6::Divide>({ sub1, outsideOrInside });
    const auto powerMulOrDiv = std::make_shared<pattern::op::Or>(OutputVector{ div, divAlt });

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& patternToOutput = m.get_pattern_value_map();
        auto exp_input = patternToOutput.at(x);
    
        auto const_eps_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(patternToOutput.at(eps).get_node_shared_ptr());
        float eps_value;
        if (!op::util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        auto axes1Node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(patternToOutput.at(mean1Axes).get_node_shared_ptr());
        auto axes2Node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(patternToOutput.at(mean2Axes).get_node_shared_ptr());

        if (!axes1Node || !axes2Node) {
            return false;
        }

        auto axes1Value = axes1Node->cast_vector<int64_t>();
        auto axes2Value = axes2Node->cast_vector<int64_t>();

        if (axes1Value != axes2Value) {
            return false;
        }

        const auto inputShape = exp_input.get_node_shared_ptr()->get_output_shape(0);
        auto reductionMode = getReductionMode(inputShape.size(), axes1Value);
        if (reductionMode == ReductionMode::UNSUPPORTED) {
            return false;
        }
        bool across_channels = reductionMode == ReductionMode::ACROSS_CHANNELS;

        ngraph::NodeVector nodes_to_copy_info({patternToOutput.at(mean1).get_node_shared_ptr(),
                                                patternToOutput.at(mean2).get_node_shared_ptr(),
                                                patternToOutput.at(sub1).get_node_shared_ptr(),
                                                patternToOutput.at(power).get_node_shared_ptr() });

        op::MVNEpsMode mode;
        if (patternToOutput.count(addEpsOs)) {
            mode = op::MVNEpsMode::OUTSIDE_SQRT;
            nodes_to_copy_info.push_back(patternToOutput.at(addEpsOs).get_node_shared_ptr());
            if (patternToOutput.count(powerSqrtOs)) {
                nodes_to_copy_info.push_back(patternToOutput.at(powerSqrtOs).get_node_shared_ptr());
            } else if (patternToOutput.count(sqrtOs)) {
                nodes_to_copy_info.push_back(patternToOutput.at(sqrtOs).get_node_shared_ptr());
            }
        } else if (patternToOutput.count(powerOrSqrtIs)) {
            mode = op::MVNEpsMode::INSIDE_SQRT;
            nodes_to_copy_info.push_back(patternToOutput.at(addEpsIs).get_node_shared_ptr());
            if (patternToOutput.count(powerSqrtIs)) {
                nodes_to_copy_info.push_back(patternToOutput.at(powerSqrtIs).get_node_shared_ptr());
            } else if (patternToOutput.count(sqrtIs)) {
                nodes_to_copy_info.push_back(patternToOutput.at(sqrtIs).get_node_shared_ptr());
            }
        } else {
            return false;
        }
        // Runtime kernel implement OUTSIDE_SQRT mode, but with small eps values there no difference in results
        if (mode == op::MVNEpsMode::INSIDE_SQRT && eps_value > EPS_THRESHOLD) {
            return false;
        }

        if (patternToOutput.count(cast)) {
            nodes_to_copy_info.push_back(patternToOutput.at(cast).get_node_shared_ptr());
        }

        if (patternToOutput.count(divAlt)) {
            nodes_to_copy_info.push_back(patternToOutput.at(divAlt).get_node_shared_ptr());
        } else if (patternToOutput.count(powerDiv) && patternToOutput.count(div)) {
            nodes_to_copy_info.push_back(patternToOutput.at(powerDiv).get_node_shared_ptr());
            nodes_to_copy_info.push_back(patternToOutput.at(div).get_node_shared_ptr());
        }

        const auto inputRank = inputShape.size();
        if (inputRank == 2 || inputRank == 3) {
            auto newShape(inputShape);
            while (newShape.size() != 4) {
                newShape.push_back(1);
            }

            auto newAxis = opset6::Constant::create(ov::element::i64, ov::Shape{newShape.size()}, newShape);
            auto preReshape = std::make_shared<ngraph::opset6::Reshape>(exp_input.get_node_shared_ptr(), newAxis, false);

            auto mvn = std::make_shared<ngraph::op::v0::MVN>(preReshape, across_channels, true, eps_value);

            auto shape = opset6::Constant::create(ov::element::i64, ov::Shape{inputRank}, inputShape);
            auto postReshape = std::make_shared<ngraph::opset6::Reshape>(mvn, shape, false);

            postReshape->set_friendly_name(m.get_match_root()->get_friendly_name());
            ngraph::copy_runtime_info(nodes_to_copy_info, postReshape);
            ngraph::replace_node(m.get_match_root(), postReshape);
        } else {
            auto mvn = std::make_shared<ngraph::op::v0::MVN>(exp_input.get_node_shared_ptr(), across_channels, true, eps_value);
            mvn->set_friendly_name(m.get_match_root()->get_friendly_name());
            ngraph::copy_runtime_info(nodes_to_copy_info, mvn);
            ngraph::replace_node(m.get_match_root(), mvn);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(powerMulOrDiv, "ConvertLayerNormToMVN");
    register_matcher(m, matcher_pass_callback);
}



}  // namespace passes
}  // namespace vpux
// clang-format on
