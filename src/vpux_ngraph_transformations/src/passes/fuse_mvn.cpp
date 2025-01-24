//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/fuse_mvn.hpp"

#include <memory>
#include <vector>

#include <openvino/core/rt_info.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/mvn.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include "openvino/core/node.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/util/log.hpp"

#include "transformations/utils/utils.hpp"

#include "vpux/utils/core/error.hpp"

#include <algorithm>

namespace vpux {

namespace passes {

using namespace ov;

namespace {

enum class ReductionMode { ACROSS_CHANNELS, SPATIAL, UNSUPPORTED };

const double EPS_THRESHOLD = 1e-4;

ReductionMode getReductionMode(size_t inputOrder, std::vector<int64_t>& axes) {
    if (inputOrder < 2 || inputOrder > 4) {
        return ReductionMode::UNSUPPORTED;
    }

    int64_t referenceNumAxes = static_cast<int64_t>(inputOrder) - 1;
    const int64_t lastAxisNum = 3;
    int numMatched = 0;

    std::sort(axes.begin(), axes.end());
    for (int64_t i = 0; i < referenceNumAxes; ++i) {
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
std::function<bool(ov::Output<ov::Node>)> value_is_equal_to(const std::vector<T>& ref_values) {
    return [ref_values = ref_values](ov::Output<ov::Node> output) -> bool {
        auto node = output.get_node_shared_ptr();
        if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
            return const_node->template cast_vector<T>() == ref_values;
        }
        return false;
    };
}

};  // namespace

ConvertLayerNormToMVN::ConvertLayerNormToMVN() {
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps)
    const auto& x = pass::pattern::any_input();
    // (x - Reshape(ReduceMean(Unsqueeze(x), axes)))
    //     `-------------reshape1------------'

    auto unsqueeze1 = pass::pattern::wrap_type<opset6::Reshape>({x, ov::pass::pattern::any_input()});

    const auto& mean1Axes = pass::pattern::wrap_type<opset6::Constant>();
    const auto& mean1 = pass::pattern::wrap_type<opset6::ReduceMean>({unsqueeze1, mean1Axes});

    auto shape1 = pass::pattern::wrap_type<opset6::Constant>();
    auto reshape1 = pass::pattern::wrap_type<opset6::Reshape>({mean1, shape1});

    auto hasReshape1OrNot = std::make_shared<pass::pattern::op::Or>(OutputVector{mean1, reshape1});

    // (x - ReduceMean(x, axes))
    // `-sub1------------------'
    const auto& sub1 = pass::pattern::wrap_type<opset6::Subtract>({x, hasReshape1OrNot});

    const auto& cast = pass::pattern::wrap_type<opset6::Convert>({sub1});
    const auto hasConvertOrNot = std::make_shared<pass::pattern::op::Or>(OutputVector{cast, sub1});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `---------------------power--'
    auto const_2 = pass::pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({2.0}));
    const auto& power = pass::pattern::wrap_type<opset6::Power>({hasConvertOrNot, const_2});

    // Sqrt(Reshape(ReduceMean(Unsqueeze((x - ReduceMean(x, axes)) ^ 2))))
    //     `---mean2---------------------------------------------------'
    auto unsqueeze2 = pass::pattern::wrap_type<opset6::Reshape>({power, pass::pattern::any_input()});

    const auto& mean2Axes = pass::pattern::wrap_type<opset6::Constant>();
    const auto& mean2 = pass::pattern::wrap_type<opset6::ReduceMean>({unsqueeze2, mean2Axes});

    auto shape2 = pass::pattern::wrap_type<opset6::Constant>();
    auto reshape2 = pass::pattern::wrap_type<opset6::Reshape>({mean2, shape2});

    auto hasReshape2OrNot = std::make_shared<pass::pattern::op::Or>(OutputVector{mean2, reshape2});

    auto const_0_5 = pass::pattern::wrap_type<ov::opset6::Constant>(value_is_equal_to<float>({0.5}));
    const auto& eps = pass::pattern::wrap_type<opset6::Constant>();
    // ------------------- OUTSIDE_SQRT ----------------------

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    const auto& powerSqrtOs = pass::pattern::wrap_type<opset6::Power>({hasReshape2OrNot, const_0_5});
    const auto& sqrtOs = pass::pattern::wrap_type<opset6::Sqrt>({hasReshape2OrNot});
    const auto powerOrSqrtOs = std::make_shared<pass::pattern::op::Or>(OutputVector{powerSqrtOs, sqrtOs});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps
    // `----------------------------------------------Add---'
    const auto& addEpsOs = pass::pattern::wrap_type<opset6::Add>({powerOrSqrtOs, eps});

    // // ------------------- INSIDE_SQRT ----------------------

    // (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps))
    // `-----------------------------------------------Add---'
    const auto& addEpsIs = pass::pattern::wrap_type<opset6::Add>({hasReshape2OrNot, eps});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    const auto& powerSqrtIs = pass::pattern::wrap_type<opset6::Power>({addEpsIs, const_0_5});
    const auto& sqrtIs = pass::pattern::wrap_type<opset6::Sqrt>({addEpsIs});
    const auto powerOrSqrtIs = std::make_shared<pass::pattern::op::Or>(OutputVector{powerSqrtIs, sqrtIs});

    auto outsideOrInside = std::make_shared<pass::pattern::op::Or>(OutputVector{addEpsOs, powerOrSqrtIs});

    // Final Divide
    auto const_neg_1 = pass::pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({-1}));
    const auto& powerDiv = pass::pattern::wrap_type<opset6::Power>({outsideOrInside, const_neg_1});
    const auto& div = pass::pattern::wrap_type<opset6::Multiply>({sub1, powerDiv});

    const auto& divAlt = pass::pattern::wrap_type<opset6::Divide>({sub1, outsideOrInside});
    const auto powerMulOrDiv = std::make_shared<pass::pattern::op::Or>(OutputVector{div, divAlt});

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        auto& patternToOutput = m.get_pattern_value_map();
        auto exp_input = patternToOutput.at(x);

        auto const_eps_node =
                std::dynamic_pointer_cast<ov::opset6::Constant>(patternToOutput.at(eps).get_node_shared_ptr());
        float eps_value;
        if (const_eps_node == nullptr || !ov::op::util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        auto axes1Node =
                std::dynamic_pointer_cast<ov::opset6::Constant>(patternToOutput.at(mean1Axes).get_node_shared_ptr());
        auto axes2Node =
                std::dynamic_pointer_cast<ov::opset6::Constant>(patternToOutput.at(mean2Axes).get_node_shared_ptr());

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

        ov::NodeVector nodes_to_copy_info(
                {patternToOutput.at(mean1).get_node_shared_ptr(), patternToOutput.at(mean2).get_node_shared_ptr(),
                 patternToOutput.at(sub1).get_node_shared_ptr(), patternToOutput.at(power).get_node_shared_ptr()});

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
            auto preReshape = std::make_shared<ov::opset6::Reshape>(exp_input.get_node_shared_ptr(), newAxis, false);

            auto mvn = std::make_shared<ov::op::v0::MVN>(preReshape, across_channels, true, eps_value);

            auto shape = opset6::Constant::create(ov::element::i64, ov::Shape{inputRank}, inputShape);
            auto postReshape = std::make_shared<ov::opset6::Reshape>(mvn, shape, false);

            postReshape->set_friendly_name(m.get_match_root()->get_friendly_name());
            ov::copy_runtime_info(nodes_to_copy_info, postReshape);
            ov::replace_node(m.get_match_root(), postReshape);
        } else {
            auto mvn = std::make_shared<ov::op::v0::MVN>(exp_input.get_node_shared_ptr(), across_channels, true,
                                                         eps_value);
            mvn->set_friendly_name(m.get_match_root()->get_friendly_name());
            ov::copy_runtime_info(nodes_to_copy_info, mvn);
            ov::replace_node(m.get_match_root(), mvn);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(powerMulOrDiv, "ConvertLayerNormToMVN");
    register_matcher(m, matcher_pass_callback);
}

ConvertInstanceNormToMVN::ConvertInstanceNormToMVN() {
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) * gamma / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) + beta
    const auto& x = pass::pattern::any_input();

    // (x - ReduceMean(x, axes))^2
    //     `------mean1-------'
    const auto& mean1Axis = pass::pattern::wrap_type<opset6::Constant>();
    const auto& mean1 = pass::pattern::wrap_type<opset6::ReduceMean>({x, mean1Axis});

    // (x - ReduceMean(x, axes))^2
    // `-squared_difference------'
    const auto& sqDiff = pass::pattern::wrap_type<opset6::SquaredDifference>({x, mean1});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    //         `---mean2--------------------------------'
    const auto& mean2Axis = pass::pattern::wrap_type<opset6::Constant>();
    const auto& mean2 = pass::pattern::wrap_type<opset6::ReduceMean>({sqDiff, mean2Axis});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    //         `------------------------------------------add--'
    const auto& eps = pass::pattern::wrap_type<opset6::Constant>();
    const auto& addEps = pass::pattern::wrap_type<opset6::Add>({mean2, eps});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `-power-------------------------------------------------'
    const auto& const_0_5 = pass::pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({-0.5}));
    const auto& power = pass::pattern::wrap_type<opset6::Power>({addEps, const_0_5});

    // gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---mul1----------------------------------------------------'
    const auto& gamma = pass::pattern::wrap_type<opset6::Constant>();
    const auto& mul1 = pass::pattern::wrap_type<opset6::Multiply>({power, gamma});

    // x * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---mul2--------------------------------------------------------'
    const auto& mulLeft = pass::pattern::wrap_type<opset6::Multiply>({x, mul1});

    // ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) - beta
    // `-------------------mul3----------------------------------------------------------'
    const auto& mulRight = pass::pattern::wrap_type<opset6::Multiply>({mul1, mean1});

    // beta - ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---sub-----------------------------------------------------------------------------------'
    const auto& beta = pass::pattern::wrap_type<opset6::Constant>();
    const auto& sub = pass::pattern::wrap_type<opset6::Subtract>({beta, mulRight});

    // Final Add
    // x * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) +
    // beta - ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) =
    // gamma * (x - ReduceMean(x, axes)) / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) + beta
    auto add = pass::pattern::wrap_type<opset6::Add>({mulLeft, sub});

    ov::matcher_pass_callback matcherPassCallback = [=](ov::pass::pattern::Matcher& matcher) {
        auto& patternToOutput = matcher.get_pattern_value_map();
        auto xOutput = patternToOutput.at(x);

        auto const_0_5_node =
                std::dynamic_pointer_cast<ov::opset6::Constant>(patternToOutput.at(const_0_5).get_node_shared_ptr());
        auto constEpsNode =
                std::dynamic_pointer_cast<ov::opset6::Constant>(patternToOutput.at(eps).get_node_shared_ptr());
        auto constGammaNode =
                std::dynamic_pointer_cast<ov::opset6::Constant>(patternToOutput.at(gamma).get_node_shared_ptr());
        auto constBetaNode =
                std::dynamic_pointer_cast<ov::opset6::Constant>(patternToOutput.at(beta).get_node_shared_ptr());

        if (!const_0_5_node || !constBetaNode || !constGammaNode || !constEpsNode) {
            return false;
        }

        float epsValue = 0.0;
        bool validConstValues = ov::op::util::has_constant_value<float>(const_0_5_node, -0.5) &&
                                ov::op::util::get_single_value(constEpsNode, epsValue);
        if (!validConstValues) {
            return false;
        }

        auto mean1AxisNode =
                std::dynamic_pointer_cast<ov::opset6::Constant>(patternToOutput.at(mean1Axis).get_node_shared_ptr());
        auto mean2AxisNode =
                std::dynamic_pointer_cast<ov::opset6::Constant>(patternToOutput.at(mean2Axis).get_node_shared_ptr());
        if (!mean1AxisNode || !mean2AxisNode) {
            return false;
        }

        auto mean1AxisValue = mean1AxisNode->cast_vector<int64_t>();
        auto mean2AxisValue = mean2AxisNode->cast_vector<int64_t>();

        if (mean1AxisValue != mean2AxisValue) {
            return false;
        }

        // The reduction axes are 1,2 hence a transpose pair is required to shift the axes to 2,3 as MVN can only work
        // with across_channels true or false Reduction axes are determined by across channles flag across_channels =
        // true reduction axes are 1, 2, 3 across channels = false reduction axes 2, 3
        const std::vector<int64_t> transposeArg0Weights = {0, 3, 2, 1};
        auto transposeArg0Const = std::make_shared<ov::op::v0::Constant>(
                ov::element::Type_t::i64, ov::Shape{transposeArg0Weights.size()}, transposeArg0Weights.data());
        auto transposeArg0Node =
                std::make_shared<ov::op::v1::Transpose>(xOutput.get_node_shared_ptr(), transposeArg0Const->output(0));

        auto mvn = std::make_shared<ov::op::v0::MVN>(transposeArg0Node, /*across_channels*/ false,
                                                     /*normalize_variance*/ true, epsValue);

        const std::vector<int64_t> transposeArg1Weights = {0, 3, 2, 1};
        auto transposeArg1Const = std::make_shared<ov::op::v0::Constant>(
                ov::element::Type_t::i64, ov::Shape{transposeArg1Weights.size()}, transposeArg1Weights.data());
        auto transposeArg1Node = std::make_shared<ov::op::v1::Transpose>(mvn, transposeArg1Const->output(0));

        auto mulGamma = std::make_shared<ov::opset6::Multiply>(transposeArg1Node, constGammaNode);
        auto addBeta = std::make_shared<ov::opset6::Add>(mulGamma, constBetaNode);

        ov::NodeVector nodes_to_copy_info(
                {patternToOutput.at(mean1).get_node_shared_ptr(), patternToOutput.at(sqDiff).get_node_shared_ptr(),
                 patternToOutput.at(addEps).get_node_shared_ptr(),

                 patternToOutput.at(power).get_node_shared_ptr(), patternToOutput.at(mul1).get_node_shared_ptr(),

                 patternToOutput.at(mulLeft).get_node_shared_ptr(), patternToOutput.at(mulRight).get_node_shared_ptr(),

                 patternToOutput.at(sub).get_node_shared_ptr(), patternToOutput.at(add).get_node_shared_ptr()});

        ov::copy_runtime_info(nodes_to_copy_info,
                              {transposeArg0Const, transposeArg0Node, mvn, transposeArg1Const, transposeArg1Node,
                               constGammaNode, mulGamma, constBetaNode, addBeta});
        addBeta->set_friendly_name(matcher.get_match_root()->get_friendly_name());
        ov::replace_node(matcher.get_match_root(), addBeta);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(add, "ConvertInstanceNormToMVN");
    register_matcher(matcher, matcherPassCallback);
}

}  // namespace passes
}  // namespace vpux
