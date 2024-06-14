//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "graph_transformations.h"
#include "common_test_utils/test_assertions.hpp"
#include "vpux_driver_compiler_adapter.h"

#include <gtest/gtest.h>

#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset6.hpp>

using namespace vpux::driverCompilerAdapter;

class GraphTransformations_UnitTests : public ::testing::Test {
protected:
    std::shared_ptr<ov::Model> opset6mvn;

    void SetUp() override;
};

void GraphTransformations_UnitTests::SetUp() {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3, 4});
    const auto axesConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
    const auto mvn = std::make_shared<ov::opset6::MVN>(data, axesConst, false, 1e-5, ov::op::MVNEpsMode::OUTSIDE_SQRT);

    opset6mvn = std::make_shared<ov::Model>(ov::NodeVector{mvn}, ov::ParameterVector{data});
}

//------------------------------------------------------------------------------
using GraphTransformations_Serialize = GraphTransformations_UnitTests;

TEST_F(GraphTransformations_Serialize, canSerializeToIR) {
    OV_ASSERT_NO_THROW(serializeToIR(opset6mvn));
}

TEST_F(GraphTransformations_Serialize, resultOfSerializationIsNotEmpy) {
    const IR ir = serializeToIR(opset6mvn);

    EXPECT_GT(ir.xml.rdbuf()->in_avail(), 0);
    EXPECT_GT(ir.weights.rdbuf()->in_avail(), 0);
}
