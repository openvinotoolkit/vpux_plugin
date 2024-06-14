//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/simple_math.hpp"

#include <gtest/gtest.h>

TEST(MLIR_IntOrFloat, RawCtor) {
    vpux::IntOrFloat integer(5);
    EXPECT_TRUE(integer.isInt());
    EXPECT_EQ(5, integer.asInt());
    EXPECT_EQ(5.0f, integer.asFloat());

    vpux::IntOrFloat fp(1.5f);
    EXPECT_FALSE(fp.isInt());
    EXPECT_EQ(1.5f, fp.asFloat());
    EXPECT_EQ(1, fp.asInt()) << "rounding mode";
}

TEST(MLIR_IntOrFloat, ParseCtor) {
    vpux::IntOrFloat integer("5");
    EXPECT_TRUE(integer.isInt());
    EXPECT_EQ(5, integer.asInt());
    EXPECT_EQ(5.0f, integer.asFloat());

    vpux::IntOrFloat fp("1.5");
    EXPECT_FALSE(fp.isInt());
    EXPECT_EQ(1.5f, fp.asFloat());
    EXPECT_EQ(1, fp.asInt()) << "rounding mode";
}

TEST(MLIR_IntOrFloat, Operators) {
    vpux::IntOrFloat int4(4);
    vpux::IntOrFloat fp(0.5f);

    EXPECT_EQ(8, (int4 + int4).asInt());
    EXPECT_EQ(0, (int4 - int4).asInt());
    EXPECT_EQ(16, (int4 * int4).asInt());
    EXPECT_EQ(1, (int4 / int4).asInt());
    EXPECT_EQ(0, (int4 % int4).asInt());

    EXPECT_EQ(1.0f, (fp + fp).asFloat());
    EXPECT_EQ(0.0f, (fp - fp).asFloat());
    EXPECT_EQ(0.25f, (fp * fp).asFloat());
    EXPECT_EQ(1.0f, (fp / fp).asFloat());

    EXPECT_EQ(4.5f, (int4 + fp).asFloat());
    EXPECT_EQ(3.5f, (int4 - fp).asFloat());
    EXPECT_EQ(2.0f, (int4 * fp).asFloat());
    EXPECT_EQ(8.0f, (int4 / fp).asFloat());
}

TEST(MLIR_MathExpression, Constant) {
    vpux::MathExpression expr;

    expr.parse("1");
    EXPECT_EQ(1, expr.evaluate().asInt());

    expr.parse("3+4*5");
    EXPECT_EQ(3 + 4 * 5, expr.evaluate().asInt());

    expr.parse("(2+4)*0.5");
    EXPECT_EQ((2 + 4) / 2, expr.evaluate().asInt());
}

TEST(MLIR_MathExpression, WithVariables) {
    const vpux::MathExpression::VarMap vars = {{"a", "3"}, {"b", "5"}};

    vpux::MathExpression expr;
    expr.setVariables(vars);

    expr.parse("a+4*b");
    EXPECT_EQ(3 + 4 * 5, expr.evaluate().asInt());
}
