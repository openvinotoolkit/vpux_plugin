//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/utils/attributes_properties_conversion.hpp"

using namespace vpux;

// We use IE::SparsityInfoOp for these tests as it has 3 simple attributes.

class AttributesPropertiesConversionTest : public MLIR_UnitBase {
public:
    void SetUp() override {
        ctx = std::make_unique<mlir::MLIRContext>(registry);
        builder = std::make_unique<mlir::OpBuilder>(ctx.get());
    }

    std::unique_ptr<mlir::MLIRContext> ctx;
    std::unique_ptr<mlir::OpBuilder> builder;
};

TEST_F(AttributesPropertiesConversionTest, SuccessfulConversion) {
    auto dict = mlir::DictionaryAttr::get(
            ctx.get(),
            mlir::ArrayRef<mlir::NamedAttribute>{{builder->getStringAttr("name"), builder->getStringAttr("MyNodeName")},
                                                 {builder->getStringAttr("inputId"), builder->getI32IntegerAttr(123)},
                                                 {builder->getStringAttr("ratio"), builder->getF32FloatAttr(4.2)}});

    auto props = toProperties<IE::SparsityInfoOp>(dict);

    ASSERT_EQ(props.getName(), builder->getStringAttr("MyNodeName"));
    ASSERT_EQ(props.getInputId(), builder->getI32IntegerAttr(123));
    ASSERT_EQ(props.getRatio(), builder->getF32FloatAttr(4.2));
}

TEST_F(AttributesPropertiesConversionTest, WrongInputIdAttributeType) {
    auto dict = mlir::DictionaryAttr::get(
            ctx.get(),
            mlir::ArrayRef<mlir::NamedAttribute>{{builder->getStringAttr("name"), builder->getStringAttr("MyNodeName")},
                                                 {builder->getStringAttr("inputId"), builder->getF32FloatAttr(123)},
                                                 {builder->getStringAttr("ratio"), builder->getF32FloatAttr(4.2)}});

    ASSERT_THROW(toProperties<IE::SparsityInfoOp>(dict), vpux::Exception);
}

TEST_F(AttributesPropertiesConversionTest, MissingRatioAttribute) {
    auto dict = mlir::DictionaryAttr::get(
            ctx.get(),
            mlir::ArrayRef<mlir::NamedAttribute>{{builder->getStringAttr("name"), builder->getStringAttr("MyNodeName")},
                                                 {builder->getStringAttr("inputId"), builder->getI32IntegerAttr(123)}});

    ASSERT_THROW(toProperties<IE::SparsityInfoOp>(dict), vpux::Exception);
}

TEST_F(AttributesPropertiesConversionTest, EmptyDictionary) {
    auto dict = mlir::DictionaryAttr::get(ctx.get());
    ASSERT_THROW(toProperties<IE::SparsityInfoOp>(dict), vpux::Exception);
}

TEST_F(AttributesPropertiesConversionTest, NotDictionaryAttr) {
    auto dict = builder->getF16FloatAttr(1.0);
    ASSERT_THROW(toProperties<IE::SparsityInfoOp>(dict), vpux::Exception);
}
