//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

class MLIR_MultiContentTest : public MLIR_UnitBase {
public:
    MLIR_MultiContentTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }

    template <class OpType>
    OpType findOpById(mlir::Operation* op, int64_t id) {
        OpType result;

        op->walk([&result, id](OpType op) {
            if (auto attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(op->getAttr("id")); attr != nullptr) {
                if (attr.getValue().getSExtValue() == id) {
                    result = op;
                }
            }
        });

        return result;
    }

    mlir::MLIRContext ctx;
};

TEST_F(MLIR_MultiContentTest, ParseValid) {
    constexpr StringLiteral MLIR_SOURCE = R"(
module {
    const.Data @Data {
        const.Rodata @weights_0 dense<1.0> : tensor<4x4xf32> {id = 1}
        const.Rodata @weights_1 dense<2.0> : tensor<4x4xf32> {id = 11}
    }

    const.BundleData @BundleStore {
        const.RodataBundle @bundle = [@Data::@weights_0, @Data::@weights_1, @Data::@weights_0] : tensor<4x4xf32> {id = 2}
    }

    func.func @f() -> tensor<4x4xf32> {
        %cst = const.MultiDeclare {id = 3} tensor<4x4xf32> = @BundleStore::@bundle : tensor<4x4xf32>
        return %cst : tensor<4x4xf32>
    }
}
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(MLIR_SOURCE, &ctx);
    auto moduleOp = module.get();

    ASSERT_TRUE(moduleOp != nullptr);

    auto rodataOp1 = findOpById<Const::RodataOp>(moduleOp, 1);
    auto rodataOp2 = findOpById<Const::RodataOp>(moduleOp, 11);
    auto rodataBundleOp = findOpById<Const::RodataBundleOp>(moduleOp, 2);
    auto multiDeclareOp = findOpById<Const::MultiDeclareOp>(moduleOp, 3);

    ASSERT_TRUE(rodataOp1 != nullptr);
    ASSERT_TRUE(rodataOp2 != nullptr);
    ASSERT_TRUE(rodataBundleOp != nullptr);
    ASSERT_TRUE(multiDeclareOp != nullptr);

    auto multiContentAttr = multiDeclareOp.dereferenceMultiContentSymbol();

    EXPECT_EQ(multiContentAttr.getBaseContent().size(), 3);

    EXPECT_EQ(multiContentAttr.getBaseContent()[0], rodataOp1.getContent());
    EXPECT_EQ(multiContentAttr.getBaseContent()[1], rodataOp2.getContent());
    EXPECT_EQ(multiContentAttr.getBaseContent()[2], rodataOp1.getContent());

    EXPECT_EQ(multiContentAttr.getFinalType(), rodataOp1.getContent().getType());
    EXPECT_EQ(multiContentAttr.getFinalType(), rodataOp2.getContent().getType());
    EXPECT_EQ(multiContentAttr.getFinalType(), rodataOp1.getContent().getType());
}

TEST_F(MLIR_MultiContentTest, DifferingBaseContentTypes) {
    const auto e1Type = mlir::RankedTensorType::get({4}, mlir::Float32Type::get(&ctx));
    const auto e1 = mlir::DenseElementsAttr::get(e1Type, 1.0f);

    const auto e2Type = mlir::RankedTensorType::get({2, 2}, mlir::Float32Type::get(&ctx));
    const auto e2 = mlir::DenseElementsAttr::get(e2Type, 1.0f);

    auto multiAttr = Const::MultiContentAttr::getChecked(mlir::UnknownLoc::get(&ctx), &ctx,
                                                         mlir::ArrayRef<mlir::ElementsAttr>{e1, e2},
                                                         mlir::ArrayRef<Const::TransformAttrInterface>{});
    EXPECT_EQ(multiAttr, nullptr);
}
