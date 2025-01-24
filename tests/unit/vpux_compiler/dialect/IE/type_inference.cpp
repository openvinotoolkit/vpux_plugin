//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include <gtest/gtest.h>

using namespace vpux;

namespace {

class MLIR_TypeInferenceTest : public MLIR_UnitBase {
public:
    MLIR_TypeInferenceTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect, mlir::func::FuncDialect, IE::IEDialect, mlir::tensor::TensorDialect>();
        listener = std::make_unique<mlir::OpBuilder::Listener>();
        builder = std::make_unique<mlir::OpBuilder>(&ctx, listener.get());
    }

    MLIR_TypeInferenceTest(const MLIR_TypeInferenceTest&) = delete;
    MLIR_TypeInferenceTest& operator=(const MLIR_TypeInferenceTest&) = delete;

    mlir::Value createOperand(ArrayRef<int64_t> shape, DimsOrder order) {
        return builder->create<mlir::tensor::EmptyOp>(builder->getUnknownLoc(), shape, mlir::Float32Type::get(&ctx),
                                                      getTensorAttr(&ctx, order, nullptr, nullptr));
    }

    mlir::DictionaryAttr createAttributes(ArrayRef<mlir::NamedAttribute> attributes) {
        return mlir::DictionaryAttr::get(&ctx, attributes);
    }

    mlir::MLIRContext ctx;
    std::unique_ptr<mlir::OpBuilder::Listener> listener;
    std::unique_ptr<mlir::OpBuilder> builder;
};

}  // namespace

TEST_F(MLIR_TypeInferenceTest, PadOp) {
    SmallVector<int64_t> shape{2, 2, 2, 2};
    auto operand = createOperand({2, 2, 2, 2}, DimsOrder::NHCW);

    IE::PadOp::Properties properties{};
    properties.pads_begin_attr = getIntArrayAttr<ArrayRef<int32_t>>(&ctx, {0, 0, 0, 0});
    properties.pads_end_attr = getIntArrayAttr<ArrayRef<int32_t>>(&ctx, {4, 0, 0, 0});
    properties.pad_value_attr = getFPAttr(&ctx, 0.0);
    properties.mode = IE::PadModeAttr::get(&ctx, IE::PadMode::CONSTANT);

    SmallVector<mlir::ShapedTypeComponents> typeComponents;
    ASSERT_TRUE(mlir::succeeded(IE::PadOp::inferReturnTypeComponents(&ctx, builder->getUnknownLoc(), {operand}, {},
                                                                     &properties, {}, typeComponents)));
    ASSERT_EQ(typeComponents.size(), 1);
    SmallVector<int64_t> expectedShape{6, 2, 2, 2};
    ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedShape));

    auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
    ASSERT_TRUE(tensorAttr != nullptr);
    auto expectedOrder = mlir::AffineMapAttr::get(DimsOrder::NHCW.toAffineMap(&ctx));
    EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
}

using BinaryOpTestParams = std::tuple<SmallVector<int64_t>, DimsOrder, SmallVector<int64_t>, DimsOrder>;

// This workaround is needed because using DimsOrder::NWHC to initialize global variables (as needed for gtest test
// instances) is undefined behaviour.
SmallVector<BinaryOpTestParams> getBinaryOpTestCases() {
    return {BinaryOpTestParams{{2, 2, 2, 2}, vpux::DimsOrder::NWHC, {2, 2}, vpux::DimsOrder::CN},
            BinaryOpTestParams{{2, 2}, vpux::DimsOrder::CN, {2, 2, 2, 2}, vpux::DimsOrder::NWHC},
            BinaryOpTestParams{{9, 9, 9, 9}, vpux::DimsOrder::HWCN, {9, 9, 9, 9}, vpux::DimsOrder::NCWH}};
}

TEST_F(MLIR_TypeInferenceTest, MultiplyOp) {
    for (auto [lhsShape, lhsOrder, rhsShape, rhsOrder] : getBinaryOpTestCases()) {
        bool takeLeft = lhsShape.size() >= rhsShape.size();
        auto expectedDims = takeLeft ? lhsShape : rhsShape;
        auto expectedOrder = mlir::AffineMapAttr::get((takeLeft ? lhsOrder : rhsOrder).toAffineMap(&ctx));

        auto lhs = createOperand(lhsShape, lhsOrder);
        auto rhs = createOperand(rhsShape, rhsOrder);

        IE::MultiplyOp::Properties properties{};
        properties.auto_broadcast = IE::AutoBroadcastTypeAttr::get(&ctx, IE::AutoBroadcastType::NUMPY);

        SmallVector<mlir::ShapedTypeComponents> typeComponents;
        ASSERT_TRUE(mlir::succeeded(IE::MultiplyOp::inferReturnTypeComponents(
                &ctx, builder->getUnknownLoc(), {lhs, rhs}, {}, &properties, {}, typeComponents)));
        ASSERT_EQ(typeComponents.size(), 1);
        ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

        auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
        ASSERT_TRUE(tensorAttr != nullptr);
        EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
    }
}

TEST_F(MLIR_TypeInferenceTest, DivideOp) {
    for (auto [lhsShape, lhsOrder, rhsShape, rhsOrder] : getBinaryOpTestCases()) {
        bool takeLeft = lhsShape.size() >= rhsShape.size();
        auto expectedDims = takeLeft ? lhsShape : rhsShape;
        auto expectedOrder = mlir::AffineMapAttr::get((takeLeft ? lhsOrder : rhsOrder).toAffineMap(&ctx));

        auto lhs = createOperand(lhsShape, lhsOrder);
        auto rhs = createOperand(rhsShape, rhsOrder);

        IE::DivideOp::Properties properties{};
        properties.auto_broadcast = IE::AutoBroadcastTypeAttr::get(&ctx, IE::AutoBroadcastType::NUMPY);

        SmallVector<mlir::ShapedTypeComponents> typeComponents;
        ASSERT_TRUE(mlir::succeeded(IE::DivideOp::inferReturnTypeComponents(&ctx, builder->getUnknownLoc(), {lhs, rhs},
                                                                            {}, &properties, {}, typeComponents)));
        ASSERT_EQ(typeComponents.size(), 1);
        ASSERT_EQ(typeComponents[0].getDims(), ArrayRef<int64_t>(expectedDims));

        auto tensorAttr = mlir::dyn_cast_or_null<TensorAttr>(typeComponents[0].getAttribute());
        ASSERT_TRUE(tensorAttr != nullptr);
        EXPECT_EQ(tensorAttr.getOrder(), expectedOrder);
    }
}
