//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

class MLIR_ConstDeclareSymbolTest : public MLIR_UnitBase {
public:
    MLIR_ConstDeclareSymbolTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }

    mlir::SymbolRefAttr getSymbol(ArrayRef<StringRef> names) {
        auto root = names.front();

        SmallVector<mlir::FlatSymbolRefAttr> nestedRefs;
        nestedRefs.reserve(names.size() - 1);

        for (auto name = names.begin() + 1; name != names.end(); name++) {
            nestedRefs.push_back(mlir::FlatSymbolRefAttr::get(&ctx, *name));
        }

        return mlir::SymbolRefAttr::get(&ctx, root, nestedRefs);
    }

    // Utility to convert our returned uses into an order-agnostic container because SymbolTables under
    // the hood use llvm::DenseMap.
    llvm::DenseSet<int64_t> getUidSet(ArrayRef<Const::DeclareOp> uses) {
        auto idRange = uses | transformed([](Const::DeclareOp op) {
                           return llvm::cast<mlir::IntegerAttr>(op->getAttr("uid")).getInt();
                       });
        return llvm::DenseSet<int64_t>(idRange.begin(), idRange.end());
    };

    mlir::MLIRContext ctx;
};

// E#136692: This test no longer works due to symbol lookup procedure being
// broken (ContentAttr property breaks the lookup's recursion).
TEST_F(MLIR_ConstDeclareSymbolTest, DISABLED_FindDeclareOpsUsingSymbol) {
    constexpr StringLiteral MLIR_SOURCE = R"(
module {
    const.Data @ov_bin_1 {
        const.Rodata @w_1 dense<0.0> : tensor<1xf32>
        const.Rodata @w_2 dense<0.0> : tensor<2xf32>
    }

    const.Data @ov_bin_2 {
        const.Rodata @w_1 dense<0.0> : tensor<3xf32>
        const.Rodata @w_2 dense<0.0> : tensor<4xf32>
    }

    func.func @f() -> () {
        %a = const.Declare {uid = 1} tensor<1xf32> = ref<@ov_bin_1::@w_1> : tensor<1xf32>
        %b = const.Declare {uid = 2} tensor<2xf32> = ref<@ov_bin_1::@w_2> : tensor<2xf32>
        return
    }

    func.func @g() -> () {
        %a = const.Declare {uid = 3} tensor<1xf32> = ref<@ov_bin_1::@w_1> : tensor<1xf32>
        %b = const.Declare {uid = 4} tensor<2xf32> = ref<@ov_bin_1::@w_2> : tensor<2xf32>
        %c = const.Declare {uid = 5} tensor<3xf32> = ref<@ov_bin_2::@w_1> : tensor<3xf32>
        %d = const.Declare {uid = 6} tensor<4xf32> = ref<@ov_bin_2::@w_2> : tensor<4xf32>
        %e = const.Declare {uid = 7} tensor<3xf32> = ref<@ov_bin_2::@w_1> : tensor<3xf32>
        %f = const.Declare {uid = 8} tensor<4xf32> = ref<@ov_bin_2::@w_2> : tensor<4xf32>
        return
    }

    func.func @h() -> () {
        %a = const.Declare {uid = 9} tensor<1xf32> = ref<@ov_bin_1::@w_1> : tensor<1xf32>
        return
    }
}
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(MLIR_SOURCE, &ctx);
    auto moduleOp = module.get();

    ASSERT_TRUE(moduleOp != nullptr);

    {
        auto uses = Const::getDeclareOpsUses(getSymbol({"ov_bin_1", "w_1"}), moduleOp);
        auto usesSet = getUidSet(uses);
        ASSERT_EQ(usesSet.size(), 3);
        ASSERT_TRUE(usesSet.contains(1));
        ASSERT_TRUE(usesSet.contains(3));
        ASSERT_TRUE(usesSet.contains(9));
    }

    {
        auto uses = Const::getDeclareOpsUses(getSymbol({"ov_bin_1", "w_2"}), moduleOp);
        auto usesSet = getUidSet(uses);
        ASSERT_EQ(uses.size(), 2);
        ASSERT_TRUE(usesSet.contains(2));
        ASSERT_TRUE(usesSet.contains(4));
    }

    {
        auto uses = Const::getDeclareOpsUses(getSymbol({"ov_bin_2", "w_1"}), moduleOp);
        auto usesSet = getUidSet(uses);
        ASSERT_EQ(uses.size(), 2);
        ASSERT_TRUE(usesSet.contains(5));
        ASSERT_TRUE(usesSet.contains(7));
    }

    {
        auto uses = Const::getDeclareOpsUses(getSymbol({"ov_bin_2", "w_2"}), moduleOp);
        auto usesSet = getUidSet(uses);
        ASSERT_EQ(usesSet.size(), 2);
        ASSERT_TRUE(usesSet.contains(6));
        ASSERT_TRUE(usesSet.contains(8));
    }
}
