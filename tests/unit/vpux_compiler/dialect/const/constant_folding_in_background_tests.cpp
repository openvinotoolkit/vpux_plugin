//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"

#include "common/utils.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include <gtest/gtest.h>
#include <thread>

using namespace vpux;
using namespace std::chrono_literals;

namespace {

using ContentAttrFunction = llvm::function_ref<std::pair<Const::ContentAttr, SmallVector<float>>(mlir::MLIRContext*)>;

std::pair<Const::ContentAttr, SmallVector<float>> createContentAttrSameTransformation(mlir::MLIRContext* ctx) {
    const size_t numElements = 100;
    const float baseValue = 1.0f;
    const auto baseType = mlir::RankedTensorType::get({numElements}, mlir::Float32Type::get(ctx));
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, baseValue);
    auto contentAttr = Const::ContentAttr::get(baseAttr);

    const size_t numTransformations = 5;
    for (size_t i = 0; i < numTransformations; ++i) {
        contentAttr = contentAttr.add(1.0);
    }

    const float expectedValue = baseValue + numTransformations;
    SmallVector<float> expectedFoldedResults(numElements, expectedValue);

    return std::make_pair(contentAttr, expectedFoldedResults);
}

std::pair<Const::ContentAttr, SmallVector<float>> createContentAttrMixedTransformations(mlir::MLIRContext* ctx) {
    const size_t numElements = 100;
    const float baseValue = 0.0f;
    const auto baseType = mlir::RankedTensorType::get({numElements}, mlir::Float32Type::get(ctx));
    const auto baseAttr = mlir::DenseElementsAttr::get(baseType, baseValue);
    auto contentAttr = Const::ContentAttr::get(baseAttr);
    contentAttr = contentAttr.padWithZero({10}, {10}).add(1.0).rescale(3.0).subview({0}, {numElements});

    const float expectedValue = 3.0f;
    SmallVector<float> expectedFoldedResults(numElements, expectedValue);

    return std::make_pair(contentAttr, expectedFoldedResults);
}

};  // namespace

class ConstantFoldingInBackground : public testing::TestWithParam<size_t> {};

void compile(mlir::MLIRContext* ctx, size_t numFoldingThreads, std::chrono::milliseconds sleepDuration,
             ContentAttrFunction contentAttrFn) {
    const auto collectStatistics = true;
    const auto foldingThreads = Const::initBackgroundConstantFoldingThreads(ctx, numFoldingThreads, collectStatistics);

    const auto [contentAttr, expectedValues] = contentAttrFn(ctx);

    std::this_thread::sleep_for(sleepDuration);

    auto result = contentAttr.fold();
    const auto resultValues = result.getValues<float>();
    EXPECT_EQ(resultValues.size(), expectedValues.size());
    for (auto values : zip(resultValues, expectedValues)) {
        const auto resultValue = std::get<0>(values);
        const auto expectedValue = std::get<1>(values);
        EXPECT_EQ(resultValue, expectedValue);
    }

    Const::stopBackgroundConstantFoldingThreads(ctx, foldingThreads, collectStatistics);
}

TEST_P(ConstantFoldingInBackground, CompilationFlow) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    vpux::registerCommonInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<Const::ConstDialect>();

    const auto numFoldingThreads = GetParam();
    const auto sleepDuration = 100ms;
    compile(&ctx, numFoldingThreads, sleepDuration, createContentAttrSameTransformation);
}

TEST_P(ConstantFoldingInBackground, MultipleCompilations) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    vpux::registerCommonInterfaces(registry);

    mlir::MLIRContext ctx1(registry);
    mlir::MLIRContext ctx2(registry);
    SmallVector<mlir::MLIRContext*> contexts = {&ctx1, &ctx2};
    for (auto ctx : contexts) {
        ctx->loadDialect<Const::ConstDialect>();
    }

    const auto numFoldingThreads = GetParam();
    const SmallVector<std::chrono::milliseconds> sleepDurations{100ms, 300ms};
    const SmallVector<ContentAttrFunction> contentAttrFns = {createContentAttrSameTransformation,
                                                             createContentAttrMixedTransformations};

    std::vector<std::thread> threads;
    for (size_t i = 0; i < contexts.size(); ++i) {
        const auto ctx = contexts[i];
        const auto sleepDuration = sleepDurations[i];
        const auto contentAttrFn = contentAttrFns[i];
        std::thread t([ctx, numFoldingThreads, sleepDuration, contentAttrFn]() {
            compile(ctx, numFoldingThreads, sleepDuration, contentAttrFn);
        });
        threads.push_back(std::move(t));
    }

    for (auto& t : threads) {
        t.join();
    }
}

std::vector<size_t> numThreads = {1, 2, 3, 4, 5};

INSTANTIATE_TEST_SUITE_P(MLIRThreading, ConstantFoldingInBackground, testing::ValuesIn(numThreads));

#endif
