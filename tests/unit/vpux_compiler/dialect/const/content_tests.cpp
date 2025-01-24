//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include "common/utils.hpp"

#include <llvm/Support/raw_os_ostream.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectResourceBlobManager.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>
#include <vpux/compiler/utils/quantization.hpp>

#include <cassert>
#include <memory>
#include <numeric>
#include <sstream>

using namespace vpux;

namespace {
template <typename T>
std::vector<T> generateValues(size_t n) {
    std::vector<T> vals(n);
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<T>(i);
    }

    return vals;
}

template <typename T>
std::unique_ptr<T[]> generateValuesPointer(size_t n) {
    std::unique_ptr<T[]> data = std::make_unique<T[]>(n);
    T* vals = data.get();
    for (size_t i = 0; i < n; ++i) {
        vals[i] = static_cast<T>(i);
    }

    return data;
}

template <typename T>
void checkPaddedBuffer(const Const::Content& actual, const std::vector<T>& expVals, ShapeRef buf, ShapeRef pad, T zp,
                       size_t actOffset = 0, size_t originOffset = 0) {
    const int64_t IC = buf[Dim(0)];
    const int64_t IH = buf[Dim(1)];
    const int64_t IW = buf[Dim(2)];

    const int64_t PC = pad[Dim(0)];
    const int64_t PH = pad[Dim(1)];
    const int64_t PW = pad[Dim(2)];

    const auto actVals = actual.getValues<T>();
    for (int64_t c = 0; c < IC + 2 * PC; ++c) {
        for (int64_t h = 0; h < IH + 2 * PH; ++h) {
            for (int64_t w = 0; w < IW + 2 * PW; ++w) {
                const auto newIndex = w + h * (IW + 2 * PW) + c * (IW + 2 * PW) * (IH + 2 * PH) + actOffset;
                if (c < PC || c >= IC + PC || h < PH || h >= IH + PH || w < PW || w >= IW + PW) {
                    EXPECT_EQ(zp, actVals[newIndex]) << c << " " << h << " " << w;
                } else {
                    const auto origIndex = (w - PW) + (h - PH) * IW + (c - PC) * IW * IH + originOffset;
                    EXPECT_EQ(expVals[origIndex], actVals[newIndex]) << c << " " << h << " " << w;
                }
            }
        }
    }
}
}  // namespace

class MLIR_ConstContentAttrTest : public MLIR_UnitBase {
public:
    mlir::MLIRContext ctx;

public:
    MLIR_ConstContentAttrTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }
};

TEST_F(MLIR_ConstContentAttrTest, FromDenseElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    ASSERT_NE(static_cast<const void*>(baseAttr.getRawData().data()), static_cast<const void*>(vals.data()))
            << "Local data has to be copied inside DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, FromIRParsing) {
    const auto inputIR = R"(
        module @test {
            func.func @main() -> tensor<2x3x1x1xf32> {
                %cst = const.Declare tensor<2x3x1x1xf32> = dense_resource<splat_blob> : tensor<2x3x1x1xf32> isSplat

                return %cst : tensor<2x3x1x1xf32>
            }
        }

        {-#
        dialect_resources: {
            builtin: {
            splat_blob: "0x0400000001000000"
            }
        }
        #-}
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func->walk([&](Const::DeclareOp declareOp) {
        auto contentAttr = declareOp.getContentAttr();
        EXPECT_TRUE(contentAttr.isSplat());
    });
}

// Note: some networks (in tests at least) provide 0-element constants
TEST_F(MLIR_ConstContentAttrTest, FromEmptyDenseElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({0}, mlir::Float32Type::get(&ctx));

    const std::vector<char> empty{};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(empty.data(), empty.size()));
    ASSERT_TRUE(baseAttr.empty());

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    ASSERT_TRUE(content.getValues<float>().empty());
}

TEST_F(MLIR_ConstContentAttrTest, FromSplatDenseElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const float splatVal = 4.0f;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    EXPECT_EQ(content.getSplatValue<float>(), splatVal);

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], splatVal);
    }
}

namespace {
template <typename T>
ArrayRef<char> convertArrayRef(ArrayRef<T> typed) {
    return ArrayRef<char>(reinterpret_cast<const char*>(typed.data()), typed.size() * sizeof(T));
}
}  // namespace

TEST_F(MLIR_ConstContentAttrTest, FromDenseResourceElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    std::unique_ptr<float[]> data = generateValuesPointer<float>(baseType.getNumElements());

    const auto baseAttr = Const::createExternalConstContent(
            baseType, convertArrayRef(ArrayRef<float>(data.get(), baseType.getNumElements())),
            "FromDenseResourceElementsAttr");

    float* dataPtr = data.get();
    ASSERT_EQ(static_cast<const void*>(baseAttr.getRawHandle().getBlob()->getData().data()),
              static_cast<const void*>(dataPtr))
            << "Local data is not copied inside DenseResourceElementsAttr - unlike DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], dataPtr[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, FromDenseResourceElementsAttrNonOwning) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                            "FromDenseResourceElementsAttrNonOwning");

    ASSERT_EQ(static_cast<const void*>(baseAttr.getRawHandle().getBlob()->getData().data()),
              static_cast<const void*>(vals.data()))
            << "Local data is not copied inside DenseResourceElementsAttr - unlike DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, FromSplatDenseResourceElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const float splatVal = 4.0f;
    const std::vector<float> vals = {splatVal};
    const auto baseAttr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                            "FromSplatDenseResourceElementsAttr");

    ASSERT_EQ(static_cast<const void*>(baseAttr.getRawHandle().getBlob()->getData().data()),
              static_cast<const void*>(vals.data()))
            << "Local data is not copied inside DenseResourceElementsAttr - unlike DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    EXPECT_EQ(content.getSplatValue<float>(), splatVal);

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], splatVal);
    }
}

// Note: some networks (in tests at least) provide 0-element constants
TEST_F(MLIR_ConstContentAttrTest, FromEmptyDenseResourceElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({0}, mlir::Float32Type::get(&ctx));
    const std::vector<float> empty{};
    const auto baseAttr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(empty)),
                                                            "FromEmptyDenseResourceElementsAttr");

    ASSERT_TRUE(baseAttr.empty());

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    ASSERT_TRUE(content.getValues<float>().empty());
}

TEST_F(MLIR_ConstContentAttrTest, FromEqualElementsSplatDenseResourceElementsAttr) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const float splatVal = -4.0f;
    const std::vector<float> vals(baseType.getNumElements(), splatVal);
    const auto baseAttr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                            "FromEqualElementsSplatDenseResourceElementsAttr");

    ASSERT_EQ(static_cast<const void*>(baseAttr.getRawHandle().getBlob()->getData().data()),
              static_cast<const void*>(vals.data()))
            << "Local data is not copied inside DenseResourceElementsAttr - unlike DenseElementsAttr";

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    EXPECT_EQ(content.getSplatValue<float>(), splatVal);

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], splatVal);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ExpositionOnlyDenseResourceDuplicate) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));
    const auto vals = generateValues<float>(baseType.getNumElements());

    const auto attr = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                        "ExpositionOnlyDenseResourceDuplicate");
    const auto attrDuplicate = Const::createExternalConstContent(baseType, convertArrayRef(ArrayRef<float>(vals)),
                                                                 "ExpositionOnlyDenseResourceDuplicate");

    static_assert(!std::is_copy_constructible_v<mlir::AsmResourceBlob> &&
                  !std::is_copy_assignable_v<mlir::AsmResourceBlob>);

    ASSERT_NE(attrDuplicate.getRawHandle().getKey(), attr.getRawHandle().getKey())
            << "Two separately created DenseResourceElementsAttr objects could not share the same key - otherwise, "
               "revise nGraph constant sharing in NGraphImporter::parseNode()";

    const auto attrCopy = attr;
    ASSERT_EQ(attrCopy.getRawHandle().getKey(), attr.getRawHandle().getKey());
}

TEST_F(MLIR_ConstContentAttrTest, ConvertStorageElemType) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], i);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_FP32) {
    const auto baseType = mlir::RankedTensorType::get({8}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    const auto bufSize = bufSizeBytes / sizeof(float);
    std::vector<float> tempBuf(bufSize);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    EXPECT_EQ(vals.size(), bufSize);
    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(vals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_U8) {
    const auto baseType = mlir::RankedTensorType::get({8}, mlir::IntegerType::get(&ctx, 8));

    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    const auto contentAttr = Const::ContentAttr::get(baseAttr);
    ASSERT_NE(contentAttr, nullptr);
    EXPECT_EQ(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    EXPECT_EQ(vals.size(), tempBuf.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(vals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_I4) {
    const auto baseType = mlir::RankedTensorType::get({4}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {1,    // 0x1
                                       7,    // 0x7
                                       10,   // 0xA
                                       15};  // 0xF
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 4));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0x71, 0xFA};
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_I1) {
    const auto baseType = mlir::RankedTensorType::get({16}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {0, 0, 0, 1,   // 0x1 -> 0x8 packed
                                       0, 0, 1, 1,   // 0x3 -> 0xC packed
                                       1, 1, 1, 1,   // 0xF -> 0xF packed
                                       1, 1, 1, 0};  // 0xE -> 0x7 packed
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xC8, 0x7F};
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_FP32) {
    const auto baseType = mlir::RankedTensorType::get({8}, mlir::Float32Type::get(&ctx));

    const std::vector<float> vals = {1.0f};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    auto contentAttr = Const::ContentAttr::get(baseAttr);
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    const auto bufSize = bufSizeBytes / sizeof(float);
    std::vector<float> tempBuf(bufSize);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    EXPECT_EQ(bufSize, baseType.getNumElements());
    for (size_t i = 0; i < tempBuf.size(); ++i) {
        EXPECT_EQ(tempBuf[i], vals[0]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_U8) {
    const auto baseType = mlir::RankedTensorType::get({8}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {1};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    auto contentAttr = Const::ContentAttr::get(baseAttr);
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    EXPECT_EQ(bufSizeBytes, baseType.getNumElements());
    for (size_t i = 0; i < tempBuf.size(); ++i) {
        EXPECT_EQ(tempBuf[i], vals[0]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_I4) {
    const auto baseType = mlir::RankedTensorType::get({4}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {10};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 4));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xAA, 0xAA};
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Splat_CopyTo_I1) {
    const auto baseType = mlir::RankedTensorType::get({16}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {1};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xFF, 0xFF};
    EXPECT_EQ(expectedVals.size(), bufSizeBytes);
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CopyTo_I1_With_Enough_Buffer) {
    const auto baseType = mlir::RankedTensorType::get({16}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {0, 0, 0, 1,   // 0x1 -> 0x8 packed
                                       0, 0, 1, 1,   // 0x3 -> 0xC packed
                                       1, 1, 1, 1,   // 0xF -> 0xF packed
                                       1, 1, 1, 0};  // 0xE -> 0x7 packed
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup contentAttrSetup(baseType);
    contentAttrSetup = contentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    // Offer a large enough buffer to copy the data. The data not will be packed.
    int bufSizeBytes = 512;
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> UnpackedVals = {0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};

    for (size_t i = 0; i < UnpackedVals.size(); ++i) {
        EXPECT_EQ(UnpackedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CastElemType) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(getSInt32Type(&ctx));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], i);
    }
}

TEST_F(MLIR_ConstContentAttrTest, CastElemTypeSplat) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const float splatVal = 4.0f;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(getSInt32Type(&ctx));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    EXPECT_EQ(content.getSplatValue<int32_t>(), static_cast<int32_t>(splatVal));

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], static_cast<int32_t>(splatVal));
    }
}

TEST_F(MLIR_ConstContentAttrTest, CastElemTypeSubByte) {
    const auto baseType =
            mlir::RankedTensorType::get({3}, mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned));

    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<bool>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], (i == 0) ? false : true);
    }
}

TEST_F(MLIR_ConstContentAttrTest, Add) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto bias = 10;
    const auto vals = generateValues<float>(baseType.getNumElements());
    std::vector<float> expectedVals(vals.size());
    std::transform(vals.begin(), vals.end(), expectedVals.begin(), [&](float item) {
        return item + bias;
    });

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.add(bias);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, QuantCast) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, getUInt8Type(&ctx));

    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    const auto quantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  0.078431372549019607, 128, 0, 255);

    auto contentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<uint8_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], static_cast<uint8_t>(i));
    }
}

TEST_F(MLIR_ConstContentAttrTest, Dequantize) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, getSInt8Type(&ctx));

    std::vector<int8_t> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        // -127, 0, 127
        const auto choice = (static_cast<int>(i) % 3) - 1;
        vals[i] = static_cast<int8_t>(choice * 127);
    }

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    const double scale = 2.0 / 254.0;
    const auto quantType =
            mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, getSInt8Type(&ctx),
                                                   mlir::Float32Type::get(&ctx), scale, 0, -127, 127);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(quantType).dequantize();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        const auto choice = (static_cast<int>(i) % 3) - 1;
        EXPECT_FLOAT_EQ(contentVals[i], static_cast<float>(choice));
    }
}

TEST_F(MLIR_ConstContentAttrTest, Reshape) {
    const auto baseType = mlir::RankedTensorType::get({1, 9, 2}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.reshape({1, 3, 3, 2});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ReverseCWise) {
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t H = 4;
    const int64_t W = 5;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.reverse(Dim(1));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const auto origIndex = w + h * W + c * W * H + n * W * H * C;
                    const auto newIndex = (W - w - 1) + (H - h - 1) * W + c * W * H + n * W * H * C;
                    EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " " << h << " " << w;
                }
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, ReverseNWise) {
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t H = 4;
    const int64_t W = 5;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.reverse(Dim(0));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const auto origIndex = w + h * W + c * W * H + n * W * H * C;
                    const auto newIndex = (W - w - 1) + (H - h - 1) * W + (C - c - 1) * W * H + n * W * H * C;
                    EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " " << h << " " << w;
                }
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, Reverse_Splat) {
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t H = 4;
    const int64_t W = 5;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    constexpr float splatVal = 42.f;
    const std::vector<float> vals = {splatVal};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.reverse(Dim(0));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    EXPECT_EQ(content.getSplatValue<float>(), splatVal);
}

TEST_F(MLIR_ConstContentAttrTest, Reorder) {
    const int64_t N = 1;
    const int64_t C = 2;
    const int64_t H = 2;
    const int64_t W = 2;
    const auto baseType = mlir::RankedTensorType::get({N, C, H, W}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.reorder(DimsOrder::NHWC);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const auto origIndex = w + h * W + c * W * H + n * W * H * C;
                    const auto newIndex = c + w * C + h * C * W + n * C * W * H;
                    EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " " << h << " " << w;
                }
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, ReorderAfterReshape) {
    const int64_t N = 1;
    const int64_t C = 2;
    const int64_t H = 2;
    const int64_t W = 2;
    const auto baseType = mlir::RankedTensorType::get({N, C * H * W}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.reshape({N, C, H, W}).reorder(DimsOrder::NHWC);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const auto origIndex = w + h * W + c * W * H + n * W * H * C;
                    const auto newIndex = c + w * C + h * C * W + n * C * W * H;
                    EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " " << h << " " << w;
                }
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, Pad) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt32Type(&ctx));

    const auto vals = generateValues<int32_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const int64_t PC = 1;
    const int64_t PH = 1;
    const int64_t PW = 1;

    auto contentAttrSetup = baseContentAttrSetup.padWithZero({PC, PH, PW}, {PC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_GT(contentVals.size(), vals.size());

    checkPaddedBuffer<int32_t>(content, vals, {IC, IH, IW}, {PC, PH, PW}, 0);
}

TEST_F(MLIR_ConstContentAttrTest, PadSplat) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt32Type(&ctx));

    const int32_t splatVal = 42;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const int64_t PC = 1;
    const int64_t PH = 1;
    const int64_t PW = 1;

    auto contentAttrSetup = baseContentAttrSetup.padWithZero({PC, PH, PW}, {PC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    std::vector<int32_t> vals(baseType.getNumElements(), splatVal);
    checkPaddedBuffer<int32_t>(content, vals, {IC, IH, IW}, {PC, PH, PW}, 0);
}

TEST_F(MLIR_ConstContentAttrTest, PadUniformQuant) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const int64_t OC = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto zp = 128;
    const auto quantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  0.078431372549019607, zp, 0, 255);
    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    const int64_t PC = 2;
    const int64_t PH = 2;
    const int64_t PW = 2;

    auto contentAttrSetup = quantContentAttrSetup.padWithZero({0, PC, PH, PW}, {0, PC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_GT(contentVals.size(), vals.size());

    for (int64_t oc = 0; oc < OC; ++oc) {
        checkPaddedBuffer<float>(content, vals, {IC, IH, IW}, {PC, PH, PW}, zp,
                                 oc * (IC + 2 * PC) * (IW + 2 * PW) * (IH + 2 * PH), oc * IC * IW * IH);
    }
}

TEST_F(MLIR_ConstContentAttrTest, PadPerAxisQuant) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const int64_t OC = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto zp = 127;
    std::vector<double> scales(2, 0.5);
    std::vector<int64_t> zeroPoints{zp, zp};
    const auto quantType = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx), scales, zeroPoints, 0, 0, 255);

    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    const int64_t POC = 2;
    const int64_t PIC = 2;
    const int64_t PH = 2;
    const int64_t PW = 2;

    auto contentAttrSetup = quantContentAttrSetup.padWithZero({POC, PIC, PH, PW}, {POC, PIC, PH, PW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_GT(contentVals.size(), vals.size());

    std::vector<int64_t> expZP(POC, zp);
    expZP.insert(expZP.end(), zeroPoints.begin(), zeroPoints.end());
    expZP.insert(expZP.end(), POC, zp);

    const auto channelSize = IC * IW * IH;
    std::vector<float> expVals(channelSize * POC, zp);
    expVals.insert(expVals.end(), vals.begin(), vals.end());
    expVals.insert(expVals.end(), channelSize * POC, zp);

    for (int64_t oc = 0; oc < OC + 2 * POC; ++oc) {
        checkPaddedBuffer<float>(content, expVals, {IC, IH, IW}, {PIC, PH, PW}, expZP[oc],
                                 oc * (IC + 2 * PIC) * (IW + 2 * PW) * (IH + 2 * PH), oc * channelSize);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SubView) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt32Type(&ctx));

    const auto vals = generateValues<int32_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const int64_t OFF_C = 0;
    const int64_t OFF_H = 1;
    const int64_t OFF_W = 1;

    const int64_t OC = 1;
    const int64_t OH = 1;
    const int64_t OW = 1;

    auto contentAttrSetup = baseContentAttrSetup.subview({OFF_C, OFF_H, OFF_W}, {OC, OH, OW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_LT(contentVals.size(), vals.size());

    for (int64_t c = 0; c < OC; ++c) {
        for (int64_t h = 0; h < OH; ++h) {
            for (int64_t w = 0; w < OW; ++w) {
                const auto newIndex = w + h * OW + c * OW * OH;
                const auto origIndex = (w + OFF_W) + (h + OFF_H) * IW + (c + OFF_C) * IW * IH;
                EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << c << " " << h << " " << w;
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, SubViewI1) {
    const int64_t IC = 1;
    const int64_t IH = 7;
    const int64_t IW = 5;

    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getInt1Type(&ctx));
    auto vals = SmallVector<bool>(baseType.getNumElements());
    for (auto index : irange(vals.size())) {
        vals[index] = static_cast<bool>(index % 2);
    }

    const auto valsArrayRef = ArrayRef<bool>(vals);
    const auto baseAttr = Const::createConstContent(baseType, valsArrayRef);

    Const::ContentSetup baseContentAttrSetup(baseType);

    const int64_t OFF_C = 0;
    const int64_t OFF_H = 1;
    const int64_t OFF_W = 1;

    const int64_t OC = 1;
    const int64_t OH = 1;
    const int64_t OW = 1;

    auto contentAttrSetup = baseContentAttrSetup.subview({OFF_C, OFF_H, OFF_W}, {OC, OH, OW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    // getValues is not realized for sub-byte types
    // therefore access through raw buffer
    auto contentBuf = content.getRawStorageBuf();
    auto contentData = contentBuf.data();

    for (int64_t c = 0; c < OC; ++c) {
        for (int64_t h = 0; h < OH; ++h) {
            for (int64_t w = 0; w < OW; ++w) {
                const auto newIndex = w + h * OW + c * OW * OH;
                const auto origIndex = (w + OFF_W) + (h + OFF_H) * IW + (c + OFF_C) * IW * IH;
                auto inputCoord = std::div(newIndex, checked_cast<size_t>(CHAR_BIT));
                bool bitValue = contentData[inputCoord.quot] & (1 << inputCoord.rem);
                EXPECT_EQ(bitValue, vals[origIndex]) << c << " " << h << " " << w;
            }
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, SubViewSplat) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt32Type(&ctx));

    const int32_t splatVal = 42;
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(splatVal));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const int64_t OFF_C = 0;
    const int64_t OFF_H = 1;
    const int64_t OFF_W = 1;

    const int64_t OC = 1;
    const int64_t OH = 1;
    const int64_t OW = 1;

    auto contentAttrSetup = baseContentAttrSetup.subview({OFF_C, OFF_H, OFF_W}, {OC, OH, OW});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    EXPECT_EQ(content.getSplatValue<int32_t>(), splatVal);
}

TEST_F(MLIR_ConstContentAttrTest, BitPack) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt8Type(&ctx));

    const auto vals = std::vector<int8_t>{1, 2, 3, 4, 5, 6};
    const auto expectedResult = std::vector<int8_t>{0x21, 0x43, 0x65};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto bitWidth = 4;
    auto contentAttrSetup = baseContentAttrSetup.bitPack(bitWidth);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    const auto ndBaseType = baseType.cast<vpux::NDTypeInterface>();
    const auto expectedType = ndBaseType.changeElemType(
            mlir::IntegerType::get(ndBaseType.getContext(), bitWidth, mlir::IntegerType::SignednessSemantics::Signed));
    EXPECT_EQ(content.getType(), expectedType);
    EXPECT_EQ(content.getType(), contentAttr.getType());

    std::vector<int8_t> actVals(vals.size() / 2, 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, BitPackQuant) {
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    ctx.loadDialect<mlir::quant::QuantizationDialect>();
    const auto baseType = mlir::RankedTensorType::get({IC, IH, IW}, getSInt8Type(&ctx));

    const auto vals = std::vector<int8_t>{1, 2, 3, 1, 2, 3};
    const auto expectedResult = std::vector<int8_t>{0x21, 0x13, 0x32};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const double scale = 1.0;
    const int64_t zeroPoint = 0;
    const int64_t storageTypeMin = -4;
    const int64_t storageTypeMax = 3;
    const auto quantType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed,
                                                                  getSInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  scale, zeroPoint, storageTypeMin, storageTypeMax);
    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    const auto bitWidth = 4;
    auto contentAttrSetup = quantContentAttrSetup.bitPack(bitWidth);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    const auto expectedQuantType = mlir::quant::UniformQuantizedType::get(
            mlir::quant::QuantizationFlags::Signed, mlir::IntegerType::get(&ctx, bitWidth, mlir::IntegerType::Signed),
            mlir::Float32Type::get(&ctx), scale, zeroPoint, storageTypeMin, storageTypeMax);
    const auto expectedType = baseType.cast<vpux::NDTypeInterface>().changeElemType(expectedQuantType);
    EXPECT_EQ(content.getType(), expectedType);
    EXPECT_EQ(content.getType(), contentAttr.getType());

    std::vector<int8_t> actVals(vals.size() / 2, 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, Transpose) {
    const int64_t N = 512;
    const int64_t C = 40;
    const auto baseType = mlir::RankedTensorType::get({N, C}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto permutationMap = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{1, 0}, &ctx);
    const auto orderAttr = DimsOrder::fromAffineMap(permutationMap);
    auto contentAttrSetup = baseContentAttrSetup.transpose(orderAttr);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const auto origIndex = n * C + c * 1;
            const auto newIndex = n * 1 + c * N;
            EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " ";
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, MemPermute) {
    const int64_t N = 512;
    const int64_t C = 40;
    const auto baseType = mlir::RankedTensorType::get({N, C}, mlir::Float32Type::get(&ctx));

    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto permutationMap = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{1, 0}, &ctx);
    const auto memPermAttr = DimsOrder::fromAffineMap(permutationMap);

    const auto dstOrderMap = mlir::AffineMap::getMultiDimIdentityMap(permutationMap.getNumDims(), &ctx);
    const auto dstOrderAttr = DimsOrder::fromAffineMap(dstOrderMap);

    auto contentAttrSetup = baseContentAttrSetup.memPermute(dstOrderAttr, memPermAttr);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const auto origIndex = n * C + c * 1;
            const auto newIndex = n * 1 + c * N;
            EXPECT_EQ(contentVals[newIndex], vals[origIndex]) << n << " " << c << " ";
        }
    }
}

TEST_F(MLIR_ConstContentAttrTest, BitPackIsLast) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, getSInt8Type(&ctx));

    const auto vals = std::vector<int8_t>{1, 2, 3, 4, 5, 6};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto bitWidth = 4;
    auto contentAttrSetup = baseContentAttrSetup.bitPack(bitWidth);

    auto addBitPackAttrSetup = contentAttrSetup.clone().add(17.f);
    const auto addBitPackTransformations = addBitPackAttrSetup.getTransformations();
    ASSERT_EQ(addBitPackTransformations.size(), 2);
    EXPECT_NE(addBitPackTransformations[0].dyn_cast<Const::AddAttr>(), nullptr);
    EXPECT_NE(addBitPackTransformations[1].dyn_cast<Const::BitPackAttr>(), nullptr);

    const auto addBroadcastBitPackAttr = Const::ContentAttr::get(baseAttr, addBitPackAttrSetup.broadcast(Dim(1), 42));
    const auto addBroadcastBitPackTransformations = addBroadcastBitPackAttr.getTransformations();
    ASSERT_EQ(addBroadcastBitPackTransformations.size(), 3);
    EXPECT_NE(addBroadcastBitPackTransformations[0].dyn_cast<Const::AddAttr>(), nullptr);
    EXPECT_NE(addBroadcastBitPackTransformations[1].dyn_cast<Const::BroadcastAttr>(), nullptr);
    EXPECT_NE(addBroadcastBitPackTransformations[2].dyn_cast<Const::BitPackAttr>(), nullptr);

    // Expects input type to be quantized, while the input type will be SI8
    EXPECT_ANY_THROW(std::ignore = Const::ContentAttr::get(baseAttr, contentAttrSetup.clone().dequantize()));

    EXPECT_NO_THROW(std::ignore = contentAttrSetup.clone().castElemType(getSInt32Type(&ctx)));
    EXPECT_NO_THROW(std::ignore = contentAttrSetup.clone().padWithZero({0, 1, 2, 3}, {0, 3, 2, 1}));
    EXPECT_NO_THROW(std::ignore = contentAttrSetup.clone().reorder(DimsOrder::NHWC));
    EXPECT_NO_THROW(std::ignore = contentAttrSetup.clone().rescale(19.f));
    EXPECT_NO_THROW(std::ignore = contentAttrSetup.clone().reshape(Shape({IN * IC, IH, IW})));
    EXPECT_NO_THROW(std::ignore = contentAttrSetup.clone().subview({0, 0, 0, 0}, {IN, IC, IH, IW}));
    EXPECT_NO_THROW(std::ignore = contentAttrSetup.clone().transpose(DimsOrder::NHWC));

    // Inserting another transformation that has the LAST position requirement
    EXPECT_ANY_THROW(
            std::ignore = contentAttrSetup.clone().swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX)));

    const auto quantType =
            mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, getSInt8Type(&ctx),
                                                   mlir::Float32Type::get(&ctx), 0.078431372549019607, 0, -4, 3);
    auto quantContentAttrSetup = contentAttrSetup.clone().castElemType(quantType);
}

TEST_F(MLIR_ConstContentAttrTest, ExpandDilated) {
    const int64_t OC = 2;
    const int64_t IC = 2;
    const int64_t KY = 5;
    const int64_t KX = 5;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, KY, KX}, getSInt32Type(&ctx));

    const auto vals = generateValues<int32_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const int64_t dilY = 3;
    const int64_t dilX = 3;

    auto contentAttrSetup = baseContentAttrSetup.expandDilated({dilY, dilX});

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const int64_t dKY = KY + (KY - 1) * (dilY - 1);
    const int64_t dKX = KX + (KX - 1) * (dilX - 1);
    std::vector<int8_t> expectedVals(OC * IC * dKY * dKX, 0);

    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t ic = 0; ic < IC; ++ic) {
            for (int64_t ky = 0; ky < KY; ++ky) {
                for (int64_t kx = 0; kx < KX; ++kx) {
                    const auto dky = ky + (dilY - 1) * ky;
                    const auto dkx = kx + (dilX - 1) * kx;
                    const auto expectedValsInd = dkx + dky * dKX + ic * dKX * dKY + oc * dKX * dKY * IC;
                    const auto valsInd = kx + ky * KX + ic * KX * KY + oc * KX * KY * IC;
                    expectedVals[expectedValsInd] = vals[valsInd];
                }
            }
        }
    }

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ExpandDilated_Splat) {
    const int64_t OC = 2;
    const int64_t IC = 2;
    const int64_t KY = 5;
    const int64_t KX = 5;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, KY, KX}, getSInt32Type(&ctx));

    const auto vals = std::vector<int32_t>(baseType.getNumElements(), 42);
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const int64_t dilY = 3;
    const int64_t dilX = 3;

    auto contentAttrSetup = baseContentAttrSetup.expandDilated({dilY, dilX});
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));

    ASSERT_NE(contentAttr, nullptr);
    EXPECT_NE(contentAttr.getType(), baseType);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const int64_t dKY = KY + (KY - 1) * (dilY - 1);
    const int64_t dKX = KX + (KX - 1) * (dilX - 1);
    std::vector<int8_t> expectedVals(OC * IC * dKY * dKX, 0);

    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t ic = 0; ic < IC; ++ic) {
            for (int64_t ky = 0; ky < KY; ++ky) {
                for (int64_t kx = 0; kx < KX; ++kx) {
                    const auto dky = ky + (dilY - 1) * ky;
                    const auto dkx = kx + (dilX - 1) * kx;
                    const auto expectedValsInd = dkx + dky * dKX + ic * dKX * dKY + oc * dKX * dKY * IC;
                    const auto valsInd = kx + ky * KX + ic * KX * KY + oc * KX * KY * IC;
                    expectedVals[expectedValsInd] = vals[valsInd];
                }
            }
        }
    }

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, GetSparsityMap) {
    const int64_t OC = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, getUInt8Type(&ctx));

    const auto vals = std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 0, 13, 14, 15};
    // expected result binary form:        0  1  1  1  1  1  1  1 |1  1  0   1   0  1   1   1
    // expected result HEX form:               E            F     |     B             E
    const auto expectedResult = std::vector<uint8_t>{0xFE, 0xEB};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    auto contentAttrSetup = baseContentAttrSetup.getSparsityMap();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    const auto ndBaseType = baseType.cast<vpux::NDTypeInterface>();
    const auto expectedType = ndBaseType.changeShapeElemType(
            Shape({OC, 1, 1, 128}), mlir::IntegerType::get(ndBaseType.getContext(), 1, mlir::IntegerType::Signless));
    EXPECT_EQ(content.getType(), expectedType);
    EXPECT_EQ(content.getType(), contentAttr.getType());

    const auto valsSize = static_cast<size_t>(vals.size() / 8);
    const auto alignment = static_cast<size_t>(16);
    const auto alignedValsSize = vpux::alignValUp(valsSize, alignment);
    std::vector<uint8_t> actVals(alignedValsSize, 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.begin() + valsSize, expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, GetSparsityMapQuantized) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const int64_t OC = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({OC, IC, IH, IW}, getUInt8Type(&ctx));

    // source float values: {0, -7, -6, -5, -4, -3, -2, -1,  0, 1, 0, 3, 0, 5, 6, 7};
    const double scale = 1.0;
    const int64_t zeroPoint = 7;
    const int64_t storageTypeMin = 0;
    const int64_t storageTypeMax = 14;
    // apply quantization to src values
    const auto vals = std::vector<uint8_t>{7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 10, 7, 12, 13, 14};

    // expected result binary form:        0  1  1  1  1  1  1  1 |0  1  0  1   0   1   1   1
    // expected result HEX form:                E          F      |     A             E
    const auto expectedResult = std::vector<uint8_t>{0xFE, 0xEA};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto quantType =
            mlir::quant::UniformQuantizedType::get(0, baseType.getElementType(), mlir::Float32Type::get(&ctx), scale,
                                                   zeroPoint, storageTypeMin, storageTypeMax);
    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    auto contentAttrSetup = quantContentAttrSetup.getSparsityMap();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    const auto ndBaseType = baseType.cast<vpux::NDTypeInterface>();
    const auto expectedType = ndBaseType.changeShapeElemType(
            Shape({OC, 1, 1, 128}), mlir::IntegerType::get(ndBaseType.getContext(), 1, mlir::IntegerType::Signless));
    EXPECT_EQ(content.getType(), expectedType);
    EXPECT_EQ(content.getType(), contentAttr.getType());

    const auto valsSize = static_cast<size_t>(vals.size() / 8);
    const auto alignment = static_cast<size_t>(16);
    const auto alignedValsSize = vpux::alignValUp(valsSize, alignment);
    std::vector<uint8_t> actVals(alignedValsSize, 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.begin() + valsSize, expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, Sparsify) {
    const int64_t IN = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, getUInt8Type(&ctx));

    const auto vals = std::vector<uint8_t>{0,  1, 2,  3,  4,  5, 6,  7,  8,  9,  0,  11, 0, 13, 14, 15,
                                           16, 0, 18, 19, 20, 0, 22, 23, 24, 25, 26, 0,  0, 29, 30, 31};
    const auto expectedResult = std::vector<uint8_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  11, 13, 14, 15, 0, 0, 0,
                                                     16, 18, 19, 20, 22, 23, 24, 25, 26, 29, 30, 31, 0,  0, 0, 0};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    auto contentAttrSetup = baseContentAttrSetup.sparsify(false);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), contentAttr.getType()) << "these types are different, this is by design";

    std::vector<uint8_t> actVals(vals.size(), 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, Sparsify_True) {
    const int64_t IN = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, getUInt8Type(&ctx));

    const auto vals = std::vector<uint8_t>{0,  1, 2,  3,  4,  5, 6,  7,  8,  9,  0,  11, 0, 13, 14, 15,
                                           16, 0, 18, 19, 20, 0, 22, 23, 24, 25, 26, 0,  0, 29, 30, 31};
    const auto expectedResult = std::vector<uint8_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  11, 13, 14, 15, 0, 0, 0,
                                                     16, 18, 19, 20, 22, 23, 24, 25, 26, 29, 30, 31, 0,  0, 0, 0};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto numElemsPerOC =
            vpux::countNonSparseElementsPerOC(Const::ContentAttr::get(baseAttr).fold(), baseType.getElementType());
    const auto numElemsPerOCType =
            mlir::RankedTensorType::get({static_cast<int64_t>(numElemsPerOC.size())}, getInt64Type(&ctx));
    const auto numElemsAttr = Const::createConstContent(numElemsPerOCType, ArrayRef(numElemsPerOC));
    const auto contentAttr = Const::ContentAttr::get(baseAttr, baseContentAttrSetup.sparsify(true, numElemsAttr));
    ASSERT_NE(contentAttr, nullptr);

    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType()) << "these types are the same";

    std::vector<uint8_t> actVals(vals.size(), 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, SparsifyQuantized) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const int64_t IN = 2;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 8;

    // source float values:{0, -15, -14,  -13,  -12,  -11, -10,  -9,  -8,  -7,  0,  -5, 0, -3, -2, -1,
    //                      0,   0,   2,    3,    4,    5,   6,   7,   8,   9, 10,   0, 0, 13, 14, 15};
    const double scale = 1.0;
    const int64_t zeroPoint = 16;
    const int64_t storageTypeMin = 0;
    const int64_t storageTypeMax = 31;
    // apply quantization to src values
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, getUInt8Type(&ctx));
    const auto vals = std::vector<uint8_t>{16, 1,  2,  3,  4,  5,  6,  7,  8,  9,  16, 11, 16, 13, 14, 15,
                                           16, 16, 18, 19, 20, 16, 22, 23, 24, 25, 26, 16, 16, 29, 30, 31};
    const auto expectedResult = std::vector<uint8_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  11, 13, 14, 15, 0, 0, 0,
                                                     18, 19, 20, 22, 23, 24, 25, 26, 29, 30, 31, 0,  0,  0, 0, 0};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto quantType =
            mlir::quant::UniformQuantizedType::get(0, baseType.getElementType(), mlir::Float32Type::get(&ctx), scale,
                                                   zeroPoint, storageTypeMin, storageTypeMax);
    auto quantContentAttrSetup = baseContentAttrSetup.castElemType(quantType);

    auto contentAttrSetup = quantContentAttrSetup.sparsify(false);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_NE(content.getType(), contentAttr.getType()) << "these types are different, this is by design";

    std::vector<uint8_t> actVals(vals.size(), 0);
    auto buf = MutableArrayRef(reinterpret_cast<char*>(actVals.data()), actVals.size());
    content.copyTo(buf);

    EXPECT_EQ(actVals.size(), expectedResult.size());
    EXPECT_TRUE(std::equal(actVals.begin(), actVals.end(), expectedResult.begin()));
}

TEST_F(MLIR_ConstContentAttrTest, PositionRequirement) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 3;
    const int64_t IW = 3;
    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::Float32Type::get(&ctx));
    Const::ContentSetup baseContentAttrSetup(baseType);

    // Inserting a transformation that has no position requirement
    auto contentAttrSetup1 = baseContentAttrSetup.rescale(10.0);

    // Inserting a transformation that has the LAST position requirement
    auto contentAttrSetup2 = contentAttrSetup1.swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX));

    // Inserting a transformation that has the PREFERRED_LAST position requirement
    auto contentAttrSetup3 = contentAttrSetup2.sparsify(false);

    // Inserting another transformation that has no position requirement
    auto contentAttrSetup4 = contentAttrSetup3.castElemType(mlir::Float16Type::get(&ctx));

    const auto finalTransformations = contentAttrSetup4.getTransformations();
    EXPECT_EQ(finalTransformations.size(), 4);
    EXPECT_EQ(finalTransformations[0].getTransformationName(), "Rescale");
    EXPECT_EQ(finalTransformations[1].getTransformationName(), "CastElemType");
    EXPECT_EQ(finalTransformations[2].getTransformationName(), "Sparsify");
    EXPECT_EQ(finalTransformations[3].getTransformationName(), "SwizzleConstant");
}

TEST_F(MLIR_ConstContentAttrTest, ChangeShapeAndElemType) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto baseType = mlir::RankedTensorType::get({2, 1, 1, 8}, getUInt8Type(&ctx));
    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto quantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float32Type::get(&ctx),
                                                                  0.078431372549019607, 128, 0, 255);
    auto quantContentAttrSetup = baseContentAttrSetup.changeShapeAndElemType({1, 2, 1, 8}, quantType);

    auto quantContentAttr = Const::ContentAttr::get(baseAttr, std::move(quantContentAttrSetup));
    const auto content = quantContentAttr.fold();
    EXPECT_EQ(content.getType(), quantContentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(quantContentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<uint8_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ChangeShapeAndElemTypePerAxisQuant) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto baseType = mlir::RankedTensorType::get({2, 1, 1, 8}, getUInt8Type(&ctx));
    const auto vals = generateValues<uint8_t>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseType);

    const auto zp = 127;
    std::vector<double> scales(2, 0.5);
    std::vector<int64_t> zeroPoints{zp, zp};
    int32_t quantizedDim1 = 0;
    const auto quantElemType1 = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx), scales, zeroPoints, quantizedDim1, 0, 255);
    auto quantContentAttrSetup1 = baseContentAttrSetup.castElemType(quantElemType1);

    int32_t quantizedDim2 = 1;
    const auto quantElemType2 = mlir::quant::UniformQuantizedPerAxisType::get(
            0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx), scales, zeroPoints, quantizedDim2, 0, 255);
    auto quantContentAttrSetup2 = quantContentAttrSetup1.clone().changeShapeAndElemType({1, 2, 1, 8}, quantElemType2);

    auto quantContentAttr1 = Const::ContentAttr::get(baseAttr, std::move(quantContentAttrSetup1));
    auto quantContentAttr2 = Const::ContentAttr::get(baseAttr, std::move(quantContentAttrSetup2));
    const auto content1 = quantContentAttr1.fold();
    const auto content2 = quantContentAttr2.fold();
    EXPECT_EQ(content1.getType(), quantContentAttr1.getType());
    EXPECT_EQ(content2.getType(), quantContentAttr2.getType());
    EXPECT_NE(content1.getType(), content2.getType());
    EXPECT_FALSE(content1.isSplat());
    EXPECT_EQ(quantContentAttr1.isSplat(), content1.isSplat());
    EXPECT_FALSE(content2.isSplat());
    EXPECT_EQ(quantContentAttr2.isSplat(), content2.isSplat());

    const auto contentVals1 = content1.getValues<uint8_t>();
    const auto contentVals2 = content2.getValues<uint8_t>();
    EXPECT_EQ(contentVals1.size(), vals.size());
    EXPECT_EQ(contentVals2.size(), vals.size());

    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(contentVals1[i], vals[i]);
        EXPECT_EQ(contentVals2[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ChangeShapeAndElemTypeFloat) {
    const auto baseType = mlir::RankedTensorType::get({2, 1, 1, 8}, mlir::Float32Type::get(&ctx));
    const auto vals = generateValues<float>(baseType.getNumElements());
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseType);

    auto newContentAttrSetup = baseContentAttrSetup.changeShapeAndElemType({1, 2, 1, 8}, getSInt32Type(&ctx));

    auto newContentAttr = Const::ContentAttr::get(baseAttr, std::move(newContentAttrSetup));
    const auto content = newContentAttr.fold();
    EXPECT_EQ(content.getType(), newContentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(newContentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<int32_t>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

#ifdef BACKGROUND_FOLDING_ENABLED

TEST_F(MLIR_ConstContentAttrTest, GetTransformationsRange) {
    const auto baseType = mlir::RankedTensorType::get({10}, mlir::Float32Type::get(&ctx));
    Const::ContentSetup contentAttrSetup(baseType);

    contentAttrSetup = contentAttrSetup.reshape({2, 5}).subview({0, 0}, {1, 5}).add(1.0);

    auto transformations = contentAttrSetup.getTransformations();
    ASSERT_EQ(transformations.size(), 3);

    auto subviewTransformation = transformations[1];
    ASSERT_NE(subviewTransformation, nullptr);

    auto headTransformations = vpux::Const::BackgroundConstantFolding::stripTransformationsFrom(
            contentAttrSetup.getTransformations(), subviewTransformation);
    ASSERT_EQ(headTransformations.size(), 1);
    EXPECT_EQ(headTransformations[0].getTransformationName(), "Reshape");

    auto tailTransformations = vpux::Const::BackgroundConstantFolding::getLastTransformationsFrom(
            contentAttrSetup.getTransformations(), subviewTransformation);
    ASSERT_EQ(tailTransformations.size(), 2);
    EXPECT_EQ(tailTransformations[0].getTransformationName(), "SubView");
    EXPECT_EQ(tailTransformations[1].getTransformationName(), "Add");
}

#endif

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_SubBytes_I1) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 4;
    const int64_t IW = 4;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {0, 0, 0, 1,   // 0x1 -> 0x8 packed
                                       0, 0, 1, 1,   // 0x3 -> 0xC packed
                                       1, 1, 1, 1,   // 0xF -> 0xF packed
                                       1, 1, 1, 0};  // 0xE -> 0x7 packed
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseType);

    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));
    auto contentAttrSetup1 = contentAttrSetup.clone().swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    auto contentAttr1 = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup1));
    const auto content = contentAttr1.fold();
    EXPECT_EQ(content.getType(), contentAttr1.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(VPU::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));
    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));
    const std::vector<uint8_t> expectedVals = {0xC8, 0x7F};

    EXPECT_EQ(acheAlignSize, static_cast<size_t>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_SubBytes_I4) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const auto vals = std::vector<uint8_t>{1, 7,     // 0x17
                                           10, 15};  // 0xAF
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 4));

    auto contentAttrSetup1 = contentAttrSetup.clone().swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX));

    auto contentAttr1 = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup1));
    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr1.fold();
    EXPECT_EQ(content.getType(), contentAttr1.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(VPU::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0x71, 0xFA};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_U8) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const auto vals = std::vector<uint8_t>{255, 100, 0, 1};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseType);

    auto contentAttrSetup = baseContentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(VPU::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));
    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));
    const std::vector<uint8_t> expectedVals = {255, 100, 0, 1};

    EXPECT_EQ(acheAlignSize, static_cast<size_t>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_FP32) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const auto vals = std::vector<float>{700, 800, 900, 900};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseType);

    auto contentAttrSetup = baseContentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());
    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(VPU::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));
    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<float> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));
    const std::vector<float> expectedVals = {700, 800, 900, 900};

    EXPECT_EQ(acheAlignSize, static_cast<size_t>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_SubBytes_Splat_I1) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 4;
    const int64_t IW = 4;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {1};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 1));

    const auto contentType = Const::inferFinalType(baseType, contentAttrSetup.getTransformations());
    ASSERT_NE(contentType, nullptr);

    auto contentAttrSetup1 = contentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX));

    auto contentAttr1 = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup1));
    const auto content = contentAttr1.fold();
    EXPECT_EQ(content.getType(), contentAttr1.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr1.isSplat(), content.isSplat());

    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(VPU::ArchKind::NPU37XX);
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xFF, 0xFF};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_SubBytes_Splat_I4) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 4;
    const int64_t IW = 4;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {10};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    auto contentAttrSetup = baseContentAttrSetup.castElemType(mlir::IntegerType::get(&ctx, 4));

    const auto contentType = Const::inferFinalType(baseType, contentAttrSetup.getTransformations());
    ASSERT_NE(contentType, nullptr);

    auto contentAttrSetup1 = contentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX));

    auto contentAttr1 = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup1));
    const auto content = contentAttr1.fold();
    EXPECT_EQ(content.getType(), contentAttr1.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr1.isSplat(), content.isSplat());

    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(VPU::ArchKind::NPU37XX);
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_Splat_U8) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::IntegerType::get(&ctx, 8));

    const std::vector<uint8_t> vals = {10};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    auto contentAttrSetup = baseContentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(VPU::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<uint8_t> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<uint8_t> expectedVals = {0x0A, 0x0A, 0x0A, 0x0A};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));
    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, SwizzleConstant_Splat_FP32) {
    const int64_t IN = 1;
    const int64_t IC = 1;
    const int64_t IH = 2;
    const int64_t IW = 2;

    const auto baseType = mlir::RankedTensorType::get({IN, IC, IH, IW}, mlir::Float32Type::get(&ctx));

    const std::vector<float> vals = {10};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    auto contentAttrSetup = baseContentAttrSetup.swizzleConstant(5, static_cast<uint64_t>(VPU::ArchKind::NPU37XX));

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    VPU::ArchKind archKind = static_cast<VPU::ArchKind>(VPU::ArchKind::NPU37XX);
    const auto contentType = contentAttr.getType();
    auto acheAlignSize = static_cast<size_t>(
            alignSizeForSwizzling(contentType.getTotalAllocSize().count(), getSizeAlignmentForSwizzling(archKind)));

    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    std::vector<float> tempBuf(bufSizeBytes);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuf.data()), bufSizeBytes));

    const std::vector<float> expectedVals = {10, 10, 10, 10};

    EXPECT_EQ(acheAlignSize, static_cast<int>(bufSizeBytes));

    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], tempBuf[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ScalarMultInverse) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const auto vals = [&]() {
        auto values = generateValues<float>(baseType.getNumElements());
        values[0] = 42.f;  // Note: change values[0] (that is 0.f) to avoid divide-by-zero
        return values;
    }();
    std::vector<float> expectedVals(vals.size());
    std::transform(vals.begin(), vals.end(), expectedVals.begin(), [&](float item) {
        return 1.f / item;
    });

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.scalarMultInverse();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, ScalarMultInverse_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(&ctx));

    const std::vector<float> vals{42.f};
    const std::vector<float> expectedVals{1.f / vals[0]};

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.scalarMultInverse();

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(contentAttr.isSplat(), content.isSplat());

    const auto contentVals = content.getValues<float>();
    EXPECT_EQ(contentVals.size(), static_cast<size_t>(baseType.getNumElements()));

    for (size_t i = 0; i < expectedVals.size(); ++i) {
        EXPECT_EQ(expectedVals[i], contentVals[i]);
    }
}

namespace {
struct QuantTypeParams {
    unsigned quantFlags = 0;
    mlir::Type storageType;
    double scale = 1.;
    int64_t zeroPoint = 0;
    int64_t storageMin = 0;
    int64_t storageMax = 0;
};
struct ConvertElemTypeParams {
    std::vector<float> inputValues;
    mlir::ShapedType baseType;
    QuantTypeParams inType;
    QuantTypeParams outType;
};
using GetConvertElemTypeParams = std::function<ConvertElemTypeParams(mlir::MLIRContext*)>;

std::tuple<mlir::quant::QuantizedType, mlir::quant::QuantizedType> getDefaultTypes(mlir::MLIRContext* ctx,
                                                                                   QuantTypeParams paramsIn,
                                                                                   QuantTypeParams paramsOut) {
    return std::make_tuple(
            mlir::quant::UniformQuantizedType::get(paramsIn.quantFlags, paramsIn.storageType,
                                                   /*expressed type*/ mlir::Float16Type::get(ctx), paramsIn.scale,
                                                   paramsIn.zeroPoint, paramsIn.storageMin, paramsIn.storageMax),
            mlir::quant::UniformQuantizedType::get(paramsOut.quantFlags, paramsOut.storageType,
                                                   /*expressed type*/ mlir::Float16Type::get(ctx), paramsOut.scale,
                                                   paramsOut.zeroPoint, paramsOut.storageMin, paramsOut.storageMax));
}
}  // namespace
class MLIR_ConstContentAttrTest_ConvertElemType :
        public MLIR_ConstContentAttrTest,
        public ::testing::WithParamInterface<GetConvertElemTypeParams> {
public:
    MLIR_ConstContentAttrTest_ConvertElemType() {
        ctx.loadDialect<mlir::quant::QuantizationDialect>();
    }
};

TEST_P(MLIR_ConstContentAttrTest_ConvertElemType, Roundtrip) {
    auto [inputVals, baseType, qParamsIn, qParamsOut] = GetParam()(&ctx);
    const auto [qTypeIn, qTypeOut] = getDefaultTypes(&ctx, qParamsIn, qParamsOut);
    const auto offsetBetweenRanges = qParamsOut.storageMax - qParamsIn.storageMax;

    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(inputVals));
    const auto contentAttr = Const::ContentAttr::get(
            baseAttr, Const::ContentSetup(baseType).castElemType(qTypeIn).convertElemType(qTypeOut));
    const auto roundtripContentAttr = contentAttr.transform().convertElemType(qTypeIn).get();

    std::vector<float> shiftedVals;
    std::transform(inputVals.begin(), inputVals.end(), std::back_inserter(shiftedVals), [&](float x) {
        return x + offsetBetweenRanges;
    });
    const auto content = contentAttr.fold();
    auto contentVals = to_std_vector(content.getValues<float>());
    ASSERT_EQ(ArrayRef(contentVals).take_front(shiftedVals.size()), ArrayRef(shiftedVals));

    const auto roundtripContent = roundtripContentAttr.fold();
    auto roundtripVals = to_std_vector(roundtripContent.getValues<float>());
    ASSERT_EQ(ArrayRef(roundtripVals).take_front(inputVals.size()), ArrayRef(inputVals));
}

INSTANTIATE_TEST_SUITE_P(
        I8_to_U8, MLIR_ConstContentAttrTest_ConvertElemType,
        ::testing::Values(
                [](mlir::MLIRContext* ctx) -> ConvertElemTypeParams {
                    // Note: from the transformation's perspective, zero-points
                    // could be anything, but from NPU perspective, de-facto
                    // values are 0 and 128.
                    QuantTypeParams i8Params{
                            mlir::quant::QuantizationFlags::Signed, getInt8Type(ctx), 0.5, 0, -128, 127};
                    QuantTypeParams u8Params{0, getUInt8Type(ctx), 0.5, 128, 0, 255};
                    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float16Type::get(ctx));
                    std::vector<float> inputVals{-5.f};
                    return {inputVals, baseType, i8Params, u8Params};
                },

                [](mlir::MLIRContext* ctx) -> ConvertElemTypeParams {
                    QuantTypeParams i8Params{
                            mlir::quant::QuantizationFlags::Signed, getInt8Type(ctx), 0.5, 0, -128, 127};
                    QuantTypeParams u8Params{0, getUInt8Type(ctx), 0.5, 128, 0, 255};
                    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float16Type::get(ctx));
                    std::vector<float> inputVals;
                    for (int64_t i = 0; i < baseType.getNumElements(); ++i) {
                        int64_t res = 0;
                        if (i % 2 == 0) {
                            res = i8Params.storageMin + i;
                        } else {
                            res = i8Params.storageMax - i;
                        }
                        inputVals.push_back(static_cast<float>(res));
                    }
                    return {inputVals, baseType, i8Params, u8Params};
                }));

INSTANTIATE_TEST_SUITE_P(
        U4_to_I4, MLIR_ConstContentAttrTest_ConvertElemType,
        ::testing::Values(
                [](mlir::MLIRContext* ctx) -> ConvertElemTypeParams {
                    QuantTypeParams u4Params{0, getUInt4Type(ctx), 1.0, 8, 0, 15};
                    QuantTypeParams i4Params{mlir::quant::QuantizationFlags::Signed, getInt4Type(ctx), 1.0, 0, -8, 7};
                    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float16Type::get(ctx));
                    std::vector<float> inputVals{5.f};
                    return {inputVals, baseType, u4Params, i4Params};
                },

                [](mlir::MLIRContext* ctx) -> ConvertElemTypeParams {
                    QuantTypeParams u4Params{0, getUInt4Type(ctx), 1.0, 8, 0, 15};
                    QuantTypeParams i4Params{mlir::quant::QuantizationFlags::Signed, getInt4Type(ctx), 1.0, 0, -8, 7};
                    const auto baseType = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float16Type::get(ctx));
                    std::vector<float> inputVals;
                    for (int64_t i = 0; i < baseType.getNumElements(); ++i) {
                        uint64_t res = (u4Params.storageMin + i) % u4Params.storageMax;
                        inputVals.push_back(static_cast<float>(res));
                    }
                    return {inputVals, baseType, u4Params, i4Params};
                }));

using CreateElementsAttr = std::function<mlir::ElementsAttr(mlir::MLIRContext*)>;
class MLIR_ConstContentAttrTypedTest :
        public MLIR_ConstContentAttrTest,
        public ::testing::WithParamInterface<CreateElementsAttr> {
    static const char* dataAddressImpl(mlir::ElementsAttr attr) {
        // support most probable candidates
        if (const auto content = attr.dyn_cast<mlir::DenseElementsAttr>()) {
            return content.getRawData().data();
        }
        if (const auto content = attr.dyn_cast<mlir::DenseResourceElementsAttr>()) {
            return content.getRawHandle().getBlob()->getData().data();
        }
        assert(false && "Extend this function with extra types");
        return nullptr;
    }

protected:
    mlir::ElementsAttr baseContent() {
        return GetParam()(&ctx);
    }
    static const void* dataAddress(mlir::ElementsAttr attr) {
        // return a void* instead to compare pointers instead of strings
        return static_cast<const void*>(dataAddressImpl(attr));
    }

    SmallVector<Const::TransformAttrInterface> getTransformations() {
        SmallVector<Const::TransformAttrInterface> randomTransformations = {
                Const::CastElemTypeAttr::get(mlir::Float16Type::get(&ctx)),
        };
        return randomTransformations;
    }
};

namespace {
// workaround for ElementsAttr::getTypeID() not being marked const
mlir::TypeID getTypeID(mlir::ElementsAttr attr) {
    return attr.getTypeID();
}
}  // namespace

// Note: expect copy behavior to be identical, regardless of the base content type
TEST_P(MLIR_ConstContentAttrTypedTest, CopyContentAttr) {
    const auto baseAttr = baseContent();
    const auto contentAttr = Const::ContentAttr::get(baseAttr, getTransformations());

    const auto copy = contentAttr;
    ASSERT_EQ(copy.getType(), contentAttr.getType());
    ASSERT_EQ(copy.getTransformations(), contentAttr.getTransformations());
    ASSERT_EQ(getTypeID(copy.getBaseContent()), getTypeID(contentAttr.getBaseContent()));
    ASSERT_EQ(dataAddress(copy.getBaseContent()), dataAddress(contentAttr.getBaseContent()))
            << "ContentAttr copy should not deepcopy data";
}

TEST_P(MLIR_ConstContentAttrTypedTest, CopyContentAttrIndirectly) {
    const auto baseAttr = baseContent();
    const auto contentAttr = Const::ContentAttr::get(baseAttr, getTransformations());

    const auto indirectCopy = Const::ContentAttr::get(contentAttr.getBaseContent());
    ASSERT_NE(indirectCopy.getType(), contentAttr.getType()) << "Content-only copy does not copy type";
    ASSERT_NE(indirectCopy.getTransformations(), contentAttr.getTransformations())
            << "Content-only copy does not copy transformations";
    ASSERT_EQ(getTypeID(indirectCopy.getBaseContent()), getTypeID(contentAttr.getBaseContent()));
    ASSERT_EQ(dataAddress(indirectCopy.getBaseContent()), dataAddress(contentAttr.getBaseContent()))
            << "ContentAttr::get() should not deepcopy data";
}

INSTANTIATE_TEST_SUITE_P(
        CommonElementsAttrImplementations, MLIR_ConstContentAttrTypedTest,
        ::testing::Values(
                [](mlir::MLIRContext* ctx) -> mlir::ElementsAttr {
                    const auto type = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(ctx));
                    const auto vals = generateValues<float>(type.getNumElements());
                    return Const::createConstContent(type, ArrayRef(vals));
                },

                [](mlir::MLIRContext* ctx) -> mlir::ElementsAttr {
                    const auto type = mlir::RankedTensorType::get({1, 2, 3, 4}, mlir::Float32Type::get(ctx));

                    // create owning blob
                    static std::unique_ptr<float[]> data = std::make_unique<float[]>(type.getNumElements());
                    auto range = convertArrayRef(ArrayRef<float>(data.get(), type.getNumElements()));

                    return Const::createExternalConstContent(type, range, "MLIR_ConstContentAttrTypedTest_resource");
                }));

TEST_F(MLIR_ConstContentAttrTest, Quantize) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto baseType = mlir::RankedTensorType::get({1, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals = {-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
                               0.0,  0.25,  0.5,  0.75,  1.0,  1.25,  1.5,  1.75};
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    // Quantize to 1/127;56 type => [-1.4409448818897637; 0.5590551181102362]
    const double scale = 1.0 / 127.0;
    const auto quantType =
            mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, getSInt8Type(&ctx),
                                                   mlir::Float32Type::get(&ctx), scale, 56, -127, 127);
    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> target = {-127, -127, -127, -103, -71, -39, -7, 24, 56, 88, 120, 127, 127, 127, 127, 127};
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], target[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, PerAxisQuantize) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto baseType = mlir::RankedTensorType::get({2, 16}, mlir::Float32Type::get(&ctx));
    std::vector<float> vals(baseType.getNumElements());
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            vals[16 * i + j] = -2.0f + j * 0.25f;
        }
    }
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));

    Const::ContentSetup baseContentAttrSetup(baseType);

    // first axis qType to 1/127;56 type => [-1.4409448818897637; 0.5590551181102362]
    // second axis qType 0.003;-21; type => [-0.318; 0.444]
    std::vector<double> scales{1.0 / 127.0, 0.003};
    std::vector<int64_t> zeroPoints{56, -21};
    const auto quantType = mlir::quant::UniformQuantizedPerAxisType::get(
            mlir::quant::QuantizationFlags::Signed, getSInt8Type(&ctx), mlir::Float32Type::get(&ctx), scales,
            zeroPoints, 0, -127, 127);

    auto contentAttrSetup = baseContentAttrSetup.quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    std::vector<char> targetAxis0 = {-127, -127, -127, -103, -71, -39, -7, 24, 56, 88, 120, 127, 127, 127, 127, 127};
    std::vector<char> targetAxis1 = {-127, -127, -127, -127, -127, -127, -127, -104,
                                     -21,  62,   127,  127,  127,  127,  127,  127};
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(contentVals[i], targetAxis0[i]);
        EXPECT_EQ(contentVals[16 + i], targetAxis1[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, DequantizeQuantize) {
    ctx.loadDialect<mlir::quant::QuantizationDialect>();

    const auto si8Type = getSInt8Type(&ctx);
    // Quantize to 0.003;-21;<-120;120> type => [-0.297;0.423]
    const auto quantType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, si8Type,
                                                                  mlir::Float16Type::get(&ctx), 0.003, 56, -120, 120);

    const auto baseType = mlir::RankedTensorType::get({1, 16}, si8Type);
    std::vector<char> vals(baseType.getNumElements());
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = -120 + 15 * i;
    }
    const auto baseAttr = Const::createConstContent(baseType, ArrayRef(vals));
    Const::ContentSetup baseContentAttrSetup(baseType);
    auto contentAttrSetup = baseContentAttrSetup.castElemType(quantType).dequantize().quantize(quantType);

    auto contentAttr = Const::ContentAttr::get(baseAttr, std::move(contentAttrSetup));
    const auto content = contentAttr.fold();
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_FALSE(content.isSplat());

    const auto contentVals = content.getValues<char>();
    EXPECT_EQ(contentVals.size(), vals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], vals[i]);
    }
}

TEST_F(MLIR_ConstContentAttrTest, OperationPrinting) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main() -> tensor<1xf32> {
                %cst0 = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
                return %cst0 : tensor<1xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);
    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);
    mlir::OpBuilder builder(func);

    Const::DeclareOp constOp = nullptr;
    func.walk([&](Const::DeclareOp op) {
        constOp = op;
    });
    ASSERT_NE(constOp, nullptr);

    ASSERT_NO_THROW(constOp.dump());
    ASSERT_NO_THROW(llvm::outs() << constOp << "\n");
    // Note: use error-level-log so that it is actually called...
    ASSERT_NO_THROW(Logger::global().error("{0}", constOp));

    // Note: special case with "generic op form", selected internally by MLIR's
    // AsmPrinter when op verification fails.
    mlir::OpPrintingFlags printUsingGenericFallbackLogic;
    printUsingGenericFallbackLogic.printGenericOpForm();
    ASSERT_NO_THROW(constOp.print(llvm::outs(), printUsingGenericFallbackLogic));
    llvm::outs() << "\n";
}
