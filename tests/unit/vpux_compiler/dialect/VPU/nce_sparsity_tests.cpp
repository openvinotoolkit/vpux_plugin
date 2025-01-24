//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/sparsity_strategy.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/sparsity.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <memory>

using namespace vpux;
using namespace VPU::NCESparsity;

namespace {

Const::DeclareOp createConstOp(mlir::MLIRContext* context, const std::initializer_list<int64_t>& rawShape,
                               double ratio) {
    const Shape shape(rawShape);
    const auto dataType = mlir::RankedTensorType::get(shape.raw(), mlir::Float32Type::get(context));

    std::vector<float> content(shape.totalSize(), 1);
    const auto numSparse = static_cast<int64_t>(shape.totalSize() * ratio);
    std::fill_n(content.begin(), numSparse, 0);

    const auto dataAttr = Const::createConstContent(dataType, mlir::ArrayRef<float>(content));

    mlir::OpBuilder builder(context);
    return builder.create<Const::DeclareOp>(mlir::UnknownLoc::get(context), dataType,
                                            Const::ContentAttr::get(dataAttr));
}

SmallVector<int64_t> getNumElemsPerOC(Const::DeclareOp constOp) {
    const auto content = constOp.getContent();
    const auto elemType = content.getType().getElementType();
    return vpux::countNonSparseElementsPerOC(content, elemType);
}

std::unique_ptr<BaseWeightsSparsityStrategy> getRatioBasedStrategy(
        double floatRatio = WEIGHTS_SPARSITY_FLOAT_RATIO_THRESHOLD,
        double intRatio = WEIGHTS_SPARSITY_INT_RATIO_THRESHOLD) {
    return std::unique_ptr<BaseWeightsSparsityStrategy>(new RatioBasedWeightsSparsityStrategy(floatRatio, intRatio));
}

std::unique_ptr<BaseWeightsSparsityStrategy> getCMXBasedStrategy(int64_t cmxSizeBytes) {
    vpux::Byte cmxSize(cmxSizeBytes);
    return std::unique_ptr<BaseWeightsSparsityStrategy>(
            new CMXConsumptionBasedWeightsSparsityStrategy(cmxSize, CMX_BASED_STRATEGY_DEFAULT_INTERVALS));
}

struct ThresholdPair {
    ThresholdPair(double reference)
            : _lowerThreshold(std::max(0., reference / 2.)), _upperThreshold(std::min(1., (1. + reference) / 2.)) {
    }
    double _lowerThreshold;
    double _upperThreshold;
};

};  // namespace

using MLIR_WeightsSparsity = MLIR_UnitBase;

void ratioBasedStrategyTestTemplate(double threshold, bool isFloat, mlir::DialectRegistry& registry) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    ctx.loadDialect<Const::ConstDialect>();

    Logger log("weights-sparsity-test", LogLevel::Debug);

    const auto defaultFloatRatioThreshold = ThresholdPair(threshold);

    auto lowSparsity = createConstOp(&ctx, {1, 64, 32, 32}, defaultFloatRatioThreshold._lowerThreshold);
    auto highSparsity = createConstOp(&ctx, {1, 64, 32, 32}, defaultFloatRatioThreshold._upperThreshold);
    const auto lowSparsityType = lowSparsity.getType().cast<vpux::NDTypeInterface>();
    const auto highSparsityType = highSparsity.getType().cast<vpux::NDTypeInterface>();
    const auto lowSparsityNumElems = getNumElemsPerOC(lowSparsity);
    const auto highSparsityNumElems = getNumElemsPerOC(highSparsity);

    // LOWER_FLOAT_THRESHOLD < threshold -> no sparsity
    EXPECT_EQ(getRatioBasedStrategy()->shouldSparsifyWeights(log, lowSparsityType, lowSparsityNumElems, isFloat),
              false);
    // UPPER_FLOAT_THRESHOLD > threshold -> enable sparsity
    EXPECT_EQ(getRatioBasedStrategy()->shouldSparsifyWeights(log, highSparsityType, highSparsityNumElems, isFloat),
              true);
}

std::vector<int32_t> constructTableToBeShifted(int size) {
    std::vector<int32_t> table(size);
    for (int index = 0; index < size; index++) {
        table[index] = index;
    }

    return table;
}

TEST_F(MLIR_WeightsSparsity, RatioBasedStrategyFloatInput) {
    ratioBasedStrategyTestTemplate(WEIGHTS_SPARSITY_FLOAT_RATIO_THRESHOLD, true, registry);
}

TEST_F(MLIR_WeightsSparsity, RatioBasedStrategyIntInput) {
    ratioBasedStrategyTestTemplate(WEIGHTS_SPARSITY_INT_RATIO_THRESHOLD, false, registry);
}

TEST_F(MLIR_WeightsSparsity, CMXBasedStrategy) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    ctx.loadDialect<Const::ConstDialect>();

    Logger log("weights-sparsity-test", LogLevel::Debug);

    const auto firstInterval = *CMX_BASED_STRATEGY_DEFAULT_INTERVALS.begin();
    VPUX_THROW_WHEN(CMX_BASED_STRATEGY_DEFAULT_INTERVALS.size() < 2 ||
                            firstInterval._floatRatioThreshold != CMXBasedSparsityThreshold::DISABLED_SPARSITY_RATIO,
                    "CMX_BASED_STRATEGY_DEFAULT_INTERVALS should contain at least 2 intervals and first of them should "
                    "disable sparsity");

    auto lightWeightAlmostDense = createConstOp(&ctx, {1, 1, 32, 32}, 0.01);
    auto lightWeightAlmostSparse = createConstOp(&ctx, {1, 1, 32, 32}, 0.99);
    const auto lightWeightAlmostDenseType = lightWeightAlmostDense.getType().cast<vpux::NDTypeInterface>();
    const auto lightWeightAlmostSparseType = lightWeightAlmostSparse.getType().cast<vpux::NDTypeInterface>();
    const auto lightWeightAlmostDenseNumElems = getNumElemsPerOC(lightWeightAlmostDense);
    const auto lightWeightAlmostSparseNumElems = getNumElemsPerOC(lightWeightAlmostSparse);

    const auto bigCMX = static_cast<int64_t>(4. * 32. * 32. / (firstInterval._cmxSizeRatio + 0.0001) * 2.);

    // No sparsity regardless to content in first interval
    EXPECT_EQ(getCMXBasedStrategy(bigCMX)->shouldSparsifyWeights(log, lightWeightAlmostDenseType,
                                                                 lightWeightAlmostDenseNumElems, true),
              false);
    EXPECT_EQ(getCMXBasedStrategy(bigCMX)->shouldSparsifyWeights(log, lightWeightAlmostSparseType,
                                                                 lightWeightAlmostSparseNumElems, true),
              false);

    auto it = CMX_BASED_STRATEGY_DEFAULT_INTERVALS.begin();
    const auto end = CMX_BASED_STRATEGY_DEFAULT_INTERVALS.end();
    for (++it; it != end;) {
        const auto sparsityThreshold = ThresholdPair(it->_intRatioThreshold);
        const double lowerCMX = it->_cmxSizeRatio;
        const double upperCMX = ++it == end ? 1. : it->_cmxSizeRatio;
        const double targetCMXRatio = (lowerCMX + upperCMX) / 2.;
        // Generating CMX size in such way, to get in just in the middle of intervals, for example
        // 5-50% => weights should have 22% of CMX size
        const auto cmxSize = static_cast<int64_t>(4 * 32. * 32. / targetCMXRatio);

        auto denseOp = createConstOp(&ctx, {1, 1, 32, 32}, sparsityThreshold._lowerThreshold);
        const auto denseType = denseOp.getType().cast<vpux::NDTypeInterface>();
        const auto denseNumElems = getNumElemsPerOC(denseOp);
        EXPECT_EQ(getCMXBasedStrategy(cmxSize)->shouldSparsifyWeights(log, denseType, denseNumElems, false), false);

        auto sparseOp = createConstOp(&ctx, {1, 1, 32, 32}, sparsityThreshold._upperThreshold);
        const auto sparseType = sparseOp.getType().cast<vpux::NDTypeInterface>();
        const auto sparseNumElems = getNumElemsPerOC(sparseOp);
        EXPECT_EQ(getCMXBasedStrategy(cmxSize)->shouldSparsifyWeights(log, sparseType, sparseNumElems, false), true);
    }
}

TEST_F(MLIR_WeightsSparsity, NewWeightsTableFormatMapper) {
    std::vector<int32_t> zeroPointsK16{0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15};
    std::vector<int32_t> zeroPointsK32{0, 2,  16, 18, 1, 3,  17, 19, 4,  6,  20, 22, 5,  7,  21, 23,
                                       8, 10, 24, 26, 9, 11, 25, 27, 12, 14, 28, 30, 13, 15, 29, 31};
    std::vector<int32_t> zeroPointsK48{0,  2,  16, 18, 32, 34, 1,  3,  17, 19, 33, 35, 4,  6,  20, 22,
                                       36, 38, 5,  7,  21, 23, 37, 39, 8,  10, 24, 26, 40, 42, 9,  11,
                                       25, 27, 41, 43, 12, 14, 28, 30, 44, 46, 13, 15, 29, 31, 45, 47};
    std::vector<int32_t> zeroPointsK64{0,  2,  16, 18, 32, 34, 48, 50, 1,  3,  17, 19, 33, 35, 49, 51,
                                       4,  6,  20, 22, 36, 38, 52, 54, 5,  7,  21, 23, 37, 39, 53, 55,
                                       8,  10, 24, 26, 40, 42, 56, 58, 9,  11, 25, 27, 41, 43, 57, 59,
                                       12, 14, 28, 30, 44, 46, 60, 62, 13, 15, 29, 31, 45, 47, 61, 63};
    std::vector<int32_t> zeroPointsK80{0,  2,  16, 18, 32, 34, 48, 50, 64, 66, 1,  3,  17, 19, 33, 35, 49, 51, 65, 67,
                                       4,  6,  20, 22, 36, 38, 52, 54, 68, 70, 5,  7,  21, 23, 37, 39, 53, 55, 69, 71,
                                       8,  10, 24, 26, 40, 42, 56, 58, 72, 74, 9,  11, 25, 27, 41, 43, 57, 59, 73, 75,
                                       12, 14, 28, 30, 44, 46, 60, 62, 76, 78, 13, 15, 29, 31, 45, 47, 61, 63, 77, 79};
    std::vector<int32_t> zeroPointsK96{0,  2,  16, 18, 32, 34, 48, 50, 64, 66, 80, 82, 1,  3,  17, 19, 33, 35, 49, 51,
                                       65, 67, 81, 83, 4,  6,  20, 22, 36, 38, 52, 54, 68, 70, 84, 86, 5,  7,  21, 23,
                                       37, 39, 53, 55, 69, 71, 85, 87, 8,  10, 24, 26, 40, 42, 56, 58, 72, 74, 88, 90,
                                       9,  11, 25, 27, 41, 43, 57, 59, 73, 75, 89, 91, 12, 14, 28, 30, 44, 46, 60, 62,
                                       76, 78, 92, 94, 13, 15, 29, 31, 45, 47, 61, 63, 77, 79, 93, 95};
    std::vector<int32_t> zeroPointsK112{
            0,   2,  16, 18, 32,  34,  48, 50, 64,  66,  80, 82, 96, 98,  1,   3,  17, 19,  33,  35, 49, 51, 65,
            67,  81, 83, 97, 99,  4,   6,  20, 22,  36,  38, 52, 54, 68,  70,  84, 86, 100, 102, 5,  7,  21, 23,
            37,  39, 53, 55, 69,  71,  85, 87, 101, 103, 8,  10, 24, 26,  40,  42, 56, 58,  72,  74, 88, 90, 104,
            106, 9,  11, 25, 27,  41,  43, 57, 59,  73,  75, 89, 91, 105, 107, 12, 14, 28,  30,  44, 46, 60, 62,
            76,  78, 92, 94, 108, 110, 13, 15, 29,  31,  45, 47, 61, 63,  77,  79, 93, 95,  109, 111};
    std::vector<int32_t> zeroPointsK128{
            0,   2,   16,  18,  32, 34, 48, 50,  64,  66,  80,  82,  96, 98, 112, 114, 1,   3,   17,  19,  33, 35,  49,
            51,  65,  67,  81,  83, 97, 99, 113, 115, 4,   6,   20,  22, 36, 38,  52,  54,  68,  70,  84,  86, 100, 102,
            116, 118, 5,   7,   21, 23, 37, 39,  53,  55,  69,  71,  85, 87, 101, 103, 117, 119, 8,   10,  24, 26,  40,
            42,  56,  58,  72,  74, 88, 90, 104, 106, 120, 122, 9,   11, 25, 27,  41,  43,  57,  59,  73,  75, 89,  91,
            105, 107, 121, 123, 12, 14, 28, 30,  44,  46,  60,  62,  76, 78, 92,  94,  108, 110, 124, 126, 13, 15,  29,
            31,  45,  47,  61,  63, 77, 79, 93,  95,  109, 111, 125, 127};
    std::vector<int32_t> zeroPointsK144{
            0,   2,   16,  18,  32,  34,  48,  50,  64,  66,  80,  82,  96,  98,  112, 114, 1,   3,  17,  19,  33,
            35,  49,  51,  65,  67,  81,  83,  97,  99,  113, 115, 4,   6,   20,  22,  36,  38,  52, 54,  68,  70,
            84,  86,  100, 102, 116, 118, 5,   7,   21,  23,  37,  39,  53,  55,  69,  71,  85,  87, 101, 103, 117,
            119, 8,   10,  24,  26,  40,  42,  56,  58,  72,  74,  88,  90,  104, 106, 120, 122, 9,  11,  25,  27,
            41,  43,  57,  59,  73,  75,  89,  91,  105, 107, 121, 123, 12,  14,  28,  30,  44,  46, 60,  62,  76,
            78,  92,  94,  108, 110, 124, 126, 13,  15,  29,  31,  45,  47,  61,  63,  77,  79,  93, 95,  109, 111,
            125, 127, 128, 130, 129, 131, 132, 134, 133, 135, 136, 138, 137, 139, 140, 142, 141, 143};
    std::vector<int32_t> zeroPointsK304{
            0,   2,   16,  18,  32,  34,  48,  50,  64,  66,  80,  82,  96,  98,  112, 114, 1,   3,   17,  19,  33,
            35,  49,  51,  65,  67,  81,  83,  97,  99,  113, 115, 4,   6,   20,  22,  36,  38,  52,  54,  68,  70,
            84,  86,  100, 102, 116, 118, 5,   7,   21,  23,  37,  39,  53,  55,  69,  71,  85,  87,  101, 103, 117,
            119, 8,   10,  24,  26,  40,  42,  56,  58,  72,  74,  88,  90,  104, 106, 120, 122, 9,   11,  25,  27,
            41,  43,  57,  59,  73,  75,  89,  91,  105, 107, 121, 123, 12,  14,  28,  30,  44,  46,  60,  62,  76,
            78,  92,  94,  108, 110, 124, 126, 13,  15,  29,  31,  45,  47,  61,  63,  77,  79,  93,  95,  109, 111,
            125, 127, 128, 130, 144, 146, 160, 162, 176, 178, 192, 194, 208, 210, 224, 226, 240, 242, 129, 131, 145,
            147, 161, 163, 177, 179, 193, 195, 209, 211, 225, 227, 241, 243, 132, 134, 148, 150, 164, 166, 180, 182,
            196, 198, 212, 214, 228, 230, 244, 246, 133, 135, 149, 151, 165, 167, 181, 183, 197, 199, 213, 215, 229,
            231, 245, 247, 136, 138, 152, 154, 168, 170, 184, 186, 200, 202, 216, 218, 232, 234, 248, 250, 137, 139,
            153, 155, 169, 171, 185, 187, 201, 203, 217, 219, 233, 235, 249, 251, 140, 142, 156, 158, 172, 174, 188,
            190, 204, 206, 220, 222, 236, 238, 252, 254, 141, 143, 157, 159, 173, 175, 189, 191, 205, 207, 221, 223,
            237, 239, 253, 255, 256, 258, 272, 274, 288, 290, 257, 259, 273, 275, 289, 291, 260, 262, 276, 278, 292,
            294, 261, 263, 277, 279, 293, 295, 264, 266, 280, 282, 296, 298, 265, 267, 281, 283, 297, 299, 268, 270,
            284, 286, 300, 302, 269, 271, 285, 287, 301, 303};

    std::vector<std::vector<int32_t>> tablesToTest{zeroPointsK16,  zeroPointsK32, zeroPointsK48,  zeroPointsK64,
                                                   zeroPointsK80,  zeroPointsK96, zeroPointsK112, zeroPointsK128,
                                                   zeroPointsK144, zeroPointsK304};
    std::vector<int32_t> k{16, 32, 48, 64, 80, 96, 112, 128, 144, 304};

    int32_t index = 0;
    for (auto table : tablesToTest) {
        auto tableToBeShifted = constructTableToBeShifted(table.size());
        auto constructedNewZeroPointOnlyTable =
                NewWeightsTableFormatMapper::constructNewZeroPointOnlyTable(tableToBeShifted);
        for (unsigned long j = 0; j < table.size(); j++) {
            EXPECT_EQ(NewWeightsTableFormatMapper::mathematicallyEncodePositionInNewZeroPointOnlyTableLayout(table[j],
                                                                                                             k[index]),
                      j);
            EXPECT_EQ(
                    NewWeightsTableFormatMapper::mathematicallyDecodePositionInNewZeroPointOnlyTableLayout(j, k[index]),
                    table[j]);

            EXPECT_EQ(NewWeightsTableFormatMapper::encodePositionInNewZeroPointOnlyTableLayout(table[j], k[index]), j);
            EXPECT_EQ(NewWeightsTableFormatMapper::decodePositionInNewZeroPointOnlyTableLayout(j, k[index]), table[j]);
        }
        EXPECT_EQ(constructedNewZeroPointOnlyTable, table);

        index += 1;
    }
}
