//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/optional.hpp"

#include <llvm/ADT/bit.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

namespace NCESparsity {

// base_ptr is 9bits size
const int BASE_PTR_SIZE = 9;

const VPU::SparsitySupport FULLY_SUPPORTED_SPARSITY_MODE =
        SparsitySupport::SPARSE_INPUTS | SparsitySupport::SPARSE_OUTPUTS | SparsitySupport::SPARSE_WEIGHTS;

constexpr int32_t SPARSITY_PTR_WHEN_NO_SPARSITY = 0xFFFFFF;

const unsigned int DEFAULT_SPARSIFIABLE_INPUT_OPERAND_ID = 0;
const unsigned int ELTWISE_SPARSIFIABLE_SECOND_INPUT_OPERAND_ID = 1;

enum class Mode { DW_CONV, POOL };

int64_t getBitPatternSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

int32_t getWeightPtrStep(mlir::Value weights);

std::vector<int32_t> getWeightsTable(mlir::Type inElemType, mlir::Type outElemType,
                                     std::optional<int32_t> weightsPtrOffset, int32_t weightsPtrStep,
                                     std::optional<int32_t> sparsityPtrOffset, int32_t sparsityPtrStep,
                                     VPU::NCESparsity::PPEConverterCb ppeConverter,
                                     VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                     mlir::Type weightsElemType = nullptr, const Const::ContentAttr& bias = {},
                                     mlir::FloatAttr constScale = nullptr);
std::vector<int32_t> getWeightsTable(mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> weightPtrs,
                                     ArrayRef<int32_t> sparsityPtrs, VPU::NCESparsity::PPEConverterCb ppeConverter,
                                     VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                     mlir::Type weightsElemType = nullptr, const Const::ContentAttr& bias = {},
                                     mlir::FloatAttr constScale = nullptr);

std::vector<float> getScaleTable(mlir::Type inElemType, mlir::Type outElemType,
                                 VPU::NCESparsity::PPEConverterCb ppeConverter, int64_t OC,
                                 mlir::Type weightsElemType = nullptr, mlir::FloatAttr constScale = nullptr);

std::vector<float> getBiasTable(mlir::Type inElemType, mlir::Type outElemType,
                                VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                mlir::Type weightsElemType = nullptr, const Const::ContentAttr& bias = {});

std::vector<int32_t> patchWeightsTableSparsityPtrs(const std::vector<std::int32_t>& weightsTableVals,
                                                   const int32_t sparsityPtrOffset, const int32_t sparsityPtrStep);

Shape inferWeightsTableShape(int64_t OC);
Shape inferScaleTableShape(int64_t OC);
Shape inferBiasTableShape(int64_t OC);
Shape inferWeightsSparsityMapShape(ShapeRef dataShape);

mlir::FailureOr<SmallVector<double>> getRescaledBias(const Const::ContentAttr& biasAttr, mlir::Type inElemType,
                                                     mlir::Type filterElemType, int64_t OC);

double getSparsityRatio(vpux::NDTypeInterface weightsType, ArrayRef<int64_t> numNonSparseElemsPerOC);

bool isSparsifiableWeightsOperand(mlir::Value operand);
bool isSuperdenseRequired(const VPU::ArchKind arch, const DimsOrder outOrder, const ShapeRef outShape,
                          const mlir::Type outElemType);
inline VPU::SparsitySupport bitwiseNot(const VPU::SparsitySupport bits) {
    static_assert(sizeof(bits) == sizeof(uint32_t), "VPU::SparsitySupport has unexpected size");
    return static_cast<VPU::SparsitySupport>(~static_cast<uint32_t>(bits));
}

// 5D weights.
int32_t get5DWeightPtrStep(mlir::Value weights);

std::vector<int32_t> create5DWeightsTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                              const Const::ContentAttr& bias, int64_t outputChannels,
                                              VPU::NCESparsity::PPEConverterCb ppeConverter,
                                              VPU::NCESparsity::BiasConverterCb biasConverter);

mlir::Value create5DWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable,
                                       int64_t outputChannels, int64_t groups);

//
// Convert real numbers to fixed point S16.16 format.
//

int32_t toFixedPoint(const double realVal);

//
// Convert real numbers to hex format.
//

int32_t toHex(double realVal);

//
// RuntimeSparsityStatsProvider
//

class RuntimeSparsityStatsProvider {
    const double MINIMAL_SPARSITY_THRESHOLD = 0.2;

public:
    RuntimeSparsityStatsProvider(mlir::func::FuncOp func, vpux::Logger log);

    bool containsStatistics() const;
    bool likelySparsityConsumer(mlir::Operation* op, int64_t requestedInputId) const;

private:
    vpux::Logger _logger;
    std::multimap<std::string, IE::SparsityInfoOp> _lookup;
};

//
// NewWeightsTableFormatMapper
//

class NewWeightsTableFormatMapper {
public:
    // utility function
    static int32_t normalizeKAndReturnCurrentGroupOf128Sets(int32_t index, int32_t& k);

    // enconding and decoding functions between the zero point initial index and its new index in the new format
    // these two functions use mathematical computations in order to compute the result
    static int32_t mathematicallyEncodePositionInNewZeroPointOnlyTableLayout(int32_t zeroPointIndex, int32_t k);
    static int32_t mathematicallyDecodePositionInNewZeroPointOnlyTableLayout(int32_t position, int32_t k);

    // utility functions
    static std::vector<int32_t> computeInversePermutation(std::vector<int32_t> v);
    static std::vector<int32_t> getZeroPointTableByK(int32_t k);
    static std::vector<int32_t> getZeroPointInversePermutationTableByK(int32_t k);

    // enconding and decoding functions between the zero point initial index and its new index in the new format
    // these two functions use statically-constructed vectors in order to deduce the result
    static int32_t encodePositionInNewZeroPointOnlyTableLayout(int32_t zeroPointIndex, int32_t k);
    static int32_t decodePositionInNewZeroPointOnlyTableLayout(int32_t position, int32_t k);

    // utility function used by constructNewZeroPointOnlyTable
    static void mapElementsToNewFormat(std::vector<int32_t>& table, int32_t start, int32_t end,
                                       std::vector<int32_t>& result);
    // function that constructs a zero point only table (in the new format) starting from a vector that
    // contains the zero points in ascending order based on their indices
    static std::vector<int32_t> constructNewZeroPointOnlyTable(std::vector<int32_t> table);

private:
    static std::vector<int32_t> zeroPointsK16;
    static std::vector<int32_t> zeroPointsK32;
    static std::vector<int32_t> zeroPointsK48;
    static std::vector<int32_t> zeroPointsK64;
    static std::vector<int32_t> zeroPointsK80;
    static std::vector<int32_t> zeroPointsK96;
    static std::vector<int32_t> zeroPointsK112;
    static std::vector<int32_t> zeroPointsK128;

    static std::vector<int32_t> zeroPointsK16InversePermutation;
    static std::vector<int32_t> zeroPointsK32InversePermutation;
    static std::vector<int32_t> zeroPointsK48InversePermutation;
    static std::vector<int32_t> zeroPointsK64InversePermutation;
    static std::vector<int32_t> zeroPointsK80InversePermutation;
    static std::vector<int32_t> zeroPointsK96InversePermutation;
    static std::vector<int32_t> zeroPointsK112InversePermutation;
    static std::vector<int32_t> zeroPointsK128InversePermutation;

    static std::vector<std::vector<int32_t>> zeroPointTables;
    static std::vector<std::vector<int32_t>> zeroPointInversePermutationTables;
};

}  // namespace NCESparsity

}  // namespace VPU
}  // namespace vpux
