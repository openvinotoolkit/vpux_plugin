//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/BuiltinOps.h>

#include <unordered_map>
#include <utility>

namespace vpux {
namespace VPUIP {

class BlobReader final {
public:
    static VPU::ArchKind parseDeviceRevision(const MVCNN::SummaryHeader* header);

public:
    BlobReader(mlir::MLIRContext* ctx, ArrayRef<char> blob, Logger log);
    mlir::OwningOpRef<mlir::ModuleOp> read();

private:
    void buildRunTimeResourcesOp();
    void buildCNNNetworkOp();
    void buildMainFunc();

private:
    void parseGraphInputsOutputs();
    void parseUserInputsOutputs(OpBuilderLogger& builderLog, IE::CNNNetworkOp& cnnOp);

    mlir::MemRefType parseTensorRef(const MVCNN::TensorReference* tensorRef);
    mlir::ArrayAttr parseOrder3(const MVCNN::order3* order3, int32_t ndims = 3);
    VPU::ArchKind parseDeviceRevision();
    mlir::Type convertType(mlir::MLIRContext* ctx, const MVCNN::DType& precision);

    mlir::Value createTensorOp(mlir::OpBuilder& builder, const MVCNN::TensorReference* tensorRef);

    mlir::Operation* parseConvert(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                  const MVCNN::UPALayerTask* task);
    mlir::Operation* parseConvolution(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                      ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseCTCGreedyDecoder(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                           ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseCTCGreedyDecoderSeqLen(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                 ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseDetectionOutput(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                          ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseEltwise(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                  const MVCNN::UPALayerTask* task);
    mlir::Operation* parseFakeQuantize(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseGather(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                 const MVCNN::UPALayerTask* task);
    mlir::Operation* parseGatherND(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                   ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseGatherElements(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                         ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseGRN(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                              const MVCNN::UPALayerTask* task);
    mlir::Operation* parseNorm(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                               const MVCNN::UPALayerTask* task);
    mlir::Operation* parseBroadcast(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                    ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseRoll(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                               const MVCNN::UPALayerTask* task);
    mlir::Operation* parseReduce(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                 const MVCNN::UPALayerTask* task);
    mlir::Operation* parseLSTMCell(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                   ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseLSTMSequence(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseNegative(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                   ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parsePad(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                              const MVCNN::UPALayerTask* task);
    mlir::Operation* parsePermute(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                  const MVCNN::UPALayerTask* task);
    mlir::Operation* parsePostOps(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                  const MVCNN::UPALayerTask* task);
    mlir::Operation* parsePooling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                  const MVCNN::UPALayerTask* task);
    mlir::Operation* parseQuantCast(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                    ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseReorgYolo(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                    ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseROIPooling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                     ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parsePSROIPooling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseROIAlign(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                   ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseSoftmax(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                  const MVCNN::UPALayerTask* task);
    mlir::Operation* parseTile(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                               const MVCNN::UPALayerTask* task);
    mlir::Operation* parseNonMaxSuppression(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                            ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseDepthToSpace(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseReverseSequence(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                          ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseSpaceToDepth(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseYuvToRgb(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                   ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseUpsampling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                     ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseExtractImagePatches(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                              ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);
    mlir::Operation* parseDeformablePSROIPooling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                 ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task);

private:
    using TensorReferenceOffset = flatbuffers::Offset<MVCNN::TensorReference>;

private:
    mlir::MLIRContext* _ctx;
    mlir::ModuleOp _module;
    mlir::FlatSymbolRefAttr _mainFuncName;
    Logger _log;
    const MVCNN::GraphFile* _graphFile;

    int32_t _constCounter = 0;

    SmallVector<mlir::Type> _inputTypes;
    SmallVector<mlir::Type> _outputTypes;

    std::vector<mlir::Value> _barriers;
};

}  // namespace VPUIP
}  // namespace vpux
