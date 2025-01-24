//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// bufferize VPU::NCEClusterTilingOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEClusterTilingOp origOp,
                                      VPU::NCEClusterTilingOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEClusterTilingOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    SmallVector<mlir::Value> inputOperands = newArgs.getOperands();
    SmallVector<VPU::DistributedTypeInterface> outputDistTypes;
    for (const auto& result : origOp.getResults()) {
        const auto clusterTilingBufferType = vpux::getBufferType(result.getType());
        auto outputDistType = clusterTilingBufferType.dyn_cast<VPU::DistributedTypeInterface>();
        outputDistTypes.push_back(outputDistType);
    }

    auto& origOpBodyBlock = origOp.getBody().front();

    rewriter.setInsertionPoint(origOp);

    // The goal is to move the allocation operations outside the NCEClusterTiling body.
    // In case the resulting type of NCEClusterTiling is a distributed type, these allocations
    // should also produce distributed types now that they are outside.
    // This function will ensure the correct buffer type is allocated.
    const auto createNewAllocOp = [&](mlir::Operation* op, VPU::DistributedTypeInterface outputDistType,
                                      const size_t allocOpIdx) {
        const auto outputType = op->getResult(0).getType();
        if (outputDistType != nullptr && outputDistType.containsDistributedTypes()) {
            auto distTypes = outputDistType.getDistributedTypes();
            auto targetType = (allocOpIdx < distTypes.size()) ? distTypes[allocOpIdx] : distTypes.front();
            auto targetDistType = targetType.cast<VPUIP::DistributedBufferType>();

            const auto ndOutputType = outputType.cast<vpux::NDTypeInterface>();
            const auto layout = mlir::AffineMapAttr::get(ndOutputType.getDimsOrder().toAffineMap(ctx));
            auto distributedBufferType = VPUIP::DistributedBufferType::get(
                    ctx, ndOutputType.getShape().raw(), ndOutputType.getElementType(), layout,
                    ndOutputType.getMemSpace(), targetDistType.getDistribution());
            return rewriter.create<VPURT::AllocDistributed>(op->getLoc(), distributedBufferType, nullptr, nullptr)
                    .getOperation();
        }
        const auto memrefOutputType = outputType.cast<mlir::MemRefType>();
        return rewriter.create<mlir::memref::AllocOp>(op->getLoc(), memrefOutputType).getOperation();
    };
    // checks whether given operation is a direct Inner or a level 2 or more inner
    const auto isLevelOneOp = [&origOpBodyBlock](mlir::Operation* op) -> bool {
        return op->getBlock() == &origOpBodyBlock;
    };

    const auto isCastOp = [](mlir::Operation* op) -> bool {
        return mlir::isa<mlir::UnrealizedConversionCastOp, mlir::bufferization::ToTensorOp,
                         mlir::bufferization::ToMemrefOp>(op);
    };

    const auto isInnerOp = [isLevelOneOp, isCastOp](mlir::Operation* op) -> bool {
        return !mlir::isa<VPU::YieldOp, VPUIP::UngroupSparseBufferOp, VPUIP::GroupSparseBufferOp>(op) &&
               !isCastOp(op) && !isBufAllocOp(op) && isLevelOneOp(op) && !VPUIP::isPureViewOp(op);
    };

    const auto skipUserCast = [isCastOp](mlir::Value operand) -> mlir::Value {
        if (!operand.hasOneUse()) {
            return operand;
        }
        auto userOp = *operand.getUsers().begin();
        if (isCastOp(userOp)) {
            return userOp->getResult(0);
        }
        return operand;
    };

    const auto skipProducerCast = [isCastOp](mlir::Value operand) -> mlir::Value {
        auto definingOp = operand.getDefiningOp();
        if (definingOp != nullptr && isCastOp(definingOp)) {
            return definingOp->getOperand(0);
        }
        return operand;
    };

    // Clone the allocation and grouping / ungrouping operations outside of the cluster tiling region
    DenseMap<mlir::Value, mlir::Value> innerOuterOpValueMapping;
    SmallVector<mlir::Operation*> newAllocOps;
    SmallVector<mlir::Operation*> newGroupOps;
    origOpBodyBlock.walk([&](mlir::Operation* op) {
        if (vpux::isBufAllocOp(op)) {
            log.nest().trace("Cloning allocation op '{0}' at '{1}", op->getName(), op->getLoc());
            auto outputDistType = outputDistTypes.size() > 1 ? outputDistTypes[newAllocOps.size()] : outputDistTypes[0];
            auto newAllocOp = createNewAllocOp(op, outputDistType, newAllocOps.size());
            newAllocOps.push_back(newAllocOp);
            innerOuterOpValueMapping[op->getResult(0)] = newAllocOp->getResult(0);

        } else if (mlir::isa<VPUIP::UngroupSparseBufferOp>(op)) {
            log.nest().trace("Cloning ungrouping op '{0}' at '{1}", op->getName(), op->getLoc());

            auto operand = op->getOperand(0);
            auto blockArg = skipProducerCast(operand).dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_WHEN(blockArg == nullptr, "Expected operand of ungroup op to be a block argument");
            auto outerArg = origOp.getOperand(blockArg.getArgNumber());
            outerArg = skipProducerCast(outerArg);

            if (mlir::isa<mlir::TensorType, VPU::DistributedTensorType, VPU::SparseTensorType>(outerArg.getType())) {
                auto outerType = vpux::getBufferType(outerArg.getType());
                outerArg = rewriter.create<mlir::bufferization::ToMemrefOp>(op->getLoc(), outerType, outerArg);
            }

            mlir::IRMapping mapper;
            mapper.map(operand, outerArg);
            auto newOp = rewriter.clone(*op, mapper);
            vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

            for (auto i : irange(op->getNumResults())) {
                innerOuterOpValueMapping[op->getResult(i)] = newOp->getResult(i);
            }

        } else if (mlir::isa<VPUIP::GroupSparseBufferOp>(op)) {
            if (llvm::none_of(op->getOperands(), [](mlir::Value operand) {
                    return vpux::isBufAllocOp(operand.getDefiningOp());
                })) {
                return;
            }
            log.nest().trace("Cloning grouping op '{0}' at '{1}", op->getName(), op->getLoc());

            mlir::IRMapping mapper;
            for (auto operand : op->getOperands()) {
                mapper.map(operand, innerOuterOpValueMapping[operand]);
            }
            auto newOp = rewriter.clone(*op, mapper);
            vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);
            newGroupOps.push_back(newOp);
            innerOuterOpValueMapping[op->getResult(0)] = newOp->getResult(0);
        } else if (VPUIP::isPureViewOp(op)) {
            log.nest().trace("Cloning pure view op '{0}' at '{1}", op->getName(), op->getLoc());

            mlir::IRMapping mapper;
            auto operand = op->getOperands().front();
            auto inputBlockArg = skipProducerCast(operand).dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_WHEN(inputBlockArg == nullptr, "Expected operand of view op to be a block argument");

            auto outerInput = origOp.getOperand(inputBlockArg.getArgNumber());
            outerInput = skipProducerCast(outerInput);

            if (mlir::isa<mlir::TensorType, VPU::DistributedTensorType, VPU::SparseTensorType>(outerInput.getType())) {
                auto outerType = vpux::getBufferType(outerInput.getType());
                outerInput = rewriter.create<mlir::bufferization::ToMemrefOp>(op->getLoc(), outerType, outerInput);
            }

            mapper.map(operand, outerInput);
            auto newOp = rewriter.clone(*op, mapper);
            vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

            innerOuterOpValueMapping[op->getResult(0)] = newOp->getResult(0);
        }
    });

    SmallVector<mlir::Value> newOutputOperands;
    auto& newOutputOps = !newGroupOps.empty() ? newGroupOps : newAllocOps;
    for (auto op : newOutputOps) {
        newOutputOperands.push_back(op->getResult(0));
    }

    // Create the vector of operands for the new bufferized cluster tiling op starting with the input operands
    // Additionally, get the mapping from the input operands index to the new indices of the operands
    SmallVector<mlir::Value> newOperands;
    DenseMap<int64_t, SmallVector<int64_t>> operandsIdxMapping;
    for (auto p : inputOperands | indexed) {
        auto operandIdx = p.index();
        auto operand = p.value();

        mlir::Value origOperand = origOpBodyBlock.getArguments()[operandIdx];
        for (auto user : origOperand.getUsers()) {
            auto operandUser = skipUserCast(user->getResult(0));
            if (std::distance(operandUser.getUsers().begin(), operandUser.getUsers().end()) == 0) {
                continue;
            }

            if (llvm::any_of(operandUser.getUsers(), [](mlir::Operation* userOp) {
                    return mlir::isa<VPUIP::UngroupSparseBufferOp>(userOp) ||
                           (VPUIP::isPureViewOp(userOp) && !mlir::isa<VPUIP::GroupSparseBufferOp>(userOp));
                })) {
                for (auto userOp : operandUser.getUsers()) {
                    for (auto result : userOp->getResults()) {
                        auto it = innerOuterOpValueMapping.find(result);
                        newOperands.push_back(it->second);
                        operandsIdxMapping[operandIdx].push_back(newOperands.size() - 1);
                    }
                }
                continue;
            }

            newOperands.push_back(operand);
            operandsIdxMapping[operandIdx].push_back(newOperands.size() - 1);
        }
    }

    // Add the output operands to the new vector of operands
    // Maintain a mapping between the outer output operand and its index in the new operands vector
    DenseMap<mlir::Value, int64_t> outputOperandsMapping;
    for (auto outputOperand : newOutputOperands) {
        newOperands.push_back(outputOperand);
        outputOperandsMapping[outputOperand] = newOperands.size() - 1;
    }

    // Get the mapping from the inner op operands to the new block operands
    DenseMap<mlir::Value, int64_t> innerOpOperandMapping;
    origOpBodyBlock.walk([&](mlir::Operation* op) {
        if (!isInnerOp(op)) {
            return;
        }

        for (auto p : op->getOperands() | indexed) {
            auto operand = p.value();
            if (auto blockArg = skipProducerCast(operand).dyn_cast<mlir::BlockArgument>()) {
                innerOpOperandMapping[operand] = operandsIdxMapping[blockArg.getArgNumber()].front();
            } else if (auto ungroupOp = operand.getDefiningOp<VPUIP::UngroupSparseBufferOp>()) {
                auto ungroupInput = skipProducerCast(ungroupOp.getInput());
                auto blockArg = ungroupInput.dyn_cast<mlir::BlockArgument>();
                VPUX_THROW_WHEN(blockArg == nullptr, "Unexpected operand for ungrouping operation");
                auto resultIt = llvm::find(ungroupOp->getResults(), operand);
                auto resultIdx = std::distance(ungroupOp->getResults().begin(), resultIt);
                innerOpOperandMapping[operand] = operandsIdxMapping[blockArg.getArgNumber()][resultIdx];
            } else if (VPUIP::isPureViewOp(operand.getDefiningOp()) &&
                       !mlir::isa<VPUIP::GroupSparseBufferOp>(operand.getDefiningOp())) {
                auto opInput = skipProducerCast(operand.getDefiningOp()->getOperands().front());
                auto blockArg = opInput.dyn_cast<mlir::BlockArgument>();
                VPUX_THROW_WHEN(blockArg == nullptr, "Unexpected operand for pureView operation");

                innerOpOperandMapping[operand] = operandsIdxMapping[blockArg.getArgNumber()].front();
            } else {
                auto outerOperand = innerOuterOpValueMapping[operand];
                innerOpOperandMapping[operand] = outputOperandsMapping[outerOperand];
            }
        }
    });

    // Compute the new resulting types of the bufferized cluster tiling op
    SmallVector<mlir::Type> newClusterTilingResultTypes;
    for (auto outputOperand : newOutputOperands) {
        newClusterTilingResultTypes.push_back(outputOperand.getType());
    }

    // Create a VPUIP::NCEClusterTiling operation whose body contains only the inner operation
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location /*loc*/, mlir::ValueRange newArgs) {
        for (auto& op : origOpBodyBlock.getOperations()) {
            if (!isInnerOp(&op)) {
                continue;
            }

            log.nest().trace("Cloning op '{0}' at '{1}", op.getName(), op.getLoc());

            mlir::IRMapping mapper;
            for (auto operand : op.getOperands()) {
                auto it = innerOpOperandMapping.find(operand);
                VPUX_THROW_WHEN(it == innerOpOperandMapping.end(), "No mapping found for operand of inner op");
                mapper.map(operand, newArgs[it->second]);
            }
            builder.clone(op, mapper);
        }
    };
    auto clusterTilingOp = rewriter.create<VPUIP::NCEClusterTilingOp>(origOp->getLoc(), newClusterTilingResultTypes,
                                                                      newOperands, bodyBuilder);

    auto outputResults = clusterTilingOp.getResults();

    // Add an output grouping operation if the original cluster tiling operation returned a grouped result
    mlir::Operation* outputGroupingOp = nullptr;
    origOpBodyBlock.walk([&](VPUIP::GroupSparseBufferOp groupOp) {
        if (llvm::none_of(groupOp->getOperands(), [](mlir::Value operand) {
                return isBufAllocOp(operand.getDefiningOp());
            })) {
            VPUX_THROW_UNLESS(outputGroupingOp == nullptr, "Multiple grouping operations are not allowed");
            outputGroupingOp = groupOp;
        }
    });
    if (outputGroupingOp != nullptr) {
        auto gop = mlir::cast<VPUIP::GroupSparseBufferOp>(outputGroupingOp);

        // TODO: #-122030 Remove call to these builders when possible.
        // We have to do this to avoid calling the broken properties builder.
        mlir::Value data = outputResults.size() > 0 ? outputResults[0] : nullptr;
        mlir::Value sparsityMap = outputResults.size() > 1 ? outputResults[1] : nullptr;
        mlir::Value storageElementTable = outputResults.size() > 2 ? outputResults[2] : nullptr;

        auto groupOp = rewriter.create<VPUIP::GroupSparseBufferOp>(outputGroupingOp->getLoc(), data, sparsityMap,
                                                                   storageElementTable, gop.getIsWeights(),
                                                                   gop.getSparsityCompressionAttr());

        outputResults = groupOp->getResults();
    }

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, outputResults);

    return mlir::success();
}

//
// registerNCEClusterTilingBufferizableOpInterfaces
//

void vpux::registerNCEClusterTilingBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::NCEClusterTilingOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEClusterTilingOp>>(*ctx);
    });
}
