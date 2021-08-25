//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <algorithm>

using namespace vpux;

namespace {

//
// Helpers
//

bool isOptimizableOp(mlir::async::ExecuteOp execOp) {
    auto module = execOp->getParentOfType<mlir::ModuleOp>();

    uint32_t numUnits = 0;
    const auto executor = vpux::IERT::IERTDialect::getExecutor(execOp, numUnits);

    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);
    auto executorInfo = resOp.getExecutor(executor);

    return numUnits == executorInfo.count() && executorInfo.subExecutors().empty();
}

bool isSameExecutor(mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
    uint32_t numUnits = 0;
    auto executor1 = vpux::IERT::IERTDialect::getExecutor(execOp1, numUnits);
    auto executor2 = vpux::IERT::IERTDialect::getExecutor(execOp2, numUnits);
    return executor1 == executor2;
}

//
// mergeAsyncExecuteOps
//

mlir::async::ExecuteOp mergeAsyncExecuteOps(mlir::async::ExecuteOp prevExecOp, mlir::async::ExecuteOp execOp,
                                            mlir::PatternRewriter& rewriter) {
    auto* prevBodyBlock = &prevExecOp.body().front();
    auto* bodyBlock = &execOp.body().front();

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
        mlir::BlockAndValueMapping mapper;

        const auto prevBlockArgs = prevBodyBlock->getArguments();
        const auto curBlockArgs = bodyBlock->getArguments();

        for (size_t i = 0; i < blockArgs.size(); ++i) {
            if (i < prevBlockArgs.size()) {
                mapper.map(prevBlockArgs[i], blockArgs[i]);
            } else {
                mapper.map(curBlockArgs[i - prevBlockArgs.size()], blockArgs[i]);
            }
        }

        SmallVector<mlir::Value> newResults;

        const auto copyOps = [&](mlir::Block* bodyBlock) {
            for (auto& op : bodyBlock->getOperations()) {
                if (!mlir::isa<mlir::async::YieldOp>(op)) {
                    builder.clone(op, mapper);
                } else {
                    for (auto operand : op.getOperands()) {
                        newResults.push_back(mapper.lookupOrDefault(operand));
                    }
                }
            }
        };

        copyOps(prevBodyBlock);
        copyOps(bodyBlock);

        builder.create<mlir::async::YieldOp>(loc, newResults);
    };

    const auto prevResultTypes = prevBodyBlock->getTerminator()->getOperandTypes();
    const auto resultTypes = bodyBlock->getTerminator()->getOperandTypes();

    SmallVector<mlir::Type> newResultTypes(prevResultTypes);
    newResultTypes.insert(newResultTypes.end(), resultTypes.begin(), resultTypes.end());

    SmallVector<mlir::Value> newDependencies(prevExecOp.dependencies());
    newDependencies.insert(newDependencies.end(), execOp.dependencies().begin(), execOp.dependencies().end());

    SmallVector<mlir::Value> newOperands(prevExecOp->getOperands());
    newOperands.insert(newOperands.end(), execOp->getOperands().begin(), execOp->getOperands().end());

    auto newExecOp = rewriter.create<mlir::async::ExecuteOp>(prevExecOp->getLoc(), newResultTypes, newDependencies,
                                                             newOperands, bodyBuilder);

    uint32_t numUnits = 0;
    auto executor = vpux::IERT::IERTDialect::getExecutor(execOp, numUnits);
    IERT::IERTDialect::setExecutor(newExecOp, executor, numUnits);

    return newExecOp;
}

//
// cleanup
//

void cleanup(mlir::async::ExecuteOp prevExecOp, mlir::async::AwaitOp prevWaitOp, mlir::async::ExecuteOp execOp,
             mlir::async::AwaitOp waitOp, mlir::async::ExecuteOp newExecOp, mlir::PatternRewriter& rewriter,
             Logger log) {
    log.trace("Clean up configuration:");
    log.nest().trace("prevExecOp.results.size = {0}", prevExecOp.results().size());
    log.nest().trace("execOp.results.size = {0}", execOp.results().size());
    log.nest().trace("newExecOp.results.size = {0}", newExecOp.results().size());
    log.nest().trace("prevWaitOp.results = {0}", prevWaitOp.result() != nullptr);
    log.nest().trace("waitOp.results = {0}", waitOp.result() != nullptr);

    log.trace("Insert new 'async.await' operation and replace orignal ones");

    mlir::async::AwaitOp newWaitOp0, newWaitOp1;
    if (prevWaitOp.result() != nullptr) {
        const auto prevExecRes = prevWaitOp.operand().dyn_cast<mlir::OpResult>();
        VPUX_THROW_UNLESS(prevExecRes != nullptr, "Got NULL pointer for prevExecRes");
        VPUX_THROW_UNLESS(prevExecRes.getOwner() == prevExecOp, "prevExecRes doesn't match prevExecOp");

        const auto prevResInd = prevExecRes.getResultNumber() - 1;
        log.nest().trace("Create new 'async.await' operation for prevExecRes.result #{0}", prevResInd);

        const auto newExecRes0 = newExecOp.results()[prevResInd];
        newWaitOp0 = rewriter.create<mlir::async::AwaitOp>(waitOp->getLoc(), newExecRes0);
    }
    if (waitOp.result() != nullptr) {
        const auto execRes = waitOp.operand().dyn_cast<mlir::OpResult>();
        VPUX_THROW_UNLESS(execRes != nullptr, "Got NULL pointer for execRes");
        VPUX_THROW_UNLESS(execRes.getOwner() == execOp, "execRes doesn't match execOp");

        const auto resInd = execRes.getResultNumber() - 1;
        log.nest().trace("Create new 'async.await' operation for execRes.result #{0}", resInd);

        const auto newExecRes1 = newExecOp.results()[prevExecOp.results().size() + resInd];
        newWaitOp1 = rewriter.create<mlir::async::AwaitOp>(waitOp->getLoc(), newExecRes1);
    }

    if (newWaitOp0 == nullptr && newWaitOp1 == nullptr) {
        log.nest().trace("Create new token-based 'async.await' operation");
        newWaitOp0 = newWaitOp1 = rewriter.create<mlir::async::AwaitOp>(waitOp->getLoc(), newExecOp.token());
    }

    log.nest().trace("Redirect original 'async.await' results");
    if (newWaitOp0 != nullptr) {
        rewriter.replaceOp(prevWaitOp, newWaitOp0->getResults());
    } else {
        rewriter.eraseOp(prevWaitOp);
    }
    if (newWaitOp1 != nullptr) {
        rewriter.replaceOp(waitOp, newWaitOp1->getResults());
    } else {
        rewriter.eraseOp(waitOp);
    }

    log.trace("Redirect results of original 'async.execute' operations");

    SmallVector<mlir::Value> matchedResultsPrev;
    SmallVector<mlir::Value> matchedResultsCurr;

    // newExecOp returns one token which replaces both tokens from original ops
    matchedResultsPrev.push_back(newExecOp.token());
    matchedResultsCurr.push_back(newExecOp.token());

    for (auto p : newExecOp.results() | indexed) {
        const auto ind = p.index();
        const auto newRes = p.value();

        if (ind < prevExecOp.results().size()) {
            matchedResultsPrev.push_back(newRes);
        } else {
            matchedResultsCurr.push_back(newRes);
        }
    }

    rewriter.replaceOp(prevExecOp, matchedResultsPrev);
    rewriter.replaceOp(execOp, matchedResultsCurr);
}

//
// GroupAsyncExecuteOps
//

class GroupAsyncExecuteOps final : public mlir::OpRewritePattern<mlir::async::AwaitOp> {
public:
    GroupAsyncExecuteOps(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::async::AwaitOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::AwaitOp waitOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupAsyncExecuteOps::matchAndRewrite(mlir::async::AwaitOp waitOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto execOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(waitOp->getPrevNode());
    if (execOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got 'async.execute' operation at '{0}'", execOp->getLoc());

    if (!isOptimizableOp(execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "The operation is not optimizible");
    }

    auto prevWaitOp = mlir::dyn_cast_or_null<mlir::async::AwaitOp>(execOp->getPrevNode());
    if (prevWaitOp == nullptr) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous operation is not 'async.await'");
    }

    auto prevExecOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(prevWaitOp->getPrevNode());
    if (prevExecOp == nullptr) {
        return matchFailed(_log.nest(), rewriter, execOp, "Operation before 'async.await' is not 'async.execute'");
    }

    if (prevWaitOp.operand().getDefiningOp() != prevExecOp) {
        return matchFailed(_log.nest(), rewriter, execOp,
                           "Previous 'async.await' does not correspond to previous 'async.execute'");
    }

    if (!isSameExecutor(prevExecOp, execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous 'async.execute' uses another executor");
    }

    /*  TODO: Remove check below when proper operands mapping is implemented.
        If current execute op depends on previous then it requires more complex operands/results mapping.
        Mapping path:
        1st execute op yield op operands -> 1st execute op result->
        -> 2nd execute op operand-> internal block argument-> operation operand */
    if (llvm::is_contained(prevExecOp->getUsers(), execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp,
                           "Current 'async.execute' depends on previous 'async.execute'");
    }

    auto newExecOp = mergeAsyncExecuteOps(prevExecOp, execOp, rewriter);
    cleanup(prevExecOp, prevWaitOp, execOp, waitOp, newExecOp, rewriter, _log.nest());

    return mlir::success();
}

//
// GroupAsyncExecuteOpsPass
//

class GroupAsyncExecuteOpsPass final : public IERT::GroupAsyncExecuteOpsBase<GroupAsyncExecuteOpsPass> {
public:
    explicit GroupAsyncExecuteOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void GroupAsyncExecuteOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GroupAsyncExecuteOps>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getFunction(), std::move(patterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createGroupAsyncExecuteOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createGroupAsyncExecuteOpsPass(Logger log) {
    return std::make_unique<GroupAsyncExecuteOpsPass>(log);
}
