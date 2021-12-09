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

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

std::string getValueForLog(mlir::Value val) {
    if (const auto arg = val.dyn_cast<mlir::BlockArgument>()) {
        return llvm::formatv("BlockArgument #{0} at '{0}'", arg.getArgNumber(), val.getLoc()).str();
    }

    const auto res = val.cast<mlir::OpResult>();
    return llvm::formatv("Operation result #{0} for '{1}' at '{2}'", res.getResultNumber(), res.getOwner()->getName(),
                         val.getLoc());
}

}  // namespace

vpux::AliasesInfo::AliasesInfo(mlir::FuncOp func): _log(Logger::global().nest("aliases-info", 0)) {
    _log.trace("Analyze aliases for Function '@{0}'", func.getName());
    _log = _log.nest();

    _log.trace("Function arguments are roots for themselves");
    _log = _log.nest();
    for (const auto funcArg : func.getArguments()) {
        _log.trace("Argument #{0}", funcArg.getArgNumber());

        VPUX_THROW_UNLESS(funcArg.getType().isa<mlir::MemRefType>(),
                          "AliasesInfo analysis works only with MemRef types, got '{0}'", funcArg.getType());
        addAlias(funcArg, funcArg);
    }
    _log = _log.unnest();

    _log.trace("Traverse the Function body");
    _log = _log.nest();
    traverse(func.getOps());
}

mlir::Value vpux::AliasesInfo::getSource(mlir::Value val) const {
    const auto it = _sources.find(val);
    VPUX_THROW_UNLESS(it != _sources.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    return it->second;
}

mlir::Value vpux::AliasesInfo::getRoot(mlir::Value val) const {
    const auto it = _roots.find(val);
    VPUX_THROW_UNLESS(it != _roots.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    return it->second;
}

const AliasesInfo::ValuesSet& vpux::AliasesInfo::getAllAliases(mlir::Value val) const {
    const auto it = _allAliases.find(val);
    VPUX_THROW_UNLESS(it != _allAliases.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    return it->second;
}

void vpux::AliasesInfo::addAlias(mlir::Value source, mlir::Value alias) {
    _log.trace("Add an alias '{0}' for '{1}'", getValueForLog(alias), getValueForLog(source));

    const auto root = source == alias ? alias : getRoot(source);

    if (alias == root) {
        _sources.insert({alias, nullptr});
    } else {
        _sources.insert({alias, source});
    }

    _roots.insert({alias, root});
    _allAliases[root].insert(alias);
}

void vpux::AliasesInfo::traverse(OpRange ops) {
    for (auto& op : ops) {
        llvm::TypeSwitch<mlir::Operation*, void>(&op)
                .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewOp) {
                    _log.trace("Got ViewLike Operation '{0}' at '{1}'", viewOp->getName(), viewOp->getLoc());
                    _log = _log.nest();

                    const auto result = viewOp->getResult(0);
                    const auto source = viewOp.getViewSource();

                    VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                                      "AliasesInfo analysis works only with MemRef types, got '{0}'", result.getType());
                    VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(),
                                      "AliasesInfo analysis works only with MemRef types, got '{0}'", source.getType());

                    addAlias(source, result);

                    _log = _log.unnest();
                })
                .Case<MultiViewOpInterface>([&](MultiViewOpInterface viewOp) {
                    _log.trace("Got MultiView Operation '{0}' at '{1}'", viewOp->getName(), viewOp->getLoc());
                    _log = _log.nest();

                    for (const auto result : viewOp->getResults()) {
                        _log.trace("Result #{0}", result.getResultNumber());

                        VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'",
                                          result.getType());

                        const auto source = viewOp.getViewSource(result.getResultNumber());
                        if (source == nullptr) {
                            addAlias(result, result);
                            continue;
                        }

                        VPUX_THROW_UNLESS(source.getType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'",
                                          source.getType());

                        addAlias(source, result);
                    }

                    _log = _log.unnest();
                })
                .Case<mlir::RegionBranchOpInterface>([&](mlir::RegionBranchOpInterface regionOp) {
                    _log.trace("Got RegionBranch Operation '{0}' at '{1}'", regionOp->getName(), regionOp->getLoc());
                    _log = _log.nest();

                    SmallVector<mlir::RegionSuccessor> entries;
                    regionOp.getSuccessorRegions(None, entries);

                    for (const auto& entry : entries) {
                        auto* entryRegion = entry.getSuccessor();
                        VPUX_THROW_UNLESS(entryRegion != nullptr,
                                          "Entry region without an attached successor region at '{0}'",
                                          regionOp->getLoc());

                        const auto outerArgs = regionOp.getSuccessorEntryOperands(entryRegion->getRegionNumber());
                        const auto innerArgs = entry.getSuccessorInputs();

                        VPUX_THROW_UNLESS(
                                outerArgs.size() == innerArgs.size(),
                                "Mismatch between RegionBranch operands and its entry region arguments at '{0}'",
                                regionOp->getLoc());

                        for (auto i : irange(outerArgs.size())) {
                            _log.trace("Check operand #{0} and corresponding region argument", i);

                            addAlias(outerArgs[i], innerArgs[i]);
                        }
                    }

                    _log.trace("Traverse the RegionBranch inner regions");
                    _log = _log.nest();
                    for (auto& region : regionOp->getRegions()) {
                        traverse(region.getOps());
                    }
                    _log = _log.unnest();

                    for (auto& region : regionOp->getRegions()) {
                        SmallVector<mlir::RegionSuccessor> successors;
                        regionOp.getSuccessorRegions(region.getRegionNumber(), successors);

                        for (auto& successor : successors) {
                            Optional<unsigned> regionIndex;
                            if (auto* regionSuccessor = successor.getSuccessor()) {
                                regionIndex = regionSuccessor->getRegionNumber();
                            }

                            for (auto& block : region) {
                                auto successorOperands =
                                        mlir::getRegionBranchSuccessorOperands(block.getTerminator(), regionIndex);

                                if (successorOperands.hasValue()) {
                                    const auto innerResults = successorOperands.getValue();
                                    const auto outerResults = successor.getSuccessorInputs();

                                    VPUX_THROW_UNLESS(
                                            innerResults.size() == outerResults.size(),
                                            "Mismatch between successor operands and its parent results at '{0}'",
                                            regionOp->getLoc());

                                    for (auto i : irange(innerResults.size())) {
                                        _log.trace("Check result #{0} and corresponding region result", i);

                                        addAlias(innerResults[i], outerResults[i]);
                                    }

                                    _log = _log.unnest();
                                }
                            }
                        }
                    }
                })
                .Case<mlir::async::AwaitOp>([&](mlir::async::AwaitOp waitOp) {
                    _log.trace("Got 'async.await' Operation at '{0}'", waitOp->getLoc());
                    _log = _log.nest();

                    if (const auto result = waitOp.result()) {
                        const auto futureType = waitOp.operand().getType().dyn_cast<mlir::async::ValueType>();
                        VPUX_THROW_UNLESS(futureType != nullptr,
                                          "AliasesInfo analysis works only with !async.value<MemRef> types, got '{0}'",
                                          waitOp.operand().getType());

                        VPUX_THROW_UNLESS(futureType.getValueType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'", futureType);
                        VPUX_THROW_UNLESS(result.getType().isa<mlir::MemRefType>(),
                                          "AliasesInfo analysis works only with MemRef types, got '{0}'",
                                          result.getType());

                        addAlias(waitOp.operand(), result);
                    }

                    _log = _log.unnest();
                })
                .Default([&](mlir::Operation* op) {
                    _log.trace("Got generic Operation '{0}' at '{1}'", op->getName(), op->getLoc());
                    _log = _log.nest();

                    for (const auto result : op->getResults()) {
                        if (result.getType().isa<mlir::MemRefType>()) {
                            addAlias(result, result);
                        }
                    }

                    _log = _log.unnest();
                });
    }
}
