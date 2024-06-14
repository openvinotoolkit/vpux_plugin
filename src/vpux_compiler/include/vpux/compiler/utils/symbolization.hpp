//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <initializer_list>
#include <utility>

namespace vpux {

// A sub-specialization of the default OpConversionPattern, dedicated for symbolization conversions.
// An opConversion pattern has hooks for type system conversions. Symbolization pattern is a form of type conversion
// where OpOperand relationships can materialize as symbolic relationships. The typeConverter does not natively support
// such semantics, since symolization relationships are materialized by attributes. However, it is still required, as
// even tough opOperand relationships dissapear, we would need to register types that would need intermediate
// materialization. The SymbolizationPattern overrides the default matchAndRewrite conversion hook and provides two
// additional hooks:
//  - getSymbolicName: called out for each Op (at initialization stage) each OP should provide a unique symbolic name
//  - symbolize: the actual conversion method, where instead of the Operand adaptor, we provide a symbolic map
//                  that can be used to lookup an ops symbolic name
template <typename SourceOp>
class SymbolizationPattern : public mlir::OpConversionPattern<SourceOp> {
public:
    using BaseOpT = SourceOp;
    using OpAdaptor = typename mlir::OpConversionPattern<SourceOp>::OpAdaptor;
    using SymbolMapper = typename llvm::DenseMap<mlir::Value, mlir::FlatSymbolRefAttr>;
    SymbolizationPattern(mlir::func::FuncOp parentFunc, mlir::TypeConverter& typeConverter, SymbolMapper& mapper,
                         mlir::MLIRContext* ctx)
            : mlir::OpConversionPattern<SourceOp>(typeConverter, ctx), _parentFunc(parentFunc), _mapper(&mapper) {
    }

    virtual mlir::LogicalResult symbolize(SourceOp op, SymbolMapper& mapper,
                                          mlir::ConversionPatternRewriter& rewriter) const = 0;

    virtual mlir::FlatSymbolRefAttr getSymbolicName(SourceOp op, size_t counter) = 0;

    void initialize() {
        size_t counter = 0;
        for (auto op : _parentFunc.getOps<SourceOp>()) {
            if (auto symbolicName = getSymbolicName(op, counter)) {
                _mapper->insert(std::make_pair(op.getResult(), symbolicName));
            }
            counter++;
        }
    }

protected:
    mlir::FlatSymbolRefAttr findSym(mlir::Value val) const;

private:
    mlir::LogicalResult matchAndRewrite(SourceOp op, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    mlir::func::FuncOp _parentFunc;
    SymbolMapper* _mapper;
};

template <typename SourceOp>
mlir::LogicalResult SymbolizationPattern<SourceOp>::matchAndRewrite(SourceOp op, OpAdaptor,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
    return symbolize(op, *_mapper, rewriter);
}

template <typename SourceOp>
mlir::FlatSymbolRefAttr SymbolizationPattern<SourceOp>::findSym(mlir::Value val) const {
    auto it = _mapper->find(val);

    VPUX_THROW_WHEN(it == _mapper->end(), "Could not find symbol name entry for {0}, val {1}",
                    SourceOp::getOperationName(), val);

    return it->getSecond();
}

// Wrapper around the native RewritePatternSet, which adds the extra verification that each pattern added is a
// symbolizationPattern. By current design, SymbolizationPatterns are designed to be exclusive. Mixing symbolization
// patterns with simple OpConversion patterns can prove to be too complex with a lot of potential corner cases.
// While symbolizationPatterns should work with the base RewritePatterSet, it is recommended to use
// SymbolizationPatternSet
class SymbolizationPatternSet : private mlir::RewritePatternSet {
    using NativePatternListT = std::vector<std::unique_ptr<mlir::RewritePattern>>;

public:
    // Intentionally do not automatically inherit all constructors,
    SymbolizationPatternSet(mlir::MLIRContext* context): mlir::RewritePatternSet(context) {
    }

    template <typename OpT>
    SymbolizationPatternSet(mlir::MLIRContext* context, std::unique_ptr<SymbolizationPattern<OpT>> pattern)
            : mlir::RewritePatternSet(context, pattern) {
    }

    mlir::MLIRContext* getContext() const {
        return mlir::RewritePatternSet::getContext();
    }

    static mlir::FrozenRewritePatternSet freeze(SymbolizationPatternSet&& symbolPatterns) {
        // mlir::RewritePatternSet &patterns = *this;
        return mlir::FrozenRewritePatternSet(std::move(symbolPatterns));
    }

    NativePatternListT& getNativePatterns() {
        return RewritePatternSet::getNativePatterns();
    }

    //===--------------------------------------------------------------------===//
    // 'add' methods for adding patterns to the set.
    //===--------------------------------------------------------------------===//

    /// Add an instance of each of the pattern types 'Ts' to the pattern list with
    /// the given arguments. Return a reference to `this` for chaining insertions.
    /// Note: ConstructorArg is necessary here to separate the two variadic lists.
    template <typename... Ts, typename ConstructorArg, typename... ConstructorArgs,
              typename = std::enable_if_t<sizeof...(Ts) != 0>>
    SymbolizationPatternSet& add(ConstructorArg&& arg, ConstructorArgs&&... args) {
        // The following expands a call to emplace_back for each of the pattern
        // types 'Ts'. This magic is necessary due to a limitation in the places
        // that a parameter pack can be expanded in c++11.
        // FIXME: In c++17 this can be simplified by using 'fold expressions'.
        (void)std::initializer_list<int>{
                0, (addImpl<Ts, typename Ts::BaseOpT>(/*debugLabels=*/std::nullopt, arg, args...), 0)...};
        return *this;
    }
    /// An overload of the above `add` method that allows for attaching a set
    /// of debug labels to the attached patterns. This is useful for labeling
    /// groups of patterns that may be shared between multiple different
    /// passes/users.
    template <typename... Ts, typename ConstructorArg, typename... ConstructorArgs,
              typename = std::enable_if_t<sizeof...(Ts) != 0>>
    SymbolizationPatternSet& addWithLabel(ArrayRef<StringRef> debugLabels, ConstructorArg&& arg,
                                          ConstructorArgs&&... args) {
        // The following expands a call to emplace_back for each of the pattern
        // types 'Ts'. This magic is necessary due to a limitation in the places
        // that a parameter pack can be expanded in c++11.
        // FIXME: In c++17 this can be simplified by using 'fold expressions'.
        (void)std::initializer_list<int>{0, (addImpl<Ts, typename Ts::BaseOpT>(debugLabels, arg, args...), 0)...};
        return *this;
    }

    /// Add an instance of each of the pattern types 'Ts'. Return a reference to
    /// `this` for chaining insertions.
    template <typename... Ts>
    SymbolizationPatternSet& add() {
        (void)std::initializer_list<int>{0, (addImpl<Ts, Ts::BaseOpT>(), 0)...};
        return *this;
    }

    /// Add the given native SymbolizationPattern to the pattern list. Return a reference to
    /// `this` for chaining insertions.
    template <typename OpT>
    SymbolizationPatternSet& add(std::unique_ptr<SymbolizationPattern<OpT>> pattern) {
        RewritePatternSet::add(std::move(pattern));
        return *this;
    }

private:
    /// Add an instance of the pattern type 'T'. Return a reference to `this` for
    /// chaining insertions.
    template <typename T, typename OpT, typename... Args>
    std::enable_if_t<std::is_base_of<SymbolizationPattern<OpT>, T>::value> addImpl(ArrayRef<StringRef> debugLabels,
                                                                                   Args&&... args) {
        RewritePatternSet::addWithLabel<T>(debugLabels, std::forward<Args>(args)...);
    }
};

}  // namespace vpux
