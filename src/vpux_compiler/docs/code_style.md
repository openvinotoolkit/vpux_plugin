# Style guide
This guide is not intended to cover all possible coding problems. Its ultimate goal is:
- to formalize the existing practices;
- to collect some useful programming techniques aimed to enhance the readability and maintainability of our code base.

## Naming

### Variables and functions

The names of common variables (including function parameters) and free functions are all `lowerCamelCase` aka `camelBack`:

```cpp
int someVariable;  // OK
int SomeVariable;  // BAD: it's UpperCamelCase, we only use it for class and struct names
int some_variable; // BAD: it's lower_snake_case
int somevariable;  // TOO BAD

void someFunction(int functionOperand) {                  // OK
    int localVariable = anotherFunction(functionOperand); // OK
    auto lambdaFunction = [](int lambdaOperand) {         // OK
        // <...>
    };
    // <...>
```

### Structures

Structures' names are `UpperCamelCase` aka `PascalCase`
Their data members (incl. `static` ones) and methods are named like ordinary non-member variables and functions:

```cpp
struct SomeStructure final {            // OK: UpperCamelCase for structure names
    void someMethod(int methodOperand); // OK: lowerCamelCase for structure data members and methods
    int dataMember;
    static int staticDataMember;
};
```

### Classes

Classes themselves are named the same way as structures.
The only difference is that their data members (incl. `static` ones) have **an initial underscore (!)**:

```cpp
class SomeClass final { // OK: UpperCamelCase for class names
public:
    void someMethod(int methodOperand) const; // OK: lowerCamelCase for methods

private:
    int _classDataMember; // OK: underscored _lowerCamelCase for all data members of classes
    static int _staticDataMember;
};
```

### Semantic correctness

Do your best to name the things in a most descriptive way, even when it makes the names longer.
Some common abbreviations are acceptable when you're sure they cannot be misunderstood.
Ideally, the reader who finds a use-case of your object should be able to figure out what it is and why it's there with no need to look for the place where it was defined or initialized.

These tips may help:
- Focus on describing **"what exactly it is?"** when naming a thing.
- Stick to using **nouns and adjectives for objects** but **verbs and adverbs for functions**: `auto goodDream = person.sleepWell()`.
- For Booleans, stick to using **statement = question()** form: `const bool somethingIsDone = isSomethingDone()`.
- Ask yourself if an _unprepared reader will get the point_ without having to delve too deeply into the surrounding context.

```cpp
int sourceStride; // OK, just minor context understanding is required (e.g. read the function name)
int srcStride;    // OK: "src" is quite commonly used for "source", same as "num" for "number", etc.

int sStride;      // BAD: "s" for what?
int intSrcStride; // BAD: do not mention type in the name, it can be easily infered from the code in any modern IDE
int ss, t, a, k;  // TOO BAD

// BAD: what are these things? Need to go see all the three definitions (lucky if they're 'const's at least)
if (temp / len < maxCount) {
    // <...>
}
```

## Operators

It is preferred to use symbols in place of keywords for operators i.e. `&&` should be used in place of `and`.

```cpp
if (sourceOp != nullptr and sourceOp->getResult(0).use_empty()) { // BAD
    // <...>
}

if (sourceOp != nullptr && sourceOp->getResult(0).use_empty()) { // OK
    // <...>
}
```

- `and` => `&&`
- `or` => `||`
- `not` => `!`

Check [here](https://en.cppreference.com/w/cpp/language/operator_alternative) for more examples.

## Methods

### Arguments

#### Base types

Usually the types provided by MLIR framework such as `mlir::Value`, `mlir::Type`, `mlir::Attribute`
are lightweight objects that only contain a pointer to the implementation:

```cpp
protected:
    ImplType *impl;
```

Pass such types **by value**, not by reference
The same rule applies to all operations from MLIR dialects: `IE`, `IERT`, `VPUIP`, etc.

```cpp
void someFunction(mlir::Value functionOperand);  // OK
void someFunction(const mlir::Value& functionOperand); // BAD: no performance gain, so pass it by value instead
```

#### Arrays
If you are passing an array to a function that does not modify it, use `mlir::ArrayRef`.

> **Be aware:** This class does not own the underlying data, it is only expected to be used in
situations where the data resides in some other buffer whose lifetime extends past that of
the `ArrayRef`. Thus, it is not generally safe to store an `ArrayRef`.

```cpp
void justPrintArray(ArrayRef<mlir::Value> array);           // OK
void justPrintArray(const SmallVector<mlir::Value>& array); // BAD: pass it by ArrayRef instead
```

## Formatting

### General Rules

- Four space indentation
- Trailing whitespace should be stripped

> Note: Most editors allow you to setup automatic space insertion and trailing whitespace stripping.

### Braces

Follow [1TBS aka OTBS](https://en.wikipedia.org/wiki/Indentation_style#Variant:_1TBS_(OTBS)) ("one true brace style").
In other words, use brackets with all constructions: function definition, `if`, `else`, `while`, etc.
The readability of this principle is debatable. But this allows us to achieve that insertion of a new line of code anywhere is always safe.

```cpp
void someFunction(mlir::Operation* op) { // OK
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) { // OK
        return;
    }

    // <...>
}

void someFunction(mlir::Operation* op)  // BAD: the opening bracket is at the next line but it should be here
{
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) // BAD: even single-line constructions must have brackets
        return;
}

```

### Surplus namespaces

Do not use `llvm::` (or other) namespace prefix for classes imported into `vpux` namespace.
You can find some of these classes in the [utils](../../../src/vpux_utils/include/vpux/utils/core).

```cpp
void someFunction() {
    SmallVector<mlir::Value> oldResults;       // OK
    llvm::SmallVector<mlir::Value> newResults; // BAD: no need to use llvm namespace
}
```

### Null-pointer checking

To check the validity of a pointer, prefer the most explicit notation i.e. `nullptr` comparison.

```cpp
void someFunction(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) { // OK
        // <...>
    }
    if (layer != nullptr) { // OK
        // <...>
    }
    if (!layer) { // BAD: implicit ptr->bool conversion
        // <...>
    }
}
```

### Explicit names

It is often better for readability to be explicit by giving names to booleans in `if` checks.

```cpp
// BAD: It is not immediately clear what this `if` is checking for.
if (llvm::any_of(layerOp.output().getUsers(), [](auto userOp) {
    return mlir::isa<IE::TransposeOp, IE::ExpandOp, IE::ConcatOp, IE::ReshapeOp>(userOp);
})) {
    return mlir::failure();
}
```

```cpp
// OK: Here we use an explicit name for this check so the intent is clearer.
auto hasRestrictedUsers = llvm::any_of(layerOp.output().getUsers(), [](auto userOp) {
    return mlir::isa<IE::TransposeOp, IE::ExpandOp, IE::ConcatOp, IE::ReshapeOp>(userOp);
});

if (hasRestrictedUsers) {
    return mlir::failure();
}
```

```cpp
// OK: Intent here is still clear because code is self documenting but above style should be preferred.
auto isRestrictedUsers = [](mlir::Operation* userOp) {...};

if (llvm::any_of(layerOp.output().getUsers(), isRestrictedUsers)) {
    return mlir::failure();
}
```

### Understandability: code structure and comments

> _...the ratio of time spent reading versus writing is well over 10 to 1. We are constantly reading old code as part of the effort to write new code._
>    _Robert C. Martin, "Clean Code: A Handbook of Agile Software Craftsmanship"_

Listen to your feelings when you read the code. If you feel like _something here might appear to be not instantly obvious for an unprepared reader_, it's a good reason to intervene. Change the code according to the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle), make it self-documenting, use comments to explain the reasons and choices made.

These tips may help:
- If the possible question starts with **"what"** or **"how"**, the best option usually is to **change the code** itself.
  Examples: _what is going on? How it works? What is this thing? How it's supposed to be used?_
  You can simplify the code, factor out a method or function, rename something for proper self-documenting, etc.
  If change is impossible, add a descriptive comment at least.
- If the possible question starts with **"why"**, the best option usually is to add an **explanatory comment**.
  Examples: _why this thing is here but not there? Why it's done this way but not another? Why it's done at all?_
- It's also nice to add header comments for classes, functions and methods.
  Focus on brief answers for possible **"what?"**s and **"why?"**s required to scroll through the code fluently.
  Examples: _what do I expect from this thing? Why it's here but not there?_
  If the reader wants to know **"how?"**, he/she can dive into the implementation.
- You can use Doxygen comments notation if you think it makes your comments more verbose and explicit.

```cpp
/// @brief Splits a Copy operation into C-wise tiles                    // OK: "What does this thing do?"
/// @details Currently, only C-wise tiling is implemented for 4D shapes // OK: "Why it does what it does?"
/// @return Outputs of the smaller Copy operations created              // OK: "What should I expect from it?"
/// @todo Replace this with a re-use of the generic tiling utilities
SmallVector<mlir::Value> CopyOpTiling::createChannelWiseTiles(VPUIP::CopyOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    // <...>

    const auto numTilesRequired =
            divUp(origInputShape[Dims4D::Act::C],
                  VPUIP::DMA_LIMIT.count() / (fullCopySize / origInputShape[Dims4D::Act::C]).count());
    // BAD. The reader's experience would be something like:
    // 1. What's going on here? O_o
    // 2. ...a plenty of time and nerves is spent to answer 1.
    // 3. Why so difficult? Maybe just divide the full size by the limit value?

    /// Split the shape into packs of full channels
    const auto numTilesRequired =
            divUp(origInputShape[Dims4D::Act::C],
                  VPUIP::DMA_LIMIT.count() / (fullCopySize / origInputShape[Dims4D::Act::C]).count());
    // STILL BAD: do not describe weird code in comments, rewrite it!

    /// We cannot _just_ divide the fullCopySize by sizeLimit to get the number of tiles required
    /// @example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but
    ///           it's obviously impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numChannelsOfFullShape = origInputShape[Dims4D::Act::C];
    const auto singleChannelSize = fullCopySize / numChannelsOfFullShape;
    const auto numChannelsPerTile = VPUIP::DMA_LIMIT.count() / singleChannelSize.count();
    const auto numTilesRequired = divUp(numChannelsOfFullShape, numChannelsPerTile);
    // OK: comments for reasoning, self-documenting code (though you can improve it of course)
```

### Headers

Header files which are in the same source distribution should be included in "quotes".
Header files for system and other external libraries should be included in <angle brackets>.

Headers with angle brackets also allow MSVC compiler apply different warning and error rules for them.


Example:
```cpp
#include <string>
#include <mlir/IR/Dialect.h>
#include <llvm/Support/TargetSelect.h>

#include "vpux/compiler/init.hpp"
```

## Patterns

### Return Early Pattern

["Return Early"](https://medium.com/swlh/return-early-pattern-3d18a41bba8) is the way of writing functions or methods
so that the expected result is returned at the end of the function while the rest of the code terminates the execution
(by returning or throwing an exception) if conditions are not met. It can be considered as a manifestation of the
Open-Closed Principle from SOLID [(OCP)](https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle).

```cpp
// OK: it's easy to manage fail-cases, therefore OCP is preserved
mlir::LogicalResult goodExample(VPUIP::SubViewOp origOp) {
    auto producerSubViewOp = origOp.source().getDefiningOp<VPUIP::SubViewOp>();
    if (producerSubViewOp == nullptr) {
        return mlir::failure();
    }

    if (origOp.static_strides().hasValue() || producerSubViewOp.static_strides().hasValue()) {
        return mlir::failure();
    }

    // <...>

    return mlir::success();
}

// BAD: to manage conditions, we have to move lots of code and introduce excess code nesting
mlir::LogicalResult badExample(VPUIP::SubViewOp origOp) {
    auto producerSubViewOp = origOp.source().getDefiningOp<VPUIP::SubViewOp>();
    if (producerSubViewOp != nullptr) {
        if (!origOp.static_strides().hasValue() && !producerSubViewOp.static_strides().hasValue()) {
            // <...>
                return mlir::success();
            // <...>
        }
    }

    return mlir::failure();
}
```

### Using checked casts

[checked_cast](../../vpux_utils/include/vpux/utils/core/checked_cast.hpp) is a helper which encapsulates run-time check
for `static_cast`. It is advisable to use it where applicable in order to avoid some unforeseen consequences during
casts. Otherwise, it maybe quite hard to investigate that class of problems. Consider the following code snippet:

```cpp
    int32_t i32 = -100;
    uint32_t u32 = (uint32_t)i32;
    assert((i32 / 10) == (u32 / 10));  // Assertion `(i32 / 10) == (u32 / 10)' failed.
```

It might be a bit syntetic, but it serves the purpose of illustration for unexpected execution flow quite obviously.
End-user may get unexpected accuracy regression without any notice. The problem would become more obvious with
`checked_cast`:

```cpp
    int32_t i32 = -100;
    uint32_t u32 = checked_cast<uint32_t>(i32); // Can not safely cast -100 from int32_t to uint32_t
    assert((i32 / 10) == (u32 / 10));
```

### Using ShapeRef

There is a difference between the Shape and ShapeRef type. ShapeRef does not own memory, while Shape does.
And Shape can be implicitly converted to ShapeRef, but not the other way around.
Thus, a typical problem is to store a local variable in an instance of the ShapeRef type and use it outside the scope of this variable.
So we get a dangling pointer:

```cpp
    // OK: The result from 'computerShape' object is stored in an instance of Shape type that owns memory
    Shape inputShape = getShape(origOp.input()).toValues(); // The return type of 'getShape' is ShapeRef,
                                                            // but we create an object of Shape type using 'toValues' method
    if (outputShape != getShape(nceOp->getResult(0))) {
        vpux::TileInfo outputTile{outputShape};
        auto computerShape = origOp.backInferTileInfo(outputTile, Logger::global());
        inputShape = computerShape.tiles[0].shape; // 'inputShape' stores a copy of field of local variable
    }

    return isSOHSupportedByDPU(inputShape, _numTiles, false, VPU::getArch(nceOp.getOperation()));

    // BAD: The result from 'computerShape' object is stored in an instance of ShapeRef type that does not own memory
    auto inputShape = getShape(origOp.input()); // The return type of 'getShape' is ShapeRef
    if (outputShape != getShape(nceOp->getResult(0))) {
        vpux::TileInfo outputTile{outputShape};
        auto computerShape = origOp.backInferTileInfo(outputTile, Logger::global());
        inputShape = computerShape.tiles[0].shape; // 'inputShape' stores the reference to field of local variable
    }

    // The lifetime of 'computerShape' is over. 'inputShape' contains a dangling pointer
    return isSOHSupportedByDPU(inputShape, _numTiles, false, VPU::getArch(nceOp.getOperation()));
```

### Using method 'llvm::make_early_inc_range'

This method is useful when you iterate through a list of objects and you need to change this range at the same time. From the description:
> Make a range that does early increment to allow mutation of the underlying
> range without disrupting iteration.
>
> The underlying iterator will be incremented immediately after it is
> dereferenced, allowing deletion of the current node or insertion of nodes to
> not disrupt iteration provided they do not invalidate the *next* iterator --
> the current iterator can be invalidated.
>
> This requires a very exact pattern of use that is only really suitable to
> range based for loops and other range algorithms that explicitly guarantee
> to dereference exactly once each element, and to increment exactly once each
> element.

```cpp
    // OK: The current operation has been erased, but the next object has already been prepared
    for (auto& op : llvm::make_early_inc_range(func.getOps())) {
        ....

        auto groupOp = builder.create<VPUIP::GroupSparseBufferOp>(op->getLoc(), dataResult, smResult, seTableResult,
                                                                  isWeightsAttr, sparsityCompressionAttr);
        op->getResult(0).replaceAllUsesExcept(groupOp.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{groupOp});
        op->erase();
    }

    // BAD: The current operation has been erased, the current iterator is invalid
    func.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
        ....

        auto groupOp = builder.create<VPUIP::GroupSparseBufferOp>(op->getLoc(), dataResult, smResult, seTableResult,
                                                              isWeightsAttr, sparsityCompressionAttr);
        op->getResult(0).replaceAllUsesExcept(groupOp.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{groupOp});
        op->erase();
    });
```

### Invalidation of MLIR operations

Methods of `mlir::PatternRewriter`, such as: `replaceOp`, `replaceOpWithNewOp`, and `eraseOp`, invalidate the
`mlir::Operation` passed to them as an argument. Avoid using the object after invalidation to prevent
undefined behavior.

```cpp
// OK: Print the location before invalidation.
_log.trace("{0}", origOp->getLoc());
rewriter.replaceOp(origOp, newOp->getResult(0));

// OK: Store a copy of the location before invalidation, then print it.
const auto loc = origOp->getLoc();
rewriter.replaceOp(origOp, newOp->getResult(0));
_log.trace("{0}", loc);

// BAD: origOp has been erased inside replaceOp. mlir::Operation::getLoc is not well-defined.
rewriter.replaceOp(origOp, newOp->getResult(0));
_log.trace("{0}", origOp->getLoc());
```

## Using AliasesInfo

[AliasesInfo](../include/vpux/compiler/core/aliases_info.hpp) is a util class that provides information about the
source variable for each alias. You can get an instance of `AliasesInfo` using `getAnalysis` method in a function pass:

```cpp
void MyFancyPass::safeRunOnFunc() {
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    // <...>
```

### Rules

#### Keep topological order
You have to maintain the topological order of the network throughout the entire time the class is used.
Otherwise, you may get operations not included in the analysis, or the analysis would become incorrect for the new network.
If you need to change the topological order, make sure that the subgraphs you are processing do not intersect.

These tips may help:
- Avoid using several patterns within one pass. This makes it easier to control changes in the topology.
- Prefer using `RetT Operation::walk(FnT &&callback)` rather than `mlir::OpRewritePattern<OpType>`.

##### AliasesInfo and Greedy pattern rewriter
Do not use `AliasesInfo` in the patterns controlled with `GreedyPatternRewriteDriver`.

```cpp
// OK
func->walk([&](VPUIP::CopyOp op) {
    if (!op.output_buff().isa<mlir::BlockArgument>()) {
        return;
    }

    auto& aliasInfo = getAnalysis<AliasesInfo>();
    fuseLastCopy(op, aliasInfo, _log);
});

// BAD: Greedy Pattern re-creates constants, therefore they will not be covered by the aliases analysis
void OptimizeCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto& aliasInfo = getAnalysis<AliasesInfo>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CopyToBlockArgument>(&ctx, aliasInfo, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
```

## Unit tests

### Using MLIR_UnitBase

There is a common class recommended as a base class for a unit test. It handles dialect registration, which is a shared
step for the compiler unit tests.
```cpp
// BAD
TEST(MLIR_IndexedSymbolAttr, CheckNestedAttr) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    ...
}

// OK
using MLIR_IndexedSymbolAttr = MLIR_UnitBase;
TEST_F(MLIR_IndexedSymbolAttr, CheckNestedAttr) {
    ...
}
```

## Pipelines and passes

### Adding rewriter to a pass

The IR traversal within a pass usually follows a specific direction: it is
either "top-down" (usual in programming e.g. in C++) or "bottom-up" (from the
end of the IR to the beginning). When adding a new rewriter to an existing pass
(or creating a new pass together with the rewriter), the rewriter's traversal
direction must match the pass' traversal direction.

```cpp
// IR structure to consider:
// func.func @foo(...) -> !outType {
//  %0 = VPUIP.NCEClusterTask(...)
//  %alloc = memref.alloc(): !outType
//  %1 = VPUIP.Copy inputs(%0) outputs(%alloc)
//  return %1
// }

struct MyBottomUpRewriter;
struct MyTopDownRewriter;
void MyTopDownPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MyBottomUpRewriter>(&ctx); // BAD: see below
    patterns.add<MyTopDownRewriter>(&ctx);  // OK

    auto func = getOperation();
    auto config = getDefaultGreedyRewriteConfig();
    config.useTopDownTraversal = true;
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

// BAD: the pass would first read the NCEClusterTask and only then CopyOp;
//      some other rewriter may end up changing the code *before* this
//      rewriter would be able to process the CopyOp!
struct MyBottomUpRewriter : mlir::OpRewritePattern<VPUIP::CopyOp> {
    MyBottomUpRewriter(mlir::MLIRContext* ctx)
        : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx) {}

    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final {
        // Remove a copy:
        //  NCEClusterTask          NCEClusterTask
        //        |          =>            |
        //       Copy
        //        |

        // ...
        return mlir::success();
    }
};

// OK: the pass would first read the NCEClusterTask and MyTopDownRewriter
//     would process this operation immediately.
struct MyTopDownRewriter : mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
    MyTopDownRewriter(mlir::MLIRContext* ctx)
        : mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx) {}

    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp taskOp, mlir::PatternRewriter& rewriter) const final {
        // Remove a copy:
        //  NCEClusterTask          NCEClusterTask
        //        |          =>            |
        //       Copy
        //        |

        // ...
        return mlir::success();
    }
};
```

For a more sophisticated example and reasoning of why this is important,
consider reading through [MLIR's good practices about
rewriters](./guides/mlir_good_practices.md#running-multiple-rewriters-within-one-pass).

### Pass integration
- Passes must be integrated as a part of a pipeline. Exceptions to this flow must be covered in comment and ticket
- Principles of combining passes in pipelines:
    - Passes that solve a complicated problem
    - Passes have dependencies
    ```cpp
    // OK: Passes are combined into a pipeline. Dependencies are obvious and can be tested.
    void vpux::IE::buildOptimizeActivationsPipeline(mlir::OpPassManager& pm, Logger log) {
        const auto grc = getDefaultGreedyRewriteConfig();

        pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations), log));
        pm.addPass(IE::createInsertIdentityPoolBeforeActivationPass(log));
        pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions37XX& options, Logger log) {
        // ...
        IE::buildOptimizeActivationsPipeline(pm, log)
        // ...
    }

    // BAD: There are a lot of passes around, the dependence is not obvious.
    // It's impossible to test the solution of the problem in isolation, need to test the entire DefaultHW pipeline.
    void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions37XX& options, Logger log) {
        // ...
        pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations), log));
        pm.addPass(IE::createInsertIdentityPoolBeforeActivationPass(log));
        pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        // ...
    }
    ```
- Passes with strict dependencies must be tested together usign LIT for pipelines
```cpp
// OK: Pass interactions are fixed. If the pipeline is changed, the test may show an error
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --optimize-activations %s | FileCheck %s

// BAD: Pass interactions are fixed, but the potential error will not be shown when changing the real pipeline(some passes could be added between SwapOperations and FusePostOps)
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --swap-operations --fuse-post-ops %s | FileCheck %s
```
- We allow calling pipelines multiples times to express dependencies in code
