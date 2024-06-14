# MLIR software layer enabling steps
- [MLIR software layer enabling steps](#mlir-software-layer-enabling-steps)
- [Introduction](#introduction)
- [Debugging tips and tricks](#debugging-tips-and-tricks)
- [Opset specification](#opset-specification)
  - [Examine layer specification](#examine-layer-specification)
- [Single layer test](#single-layer-test)
  - [Create a new file with a test](#create-a-new-file-with-a-test)
- [IE Dialect](#ie-dialect)
  - [IE Operation table gen](#ie-operation-table-gen)
  - [NGraph parser](#ngraph-parser)
  - [IE Output shape resolver](#ie-output-shape-resolver)
  - [Optional: IE Attribute parser](#optional-ie-attribute-parser)
  - [Optional: Canonicalizer](#optional-canonicalizer)
  - [Optional: Transformation pass](#optional-transformation-pass)
- [VPU Dialect](#vpu-dialect)
  - [VPU Operation table gen](#vpu-operation-table-gen)
  - [VPU Output shape resolver](#vpu-output-shape-resolver)
  - [Optional: Attributes, canonicalization, transformations](#optional-attributes-canonicalization-transformations)
- [IE → VPU lowering](#ie--vpu-lowering)
  - [IE → VPU lowering lit-test](#ie--vpu-lowering-lit-test)
- [You're half way there.](#youre-half-way-there)
- [GraphFile-schema](#graphfile-schema)
- [VPUIP Dialect](#vpuip-dialect)
  - [VPUIP table gen](#vpuip-table-gen)
  - [VPUIP UPATask builder](#vpuip-upatask-builder)
  - [Redirect interfaces for IE and VPUIP](#redirect-interfaces-for-ie-and-vpuip)
  - [VPUIP verifier](#vpuip-verifier)
- [VPU → VPUIP lowering](#vpu--vpuip-lowering)
  - [VPU → VPUIP lowering lit-test](#vpu--vpuip-lowering-lit-test)
- [IERT Dialect](#iert-dialect)
  - [IERT Table gen](#iert-table-gen)
- [IE → IERT lowering](#ie--iert-lowering)
  - [IE → IERT lowering lit-test](#ie--iert-lowering-lit-test)
# Introduction
This instruction will guide you through steps of adding a new NPU3700 software layer to the MLIR compiler. It has step-by-step plan of actions using `CTCGreedyDecoder` layer as an example. For NPU3720 software layer, please refer [npu2_7_sw_layer_enabling.md](../docs/npu2_7_sw_layer_enabling.md)
> Be aware, that MLIR compiler is in a rapid development and code snippets might be out of date.

# Debugging tips and tricks
Make sure to take a look at [debugging documentation](guides/how_to_debug.md) to have a common knowledge of technics and tools that will help you investigate problems when developing a layer.

# Opset specification
* [OpenVINO web site](https://docs.openvinotoolkit.org/latest/operations_specifications.html)
* [OpenVINO github](https://github.com/openvinotoolkit/openvino/tree/master/docs/ops)


## Examine layer specification

Let's implement [CTCGreedyDecoder](https://docs.openvinotoolkit.org/latest/openvino_docs_ops_sequence_CTCGreedyDecoder_1.html) operation from `OpenVINO opset-1`.

Even though, `ctc_merge_repeated` parameter is `Optional`, ov don't treat it as such.

https://github.com/openvinotoolkit/openvino/blob/master/src/core/include/openvino/op/ctc_greedy_decoder.hpp
If you found, that ov don't follow the operation specification, you should create a bug ticket.
Considering `CTCGreedyDecoder-1` supposed to be repalced with `CTCGreedyDecoderSeqLen-6`, we will ignore that inconsistency.

`CTCGreedyDecoder-1`:
Attributes:
* `ctc_merge_repeated` of type boolean.

Inputs:
* `data` is a floating point tensor of shape `[T, N, C]`.
* `sequence_mask` is a floating point tensor of shape `[T, N]`.

Outputs:
* `output` tensor with shape `[N, T, 1, 1]` and integer elements.

> Things to keep in mind:
> * Input count, size and type.
> * Output count, size and type.
> * Attribute types.
> * Are any of the above optional.

# Single layer test

Add OpenVINO single layer test. Copy test suites from the MKLDNN plugin for initial setup.

A simple test will be useful to have for debugging. Run it to see the build/compilation issues.
Make sure to derive `LayerTest` from `VpuOv2LayerTest`.

Useful links:
[How to run tests](../../../guides/how-to-test.md)

## Create a new file with a test
[tests/functional/shared_tests_instances/single_layer_tests/ctc_greedy_decoder.cpp](../../../tests/functional/shared_tests_instances/single_layer_tests/ctc_greedy_decoder.cpp)
```cpp
#include "single_op_tests/ctc_greedy_decoder.hpp"
#include "vpu_ov2_layer_test.hpp"
#include <vector>

using namespace ov::test;
using namespace ov::test::utils;

namespace LayerTestsDefinitions {
class CTCGreedyDecoderLayerTestCommon :
        public CTCGreedyDecoderLayerTest,
        virtual public VpuOv2LayerTest {};

class CTCGreedyDecoderLayerTest_NPU3700 :
        public CTCGreedyDecoderLayerTestCommon {
};

TEST_P(CTCGreedyDecoderLayerTest_NPU3700, HW) { // HW to reflect which pipeline is used, in this case it is DefaultHW
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

}  // namespace LayerTestsDefinitions

using namespace ov::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::element::Type> modelType = {
    ov::element::f32,
};

const std::vector<bool> mergeRepeated = {true, false};

const std::vector<std::vector<ov::Shape>> inputShapes = {
    std::vector<ov::Shape>{{ 88, 1, 71 }},
    std::vector<ov::Shape>{{ 10, 1, 16 }},
};

const auto params = testing::Combine(
    testing::ValuesIn(modelType),
    testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
    testing::ValuesIn(mergeRepeated),
    testing::Values(DEVICE_NPU)
);

INSTANTIATE_TEST_CASE_P(
    smoke_CTCGreedyDecoder,
    CTCGreedyDecoderLayerTest_NPU3700,
    params,
    CTCGreedyDecoderLayerTest::getTestCaseName
);
```

# IE Dialect
The IE Dialect represents InferenceEngine/nGraph IR in terms of MLIR framework.

It has the following properties:

* Describes network topology without HW details (memory hierarchy, memory allocation, scheduling).
* Represents the latest nGraph opset and in addition some portion of legacy IE opset (for convenience).
* Works with MLIR Tensor Types as atomic Values (no memory effects), all operations are pure.
* Performs high level transformations/optimizations, that doesn't need low level details (memory buffers, scheduling).

Documentation

* [chapters/generated/dialect/_IE.md](chapters/generated/dialect/_IE.md)

## IE Operation table gen
Let's create a table-gen representation of our layer.

* let summary – one line description of op
* let arguments – input parameters for the layer. Possible types can be found here: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td
* let results – outputs of the operation

[src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/ops.td#L2752](../tblgen/vpux/compiler/dialect/IE/ops.td#L2752)
```swift
//
// CTCGreedyDecoderOp
//

def IE_CTCGreedyDecoderOp :
        IE_LayerOp<
            "CTCGreedyDecoder",
            [
                ResultsAreFloatLike
            ]
        > {
    let summary = "InferenceEngine CTCGreedyDecoder layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$sequenceLengths,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}
```

## NGraph parser

Define parseNode function, that will transform ov operation to MLIR representation.

[src/vpux_compiler/src/frontend/IE.cpp#L151](../src/frontend/IE.cpp#L151)
```cpp
class NGraphImporter final {
public:
    NGraphImporter(mlir::MLIRContext* ctx, std::shared_ptr<const ov::Model> netGraph, bool sharedConstants,
                   Logger log)
            : _ctx(ctx), _netGraph(std::move(netGraph)), _sharedConstants(sharedConstants), _log(log) {
    }

    // Declare parser for ov operation
    void parseNode(mlir::OpBuilder& builder, const std::shared_ptr<ov::opset7::CTCGreedyDecoder>& origNode);
}
```
Check input tensors and parse ov operation.

[src/vpux_compiler/src/frontend/IE.cpp#L1167](../src/frontend/IE.cpp#L1167)
```cpp
void NGraphImporter::parseNode(mlir::OpBuilder& builder,
                               const std::shared_ptr<ov::opset7::CTCGreedyDecoder>& origNode) {
    static_assert(std::is_same<std::decay<decltype(*origNode)>::type, ov::op::v0::CTCGreedyDecoder>::value,
                  "opset operation mismatch");
    const auto inputs = getInputs(origNode);
    VPUX_THROW_UNLESS(inputs.size() == 2, "nGraph CTCGreedyDecoder node '{0}' has unsupported number of inputs '{1}'",
                      origNode->get_friendly_name(), inputs.size());

    auto op = builder.create<IE::CTCGreedyDecoderOp>(createLocation(origNode), inputs[0], inputs[1],
                                                     origNode->get_ctc_merge_repeated());
    addOutputs(origNode, op);
}
```
Add map entry for operation dispatcher.

[src/vpux_compiler/src/frontend/IE.cpp#L248](../src/frontend/IE.cpp#L248)
```cpp
mlir::func::FuncOp NGraphImporter::buildMainFunc(mlir::OpBuilder& moduleBuilder, StringRef funcName) {
    using Callback = void (NGraphImporter::*)(mlir::OpBuilder & builder, const OrigNodePtr& origNode);
    using DispatchMap = std::map<ov::NodeTypeInfo, Callback>;

    static const DispatchMap dispatchMap{

            MAP_ENTRY(ov::ov::opset7::CTCGreedyDecoder),
    };
}
```

## IE Output shape resolver
Create a new file that defines the `vpux::IE::<OpName>::inferReturnTypeComponents` function.
Given input tensors and layer parameters, this function computes output shapes and types of the operation.
[(new) src/vpux_compiler/src/dialect/IE/IR/ops/ctc_greedy_decoder.cpp](../src/dialect/IE/IR/ops/ctc_greedy_decoder.cpp)
```cpp
mlir::LogicalResult vpux::IE::CTCGreedyDecoderOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::CTCGreedyDecoderOpAdaptor ctc(operands, attrs);
    if (mlir::failed(ctc.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = ctc.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    if (inShape.size() != 3) {
        return errorAt(loc, "First input tensor should have 3 dimensions");
    }

    SmallVector<int64_t> outputShape{inShape[1], inShape[0], 1, 1};

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());

    return mlir::success();
}
```

## Optional: IE Attribute parser
For the operations with sophisticated parameters (eg parameters cannot be expressed with numbers, enum), custom attribute should be implemented. This attribute is not related to the example above.
[src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/attributes.td#L124](../tblgen/vpux/compiler/dialect/IE/attributes.td#L124)
```swift
//
// RoundingType
//

def IE_RoundingType :
        IE_I64EnumAttr<
            "RoundingType",
            "Rounding type that operations support",
            [
                I64EnumAttrCase<"FLOOR">,
                I64EnumAttrCase<"CEIL">,
            ]
        > {
}

def IE_RoundingTypeAttr : IE_EnumAttr<IE_RoundingType, "rounding_type">;

```
Additional helper function should be used for parsing the attribute.
[src/vpux_compiler/src/frontend/IE.cpp#L164](../src/frontend/IE.cpp#L164)
```cpp
private:
    IE::RoundingTypeAttr importRoundingType(ov::op::RoundingType roundingType);
```
[src/vpux_compiler/src/frontend/IE.cpp#L1333](../src/frontend/IE.cpp#L1333)
```cpp
IE::RoundingTypeAttr NGraphImporter::importRoundingType(ov::op::RoundingType roundingType) {
    switch (roundingType) {
    case ov::op::RoundingType::FLOOR:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::FLOOR);
    case ov::op::RoundingType::CEIL:
        return IE::RoundingTypeAttr::get(_ctx, IE::RoundingType::CEIL);
    default:
        VPUX_THROW("Unknown RoundingType");
    }
}
```

## Optional: Canonicalizer

IE Dialect operation can contain canonicalization pattern, which simplifies IR (fusing, using more concrete operations, converting constant operands to attribute).
Such manipulation should be done on IE Dialect level, not ngraph parser, because FrontEnd just performs 1-to-1 mapping without any transformation/adaptation logic, and with such separation we have simple frontend and Canonicalization pass, which we can cover with tests.

Most used case is converting inputs (e.g. parameters from weights) into attributes. In this case we will simplify our graph (less edges between constant and layer) and simplify approach how to work with attributes (because in case of working / manipulating with inputs, we need first check, that it's constant, then transform it, etc.)

Swish operation canonicalizer example
[src/vpux_compiler/src/dialect/IE/IR/ops/swish.cpp#L41](../src/dialect/IE/IR/ops/swish.cpp#L41)
```cpp
//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SwishOp> {
public:
    using mlir::OpRewritePattern<IE::SwishOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SwishOp swishOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SwishOp swishOp, mlir::PatternRewriter& rewriter) const {
    // Check if Input was already converted to Attribute
    auto beta = swishOp.beta();
    if (beta == nullptr) {
        return mlir::failure();  // mlir::failure() means that pass was not applied
    }

    // Check for Input to be a constant
    auto betaOp = swishOp.beta().getDefiningOp<Const::DeclareOp>();
    if (betaOp == nullptr) {
        return mlir::failure();
    }

    // Check for constant to have "one value"
    const auto betaContent = betaOp.content();
    if (!betaContent.isSplat()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::SwishOp>(swishOp, swishOp.getType(), swishOp.input(), nullptr,
                                             rewriter.getF64FloatAttr(betaContent.getSplatValue<float>()));

    return mlir::success();
}

}  // namespace

void vpux::IE::SwishOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
```
Add hasCanonicalizer variable to table gen defenition
[src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/ops.td#1531](../tblgen/vpux/compiler/dialect/IE/ops.td#1531)
```swift
def IE_SwishOp :
    ...
    let hasCanonicalizer = 1;
}
```
Some notes about `mlir::failure();`. It doesn't mean, that pass failed. It just mean, that this pass cannot be applied and should be skipped. In example above, this line mean we already converted input into attr, and don't need to do it again. Since canonicalizer pass can be executed few time, we can end-up in endless loop trying to apply this optimization, if we don't do such check.

## Optional: Transformation pass
Canonicalizer, as described, is simple version of transformation. We can do simple fusing, parameters manipulation, but, in general, we will stay with the same operation as before, but in a canonical state.

If we need to do something more complicated, we should be using a pass instead.

Documentation
* https://mlir.llvm.org/docs/DialectConversion

Let's take a look at the example of supporting 1D Convolution. We have Convolution2D already supported. By converting 1D Convolution to 2D variant, we can support Convolution1D operation without an actual kernel implementation.

[src/vpux_compiler/tblgen/vpux/compiler/dialect/IE/passes.td#L287](../tblgen/vpux/compiler/dialect/IE/passes.td#L287)
```swift
//
// ConvertConv1DToConv2D
//

def ConvertConv1DToConv2D : PassBase<"convert-conv1d-to-conv2d", "vpux::FunctionPass"> {
    let summary = "Convert Convolution1D and GroupConvolution1D to its 2D variance";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        Extends input, filter and output tensors with height = 1.
        [N, C, W] -> [N, C, 1, W]
        strides:    {2} -> strides:    {1, 2}
        pads_begin: {2} -> pads_begin: {0, 2}
        pads_end:   {2} -> pads_end:   {0, 2}
        dilations:  {2} -> dilations:  {1, 2}
    }];

    let constructor = "vpux::IE::createConvertConv1DToConv2DPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}
```
Declare a function, that will instantiate custom pass.

[src/vpux_compiler/include/vpux/compiler/dialect/IE/transforms/passes.hpp](../include/vpux/compiler/dialect/IE/transforms/passes.hpp)
```cpp
// Adjust IE Dialect IR for VPU target.
...
std::unique_ptr<mlir::Pass> createConvertMultiplyToLegacyPowerPass(Logger log = Logger::global());
...
```

Create pass implementation file. Define rewriter pass and derive from `mlir::OpRewritePattern`.
There is also more sophisticated `mlir::OpConversionPattern` you might use. https://mlir.llvm.org/docs/DialectConversion/#conversion-patterns

[src/vpux_compiler/src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp](../src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp)
```cpp
//
// ConvolutionExpansion
//

class ConvolutionExpansion final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionExpansion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvolutionExpansion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};
```
Write main pass logic that `matchesAndRewrites` desired operations

[src/vpux_compiler/src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp](../src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp)
```cpp
mlir::LogicalResult ConvolutionExpansion::matchAndRewrite(IE::ConvolutionOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::Convolution Operation '{0}'", origOp->getLoc());

    // Transform inputs and attributes
    const auto newInput = extendTensor(rewriter, origOp->getLoc(), origOp.input());
    const auto newFilter = extendTensor(rewriter, origOp->getLoc(), origOp.filter());
    const auto newBias = extendTensor(rewriter, origOp->getLoc(), origOp.bias());

    const auto newStrides = append(getContext(), origOp.strides(), 1);
    const auto newPadsBegin = append(getContext(), origOp.pads_begin(), 0);
    const auto newPadsEnd = append(getContext(), origOp.pads_end(), 0);
    const auto newDilations = append(getContext(), origOp.dilations(), 1);

    // Create new operation with transformed parameters
    auto newConvOp = rewriter.create<IE::ConvolutionOp>(origOp->getLoc(), newInput, newFilter, newBias, newStrides,
                                                        newPadsBegin, newPadsEnd, newDilations, origOp.post_opAttr());

    const auto outputShape = origOp.output().getType().cast<mlir::ShapedType>().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    // Replace old IE::ConvolutionOp with a new IE::ConvolutionOp + IE::ReshapeOp
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newConvOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 'IE::Convolution' (2D)");

    return mlir::success();
}
```
After defining a match and rewite pattern, create `safeRunOnFunc()` function.
1. It contains a list of operations that should be `legalized`. In our case its a `Convolution1D` operation.
2. `Convolution1D` has 3D input tensor and is considered illegal. `isLegalConvOp` will return `true` if given Convolution has `input != 3D`, thus should not be converted by our pass.
3. Create `ConversionTarget` and list all the operations involved in a transformation.
4. Add convertion patterns that will try to legalize all `DynamicallyLegalOps`.
5. Use `applyPartialConversion` function to run the pass. More conversion modes could be found in the [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) documentation.

[src/vpux_compiler/src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp](../src/dialect/IE/transforms/passes/convert_conv1d_to_conv2d.cpp)
```cpp
//
// ConvertConv1DToConv2DPass
//

class ConvertConv1DToConv2DPass final : public IE::ConvertConv1DToConv2DBase<ConvertConv1DToConv2DPass> {
public:
    explicit ConvertConv1DToConv2DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertConv1DToConv2DPass::safeRunOnFunc() {
    auto& ctx = getContext();

    // Illegal ops will be converted (legalized)
    const auto isLegalConvOp = [&](IE::ConvolutionOp conv) {
        const auto inputShape = conv.input().getType().cast<mlir::ShapedType>().getShape();
        return inputShape.size() != 3;
    };

    mlir::ConversionTarget target(ctx);
    // DynamicallyLegalOp is illegal op that could be legalized
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalConvOp);
    // Add legal ops that also are used in a transformation
    // Usually it will be IE::ReshapeOp, IE::ConvertOp or similar
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvolutionExpansion>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertConv1DToConv2DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertConv1DToConv2DPass(Logger log) {
    return std::make_unique<ConvertConv1DToConv2DPass>(log);
}
```
Add pass to the pipeline. Most of the transormations should be added to `buildAdjustForVPUPipeline` because they are specific to VPU platform.

[src/vpux_compiler/src/dialect/IE/transforms/pipelines.cpp](../src/dialect/IE/transforms/pipelines.cpp)
```cppvoid vpux::IE::buildAdjustForVPUPipeline(mlir::OpPassManager& pm, Logger log) {
    ...
    pm.addPass(IE::createConvertConv1DToConv2DPass(log));
}

```

# VPU Dialect
The VPU Dialect represents an extension over the IE dialect that brings hardware-specific information to the IR.

Documentation

* [chapters/generated/dialect/_VPU.md](chapters/generated/dialect/_VPU.md)

## VPU Operation table gen
Let's create a table-gen representation of our layer.

* let summary – one line description of op
* let arguments – input parameters for the layer. Possible types can be found here: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td
* let results – outputs of the operation

[src/vpux_compiler/tblgen/vpux/compiler/dialect/VPU/ops.td#L1992](../tblgen/vpux/compiler/dialect/VPU/ops.td#L1992)
```swift
//
// CTCGreedyDecoder
//

def VPU_CTCGreedyDecoderOp :
        VPU_LayerOp<
            "CTCGreedyDecoder"
        > {
    let summary = "CTCGreedyDecoder VPU layer";

    let arguments = (ins
        RankedTensorOf<[F16, F32]>:$input,
        RankedTensorOf<[F16, F32]>:$sequenceLengths,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        RankedTensorOf<[F16, F32]>:$output
    );
}
```

## VPU Output shape resolver
Create a new file that defines the `vpux::VPU::<OpName>::inferReturnTypes` function.
Compared to the IE dialect, this function will return the output types of the operation, not their shape components. This allows the operations in the dialect to work with any type, without begin restricted to `mlir::ShapedType`.

It is recommended to opt for `vpux::NDTypeInterface` while working with tensor types in this dialect, since this interface is compatible with all MLIR and custom types.

[(new) src/vpux_compiler/src/dialect/VPU/ops/ctc_greedy_decoder.cpp](../src/dialect/VPU/ops/ctc_greedy_decoder.cpp)
```cpp
mlir::LogicalResult vpux::VPU::CTCGreedyDecoderOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::CTCGreedyDecoderOpAdaptor ctc(operands, attrs);
    if (mlir::failed(ctc.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = ctc.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();

    if (inShape.size() != 3) {
        return errorAt(loc, "First input tensor should have 3 dimensions");
    }

    SmallVector<int64_t> outputShape{inShape[1], inShape[0], 1, 1};

    const auto outType = inType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
```

## Optional: Attributes, canonicalization, transformations

The same information from the IE dialect with regards to attributes, canonicalization and transformations applies to the VPU dialect, with some mentions:

- canonicalization logic to convert inputs (e.g. parameters from weights) into attributes shouldn't generally be necessary if it is present in IE dialect;
- if operands are converted to attributes in IE dialect, there should be no need to add the operands in VPU dialect as well - the attributes should suffice; the operands might be included in cases where the conversion from operands to attributes in IE is done conditionally.

## IE → VPU lowering
Now that the IE and VPU definitions of the operation are created, the logic that performs the IE->VPU lowering for it should follow.

Generally, it will be enough to add the lowering logic in [convert_layers_to_VPU.td](../tblgen/vpux/compiler/conversion/rewriters/convert_layers_to_VPU.td):

```swift
//
// IE.CTCGreedyDecoder -> VPU.CTCGreedyDecoder
//

def createCTCGreedyDecoderOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPU::CTCGreedyDecoderOp>(
                $_loc, $0, $1, $2)
        }]>;

def RewriteCTCGreedyDecoder :
        Pat<
            (IE_CTCGreedyDecoderOp $input, $sequenceLengths, $mergeRepeated),
            (createCTCGreedyDecoderOp $input, $sequenceLengths, $mergeRepeated)
        >;
```

This tablegen declaration will automatically generate the C++ code that does the lowering. Sometimes, using the tablegen is not possible since the generated code cannot cover the operands/results properly. Example of relevant cases:
- operation has a variadic number of results;
- operation has multiple results and at least one optional/variadic operand.

For such cases, it will be necessary to manually create the lowering logic in [convert_layers_to_VPU.cpp](../src/conversion/passes/IE2VPU/convert_layers_to_VPU.cpp). Although this is not applicable for CTCGreedyDecoder used as example, its manual lowering logic would have looked like:

```cpp
//
// CTCGreedyDecoderRewrite
//

class CTCGreedyDecoderRewrite final : public mlir::OpRewritePattern<IE::CTCGreedyDecoderOp> {
public:
    CTCGreedyDecoderRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::CTCGreedyDecoderOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::CTCGreedyDecoderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CTCGreedyDecoderRewrite::matchAndRewrite(IE::CTCGreedyDecoderOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CTCGreedyDecoder Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::CTCGreedyDecoderOp>(origOp, origOp.input(), origOp.sequenceLengths(),
                                                         origOp.mergeRepeatedAttr());

    return mlir::success();
}
```

The rewritter would also have to be registered:

```cpp
void ConvertLayers2VPUPass::safeRunOnFunc() {
    // ...
    mlir::RewritePatternSet patterns(&ctx);
    // ...
    patterns.add<CTCGreedyDecoderRewrite>(&ctx, _log);
    // ...
```

### IE → VPU lowering lit-test

The lowering logic should also be tested. For this, create a dedicated lit-test in [convert_layers_to_VPU_30XX.mlir](../../../tests/lit/NPU/conversion/passes/IE2VPU/convert_layers_to_VPU_30XX.mlir) containing the IE operation as input and checks for the resulting VPU operation. If needed, other operations such as constants can be included.

```cpp
// CHECK-LABEL: @CTCGreedyDecoder
func.func @CTCGreedyDecoder(%arg0: tensor<20x8x128xf16>, %arg1: tensor<20x8xf16>) -> tensor<8x20x1x1xf16> {
    %0 = IE.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    return %0 : tensor<8x20x1x1xf16>

    // CHECK:       [[VAR0:%.+]] = VPU.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated}
    // CHECK-SAME:    : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    // CHECK:       return [[VAR0]] : tensor<8x20x1x1xf16>
}
```

## You're half way there.
You should be able to compile code now. Run single layer test and look for "Unable to legalize VPU::OperationName" message. That means that MLIR compiler was not able to convert VPU::OperationName to VPUIP::OperationName. This will be the next step.

# GraphFile-schema
GraphFile-schema is a common layer between compiler and runtime. It is a tool for serializing data to the blob.

Before lowering to the VPUIP dialect, make sure that graphFile-schema repository has your operation included. For debugging purposes, you can checkout NPU-plugin schema to the custom branch with the new operation added.

```bash
cd thirdparty/elf
git checkout custom_branch
```
or you can manually add your layer to the existing schema

> graphFile-schema is a submodule that we can't link with a relative path. Please find files below after thirdparty initialization.

In relative file path is `thirdparty/elf/src/schema/software.fbs` from line `446`
```cpp
table CTCDecoderParams {
  ctc_merge_repeated: bool;
}
```
and in the same file from line `885`
```cpp
union SoftwareLayerParams{
// ...
  CTCDecoderParams,
}
```

# VPUIP Dialect

The VPUIP Dialect represents bufferized version of the VPU Dialect, with platform-specific variants of some operations. For example, hardware operations are all represented as `VPUIP::NCEClusterTaskOp`. Software operations for NPU30XX have UPA tasks representing the SHAVE kernels, while the software operations for NPU37XX use `VPUIP::SWKernelOp`. We will be creating the UPA operation for the new layer. It is worth mentioning that in some cases, multiple VPU operations can be lowered to the same UPA task; e.g. `VPU.AddOp` & `VPU.MultiplyOp` both get lowered to `VPUIP.EltwiseUPAOp`.

This dialect no longer works with tensor data. Instead, buffers are utilized by making use of `MemRefType`.

Documentation
* VPUIP dialect: [chapters/generated/dialect/_VPUIP.md](chapters/generated/dialect/_VPUIP.md#L1)
* Passes: [chapters/generated/dialect/VPUIP/_passes.md](chapters/generated/dialect/VPUIP/_passes.md#L1)
* Op interfaces: [chapters/generated/dialect/VPUIP/_ops_interfaces.md](chapters/generated/dialect/VPUIP/_ops_interfaces.md#L1)
* About assembly format: [https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format](https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format)

## VPUIP table gen

[src/vpux_compiler/tblgen/vpux/compiler/dialect/VPUIP/ops.td#L2078](../tblgen/vpux/compiler/dialect/VPUIP/ops.td#L2419)
```swift
//
// CTCGreedyDecoderUPAOp
//

def VPUIP_CTCGreedyDecoderUPAOp :
        VPUIP_UPATaskOp<1, "CTCGreedyDecoderUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "CTCGreedyDecoder UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$sequenceLengths,
        F16MemRef:$output_buff,

        UnitAttr:$mergeRepeated,
    );

    let results = (outs
        F16MemRef:$output
    );

    // Describe inputs, outputs and parameters with C++ types
    let builders = [
        OpBuilder<
            (ins
                "mlir::Value":$input, "mlir::Value":$sequenceLengths,
                "mlir::Value":$output,
                "mlir::UnitAttr":$mergeRepeated
            )
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLengths `:` type($sequenceLengths) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`waits` `(` $waitBarriers^ `:` type($waitBarriers) `)`)?
        (`updates` `(` $updateBarriers^ `:` type($updateBarriers) `)`)?
        `->` type(results)
    }];
}
```
## VPUIP UPATask builder
Serialize layer to UPATask using graphFile-schema interface.

[(new) src/vpux_compiler/src/dialect/VPUIP/ops/upa_ctc_greedy_decoder.cpp](../src/dialect/VPUIP/ops/upa_ctc_greedy_decoder.cpp)
```cpp
void vpux::VPUIP::CTCGreedyDecoderUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                               mlir::Value sequenceLengths, mlir::Value output,
                                               mlir::UnitAttr mergeRepeated) {
    // pass parameters that were described in table gen arguments variable
    build(builder, state, input, sequenceLengths, output, mlir::ValueRange{}, mlir::ValueRange{}, mergeRepeated,
          nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CTCGreedyDecoderUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::CTCDecoderParamsBuilder builder(writer);
    builder.add_ctc_merge_repeated(mergeRepeated());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CTCDecoderParams});
}
```
## Redirect interfaces for IE and VPUIP
Add two lines of code to register interfaces that resolves dependencies between dialects.

[src/vpux_compiler/src/dialect/VPUIP/ops.cpp](../src/dialect/VPUIP/ops.cpp)
```cpp
//
// redirectOpInterfacesForIE
//

template <template <class, class> class OpModelForHW, template <class> class OpModelForSW>
void redirectOpInterfacesForIE(mlir::DialectRegistry& registry) {
    // ...
    IE::CTCGreedyDecoderOp::attachInterface<OpModelForSW<VPUIP::CTCGreedyDecoderUPAOp>>(*ctx);
}
```

[src/vpux_compiler/src/dialect/VPUIP/ops.cpp](../src/dialect/VPUIP/ops.cpp)
```cpp
//
// redirectOpInterfacesForVPUIP
//

template <class OpModelForHW, class OpModelForDMA, class OpModelForSW>
void redirectOpInterfacesForVPUIP(mlir::DialectRegistry& registry) {
    // ...
    VPUIP::CTCGreedyDecoderUPAOp::attachInterface<OpModelForSW>(*ctx);
}
```

## VPUIP verifier
Verifiers are used to validate state of the operation. It is common to check input size, layout and strides for correctness. Add checks for kernel limitations if present.

Add verifier to the VPUIP table gen.

[src/vpux_compiler/tblgen/vpux/compiler/dialect/VPUIP/ops.td](../tblgen/vpux/compiler/dialect/VPUIP/ops.td)
```swift
let hasVerifier = 1;
```
Implement verify function.
[src/vpux_compiler/src/dialect/VPUIP/ops/upa_ctc_greedy_decoder.cpp](../src/dialect/VPUIP/ops/upa_ctc_greedy_decoder.cpp)
```cpp
mlir::LogicalResult vpux::VPUIP::CTCGreedyDecoderUPAOp::verify() {
    const auto inShape = getShape(input());

    if (inShape.size() != 3) {
        return errorAt(*this, "Input shape should have 3 dimensions");
    }

    if (inShape[Dim(1)] != 1) {
        return errorAt(*this, "Input tensor [T N C] = [{0} {1} {2}] has unsupported dimension size N != 1",
                       inShape[Dim(0)], inShape[Dim(1)], inShape[Dim(2)]);
    }

    return mlir::success();
}
```

## VPU → VPUIP lowering
Convert previous representation of a layer in VPU dialect down to the VPUIP dialect.

Create a `createRTLayer` method for your VPU operation:

[src/vpux_compiler/src/conversion/passes/VPU2VPUIP/convert_layers_to_VPUIP.cpp#L389](../src/conversion/passes/VPU2VPUIP/convert_layers_to_VPUIP.cpp#L389)
```cpp
mlir::Operation* createRTLayer(VPU::CTCGreedyDecoderOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CTCGreedyDecoderUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CTCGreedyDecoderUPAOp>(origOp.getLoc(), newOp.input(), newOp.sequenceLengths(),
                                                  newOp.output_buff(), origOp.mergeRepeatedAttr());
}
```

Then register it in the `LayerRewriter` so that it is utilized:

```cpp
mlir::LogicalResult LayerRewrite::matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    using CreateFunc =
            mlir::Operation* (*)(mlir::Operation * origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder & b);
    // ...
    // Add new case for the new operation
    CASE(VPU::CTCGreedyDecoderOp)
```

Some operations can end up lowered to more than a single UPA task at this step. For such cases, a dedicated rewriter should be created instead. An example can be found with `VPU.LSTMCellOp` which has the `LSTMCellRewrite` class in the same pass.

### VPU → VPUIP lowering lit-test
Similar to IE->VPU, the lowering logic will be tested by creating a lit-test in [convert_layers_to_VPUIP_30XX.mlir](../../../tests/lit/NPU/conversion/passes/VPU2VPUIP/convert_layers_to_VPUIP_30XX.mlir). If there are instances where the lowering logic behaves differently for various configurations of the operation, please make sure to cover them all with different tests.

```cpp
// CHECK-LABEL: @CTCGreedyDecoder
func.func @CTCGreedyDecoder(%arg0: memref<20x1x128xf16>, %arg1: memref<20x1xf16>) -> memref<1x20x1x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<20x1x128xf16> to tensor<20x1x128xf16>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<20x1xf16> to tensor<20x1xf16>
    %2 = VPU.CTCGreedyDecoder(%0, %1) {mergeRepeated} : tensor<20x1x128xf16>, tensor<20x1xf16> -> tensor<1x20x1x1xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x20x1x1xf16> to memref<1x20x1x1xf16>
    return %3 : memref<1x20x1x1xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x20x1x1xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.CTCGreedyDecoderUPA {mergeRepeated}
    // CHECK-SAME:      inputs(%arg0 : memref<20x1x128xf16>, %arg1 : memref<20x1xf16>) outputs([[VAR0]] : memref<1x20x1x1xf16>) -> memref<1x20x1x1xf16>
    // CHECK:       return [[VAR1]] : memref<1x20x1x1xf16>
}
```

This test has the function arguments as `memrefs` (buffers), while the VPU operation works with `tensors`. Therefore, a dummy cast layer is present to interpret `memrefs` as `tensors`. When the operation is lowered to VPUIP, the UPA task will also work with buffer data so these casts are no longer necessary - the canonicalizer handles the removal of these operations. That is why the resulting IR of the test has no casts.

# IERT Dialect
InferenceEngine RunTime Dialect The IERT Dialect represents bufferized version of IE Dialect.

Currently, it is not utilized in the compilation pipelines. Instead, it is kept into maintenance mode, so it is necessary to create the IE equivalent of the operation in IERT dialect, logic for bufferization and an associated lit-test.

The dialect has the following properties:

Works with fixed operation set (like IE Dialect).
Represents execution scheduling and memory allocation.
Works with MemRefType.
Includes transformations and optimizations closer to HW level (memory re-usage, parallel resources' usage, etc.).

Documentation:

* IERT dialect: [chapters/generated/dialect/_IERT.md](chapters/generated/dialect/_IERT.md)
* Passes: [chapters/generated/dialect/IERT/_passes.md](chapters/generated/dialect/IERT/_passes.md)
* About assembly format: https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format

## IERT Table gen
[src/vpux_compiler/tblgen/vpux/compiler/dialect/IERT/ops.td](../tblgen/vpux/compiler/dialect/IERT/ops.td)
```swift
//
// CTCGreedyDecoderOp
//

def IERT_CTCGreedyDecoderOp :
        // the `1` indicates number of outputs
        IERT_LayerOp<1, "CTCGreedyDecoder",
            [
                // Use MultiViewOpInterface instead for operations with many outputs
                ViewLikeOpInterface
            ]
        > {
    let summary = "InferenceEngine run-time CTCGreedyDecoder layer";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[F16, F32]>:$sequenceLengths,

        // Output memory buffer is an input argument of the operation
        // It acts as an in/out parameter
        // Please follow a naming convension and add _buff postfix to the output name
        MemRefOf<[F16, F32]>:$output_buff,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLengths `:` type($sequenceLengths) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}
```
## IE → IERT lowering
Convert previous representation of a layer in IE dialect down to the IERT dialect.

[src/vpux_compiler/src/conversion/passes/IE2IERT/bufferize_IE.cpp](../src/conversion/passes/IE2IERT/bufferize_IE.cpp)
```cpp
mlir::Operation* createRTLayer(IE::CTCGreedyDecoderOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    IERT::CTCGreedyDecoderOp::Adaptor newOp(allBufs);
    return b.create<IERT::CTCGreedyDecoderOp>(origOp.getLoc(), newOp.input(), newOp.sequenceLengths(),
                                              newOp.output_buff(), origOp.mergeRepeatedAttr());
}
```
Verifiers are used to validate state of the operation. It is common to check input size, layout and strides for correctness. Add checks for kernel limitations if present.

[src/vpux_compiler/src/conversion/passes/IE2IERT/bufferize_IE.cpp](../src/conversion/passes/IE2IERT/bufferize_IE.cpp)
```cpp
mlir::LogicalResult LayerRewrite::matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
 const CreateFunc createFunc =
            llvm::TypeSwitch<mlir::Operation*, CreateFunc>(origOp) CASE(mlir::quant::QuantizeCastOp)
    // Add new case for the new operation
    CASE(IE::CTCGreedyDecoderOp)

}
```
### IE → IERT lowering lit-test

The bufferization logic will be tested by creating a lit-test in [bufferize_IE_30XX_37XX_40XX.mlir](../../../tests/lit/NPU/conversion/passes/IE2IERT/bufferize_IE_30XX_37XX_40XX.mlir):

```cpp
// CHECK-LABEL: @CTCGreedyDecoder
func.func @CTCGreedyDecoder(%arg0: tensor<20x8x128xf16>, %arg1: tensor<20x8xf16>) -> tensor<8x20x1x1xf16> {
    %0 = IE.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    return %0 : tensor<8x20x1x1xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<20x8x128xf16> to memref<20x8x128xf16>
    // CHECK:       [[VAR1:%.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<20x8xf16> to memref<20x8xf16>
    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<8x20x1x1xf16>
    // CHECK:       [[VAR3:%.*]] = IERT.CTCGreedyDecoder {mergeRepeated}
    // CHECK-SAME:      inputs([[VAR0]] : memref<20x8x128xf16>, [[VAR1]] : memref<20x8xf16>) outputs([[VAR2]] : memref<8x20x1x1xf16>) -> memref<8x20x1x1xf16>
    // CHECK:       [[VAR4:%.*]] = builtin.unrealized_conversion_cast [[VAR3]] : memref<8x20x1x1xf16> to tensor<8x20x1x1xf16>
    //CHECK:        return [[VAR4]] : tensor<8x20x1x1xf16>
}
```
