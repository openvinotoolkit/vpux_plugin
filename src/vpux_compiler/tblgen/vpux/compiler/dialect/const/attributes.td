//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONST_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_CONST_ATTRIBUTES

include "vpux/compiler/dialect/const/dialect.td"
include "vpux/compiler/dialect/const/attr_interfaces.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

//
// Base classes
//

class Const_Attr<string name, list<Trait> traits = [], string baseCppClass = "mlir::Attribute"> :
        AttrDef<Const_Dialect, name, traits, baseCppClass> {
    let mnemonic = name;
}

//
// TransformAttrInterfaceArrayAttr
//

def Const_TransformAttrInterfaceArrayAttr :
    ArrayOfAttr<Const_Dialect, "TransformAttrInterfaceArray", "TransformAttrInterfaceArray", "vpux::Const::TransformAttrInterface">;

//
// SymElementsAttr
//

def Const_SymElementsAttr : Const_Attr<"SymElements", [
    DeclareAttrInterfaceMethods<ElementsAttrInterface>
]> {
    let summary = "Can reference Rodata containing the elements";
    let description = [{
        This attribute implements mlir::ElementsAttr "interface". It can be used
        instead of mlir::DenseElementsAttr or mlir::DenseResourceElementsAttr to
        reference a const.Rodata op.

        Because of some quirks of MLIR, proper verification is done by Const.DeclareOp.
        Using this attribute on its own should be done with caution.
    }];

    let parameters = (ins
        "::mlir::SymbolRefAttr":$sym_name,
        // We need an additional user-informed type attribute because we cannot lookup
        // the underlying elements to which the symbol points to in all situations.
        // Verifying if this type matches the type of the underlying elements is
        // done by Const.DeclareOp using the 'SymbolOpUserInterface' and the function
        // verifySymbolUses().
        "::mlir::ShapedType":$type
    );

    let hasCustomAssemblyFormat = 1;
}

//
// ContentAttr
//

def Const_ContentAttr : Const_Attr<"Content", [Const_RequestQueueAttrInterface]> {
    let summary = "Lazy folded constant content";

    let description = [{
        This attribute holds base constant and transformation applied to it.
        It provides and API to get transformed values on the fly.
    }];

    let parameters = (ins
        "mlir::ElementsAttr":$baseContent,
        "vpux::Const::TransformAttrInterfaceArrayAttr":$transformations,
        "vpux::NDTypeInterface":$finalType,
        "mlir::UnitAttr":$isSplat
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ElementsAttr":$baseContent), [{
                const auto [finalType, isSplat] = vpux::Const::ContentAttr::inferFinalTypeAndSplat(baseContent, {});
                return $_get(baseContent.getContext(), baseContent,
                    vpux::Const::TransformAttrInterfaceArrayAttr(), finalType, isSplat);
            }]
        >,

        AttrBuilderWithInferredContext<
            (ins "mlir::ElementsAttr":$baseContent, "mlir::ArrayRef<vpux::Const::TransformAttrInterface>":$transformations), [{
                auto transformationsAttr = vpux::Const::TransformAttrInterfaceArrayAttr::get(baseContent.getContext(), transformations);
                const auto [finalType, isSplat] = vpux::Const::ContentAttr::inferFinalTypeAndSplat(
                    baseContent, transformations);
                return $_get(baseContent.getContext(), baseContent, transformationsAttr, finalType, isSplat);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let genAccessors = 0;
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        using ValueType = vpux::Const::Content;

        ValueType fold(bool bypassCache = false) const;

        mlir::ElementsAttr getBaseContent() const;
        mlir::ArrayRef<vpux::Const::TransformAttrInterface> getTransformations() const;
        vpux::NDTypeInterface getType() const;
        bool isSplat() const;

        vpux::Const::ContentAttr broadcast(Dim axis, int64_t value) const;
        vpux::Const::ContentAttr convertElemType(mlir::Type newElemType) const;
        vpux::Const::ContentAttr quantCast(mlir::quant::QuantizedType newElemType = nullptr) const;
        vpux::Const::ContentAttr dequantize() const;
        vpux::Const::ContentAttr quantize(mlir::quant::QuantizedType targetType) const;
        vpux::Const::ContentAttr rescale(double scale) const;
        vpux::Const::ContentAttr relocateWeightsTablePointers(ArrayRef<uint32_t> weightsPtr, uint64_t sparsityPtr, vpux::ShapeRef offsets,
                                                              uint64_t weightsTableSize, uint64_t weightsElemBitSize, VPUIP::SparsityCompressionAttr weightsCompression,
                                                              uint64_t channelOffset) const;
        vpux::Const::ContentAttr swizzleConstant(uint64_t swizzleKey, uint64_t arch) const;
        vpux::Const::ContentAttr add(double bias) const;
        vpux::Const::ContentAttr reshape(vpux::ShapeRef newShape) const;
        vpux::Const::ContentAttr reverse(Dim axis) const;
        vpux::Const::ContentAttr reorder(vpux::DimsOrder newOrder) const;
        vpux::Const::ContentAttr padWithZero(vpux::ShapeRef padBefore, vpux::ShapeRef padAfter) const;
        vpux::Const::ContentAttr subview(vpux::ShapeRef offset, vpux::ShapeRef shape) const;
        vpux::Const::ContentAttr bitPack(int64_t width) const;
        vpux::Const::ContentAttr transpose(vpux::DimsOrder newOrder) const;
        vpux::Const::ContentAttr memPermute(vpux::DimsOrder dstOrder, vpux::DimsOrder memPerm) const;
        vpux::Const::ContentAttr layoutCast(vpux::DimsOrder dstOrder) const;
        vpux::Const::ContentAttr expandDilated(vpux::ShapeRef dilations) const;
        vpux::Const::ContentAttr getSparsityMap() const;
        vpux::Const::ContentAttr sparsify(bool compressOutputType, mlir::ElementsAttr numActualElements = nullptr) const;
        vpux::Const::ContentAttr changeShapeAndElemType(vpux::ShapeRef newShape, mlir::Type newElemType) const;
        vpux::Const::ContentAttr fuse(mlir::RankedTensorType fusedTensorType, Const::ContentAttr weightsTable, Const::ContentAttr weights, Const::ContentAttr sparsity, Const::ContentAttr activations) const;
        vpux::Const::ContentAttr scalarMultInverse() const;

        static vpux::Const::ContentAttr addTransformation(vpux::Const::ContentAttr input, vpux::Const::TransformAttrInterface transformation);
        static std::pair<vpux::NDTypeInterface, mlir::UnitAttr> inferFinalTypeAndSplat(
            mlir::ElementsAttr content, mlir::ArrayRef<vpux::Const::TransformAttrInterface> transformations);

        vpux::Const::ContentAttr stripTransformationsFrom(Const::TransformAttrInterface transformation);
        SmallVector<vpux::Const::TransformAttrInterface> getLastTransformationsFrom(Const::TransformAttrInterface transformation);
    }];

    let storageType = "vpux::Const::ContentAttr";
    let returnType = "vpux::Const::Content";
    let convertFromStorage = "$_self.fold()";
}

//
// Const_EquivalenceRequestAttr
//

def Const_EquivalenceRequestAttr : Const_Attr<"EquivalenceRequest", [Const_RequestQueueAttrInterface]> {
    let summary = "Marks two ContentAttr objects as being equivalent for background folding";

    let description = [{
        This attribute is meant to be used in the request queue for the background folding feature.

        It is meant to represent that two ContentAttr objects are equivalent in terms of folding,
        meaning that the `fold()` method will produce the same result for both attributes.
        The main use-case for this attribute is to allow the list of transformations of a ContentAttr
        to be optimized (e.g. produce the same result but with fewer transformations or less compute
        per transformation), without losing the computation that was already performed.
    }];

    let mnemonic = "EquivalenceRequest";

    let parameters = (ins
        "vpux::Const::ContentAttr":$originalAttr,
        "vpux::Const::ContentAttr":$newAttr
    );

    let storageNamespace = "details";
    let assemblyFormat = "`<` $originalAttr `,` $newAttr `>`";
}

//
// Const_TerminateRequestAttr
//

def Const_TerminateRequestAttr : Const_Attr<"TerminateRequest", [Const_RequestQueueAttrInterface]> {
    let summary = "Attribute representing the termination of a background folding thread";

    let description = [{
        This attribute is meant to be used in the request queue for the background folding feature.
        When a thread receives this attribute via the folding request queue, it will terminate its execution.
    }];
}

//
// ConvertElemTypeAttr
//

def Const_ConvertElemTypeAttr : Const_Attr<"ConvertElemType",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Convert constant content element type";

    let parameters = (ins
        "mlir::Type":$elemType
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::Type":$elemType), [{
                return $_get(elemType.getContext(), elemType);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// QuantCastAttr
//

def Const_QuantCastAttr : Const_Attr<"QuantCast",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Cast element type from raw integer to quantized type";

    let parameters = (ins
        "mlir::quant::QuantizedType":$elemType
    );

    let builders = [

        AttrBuilder<
            (ins "mlir::quant::QuantizedType":$elemType), [{
                return $_get($_ctxt, elemType);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

//
// DequantizeAttr
//

def Const_DequantizeAttr : Const_Attr<"Dequantize",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Dequantize constant content";
}

//
// QuantizeAttr
//

def Const_QuantizeAttr : Const_Attr<"Quantize",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Converts const values to quantized type";

    let parameters = (ins
        "mlir::quant::QuantizedType":$targetType
    );

    let builders = [

        AttrBuilder<
            (ins "mlir::quant::QuantizedType":$targetType), [{
                return $_get($_ctxt, targetType);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

//
// RescaleAttr
//

def Const_RescaleAttr : Const_Attr<"Rescale",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Rescale constant content";

    let parameters = (ins
        "mlir::FloatAttr":$scale
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::FloatAttr":$scale), [{
                return $_get(scale.getContext(), scale);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// AddAttr
//

def Const_AddAttr : Const_Attr<"Add",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Add constant content";

    let parameters = (ins
        "mlir::FloatAttr":$bias
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::FloatAttr":$bias), [{
                return $_get(bias.getContext(), bias);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// ReshapeAttr
//

def Const_ReshapeAttr : Const_Attr<"Reshape",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Reshape constant content";

    let parameters = (ins
        "mlir::ArrayAttr":$shape
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$shape), [{
                return $_get(shape.getContext(), shape);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// ReorderAttr
//

def Const_ReorderAttr : Const_Attr<"Reorder",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Reorder constant content";

    let parameters = (ins
        "mlir::AffineMapAttr":$order
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$order), [{
                return $_get(order.getContext(), order);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// ReverseAttr
//

def Const_ReverseAttr : Const_Attr<"Reverse",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Reverse constant content on specified axis";

    let description = [{
        This transformation reverses content on a specified axis.
        For example, considering an input with dimensions (d0, d1, d2, d3),
        reversing on axis d1 will result in the data of every subpart (d2, d3)
        being reversed.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$axis
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$axis), [{
                return $_get(axis.getContext(), axis);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}


//
// PadWithZeroAttr
//

def Const_PadWithZeroAttr : Const_Attr<"PadWithZero",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Pad constant content with zeros";

    let parameters = (ins
        "mlir::ArrayAttr":$padBefore,
        "mlir::ArrayAttr":$padAfter
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$padBefore, "mlir::ArrayAttr":$padAfter), [{
                return $_get(padBefore.getContext(), padBefore, padAfter);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// SubViewAttr
//

def Const_SubViewAttr : Const_Attr<"SubView",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["supportsSubByteStorageType"]>]> {
    let summary = "Extract subview from constant content";

    let parameters = (ins
        "mlir::ArrayAttr":$offset,
        "mlir::ArrayAttr":$shape
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$offset, "mlir::ArrayAttr":$shape), [{
                return $_get(offset.getContext(), offset, shape);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// BroadcastAttr
//

def Const_BroadcastAttr : Const_Attr<"Broadcast",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Broadcast axis by value of constant content";

    let parameters = (ins
        "mlir::IntegerAttr":$axis,
        "mlir::IntegerAttr":$value
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$axis, "mlir::IntegerAttr":$value), [{
                return $_get(value.getContext(), axis, value);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// BitPackAttr
//

def Const_BitPackAttr : Const_Attr<"BitPack",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement"]>]> {
    let summary = "Pack sub-byte constant content";

    let parameters = (ins
        "mlir::IntegerAttr":$width
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$width), [{
                return $_get(width.getContext(), width);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// TransposeAttr
//

def Const_TransposeAttr : Const_Attr<"Transpose",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Transpose constant content";

    let parameters = (ins
        "mlir::AffineMapAttr":$order
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$order), [{
                return $_get(order.getContext(), order);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// MemPermuteAttr
//

def Const_MemPermuteAttr : Const_Attr<"MemPermute",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Permute constant content";

    let parameters = (ins
        "mlir::AffineMapAttr":$dstOrder,
        "mlir::AffineMapAttr":$memPerm
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$dstOrder, "mlir::AffineMapAttr":$memPerm), [{
                return $_get(dstOrder.getContext(), dstOrder, memPerm);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// LayoutCastAttr
//

def Const_LayoutCastAttr : Const_Attr<"LayoutCast",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Change constant content layout";

    let parameters = (ins
        "mlir::AffineMapAttr":$dstOrder
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$dstOrder), [{
                return $_get(dstOrder.getContext(), dstOrder);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// WeightsTablePtrs
//

def Const_RelocateWeightsTableAttr : Const_Attr<"RelocateWeightsTable",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Patches offsets in the weights table";

    let description = [{
        The weights table is used by the DPU to describe the constants associated with the operation.
        It has OCx4 elements, where each element is a 32-bit integer:
          0  [weights_ptr0, sparsity_ptr0, mult_shift0, bias0]
          1  [weights_ptr1, sparsity_ptr1, mult_shift1, bias1]
          ...
         OC  [weights_ptrC, sparsity_ptrC, mult_shiftC, biasC]
        Each row corresponds to an output channel of the DPU operation, where:
          - `weights_ptr` corresponds to the offset in the cluster where the channel's weights are found
          - `sparsity_ptr` corresponds to the offset in the cluster where the channel's weights sparsity map is found
          - `mult_shift` corresponds to the multiply and shift values applied to the output
          - `bias` corresponds to the bias added to the output

        This transformation relocates the pointers from the weights table based on the parameters below:
        - weightsPtr - the base address of the weights associated with the weights table,
                       which will be added to all weights_ptr entries from the table
        - sparsityPtr - the base address of the weights sparsity map associated with the weights table,
                        which will be added to all sparsity_ptr entries from the table
        - offsets - an array of channel offsets, which describes what clusters each channel belongs to
                  - for example, if the associated weights are segmented equally over OC across two clusters,
                    the `offsets` parameter could contain [0, OC/2]; this will cause the second half of the
                    channels to have their pointers reset since this subset of values will be used by the second cluster only
        - weightsTableSize - the size of the weights table, in bytes
        - weightsElemBitSize - the elements size of the weights, in bits
        - weightsCompression - in case the weights are sparse, this parameter indicates how many elements are found
                               in the weights per output channel; it is used internally to correctly determine the new pointers
        - channelOffset - the starting offset of the channels that are being relocated; this is used in case the transformation
                          works on a slice of the channels to ensure the correct pointers are computed
                        - for example, if the transformation receives channels [16-31] for the table, the `channelOffset` would
                          be set to 16, so that the first relocated pointer corresponds to channel 16 instead of channel 0
    }];

    let parameters = (ins
        "mlir::ArrayAttr":$weightsPtr,
        "mlir::IntegerAttr":$sparsityPtr,
        "mlir::ArrayAttr":$offsets,
        "mlir::IntegerAttr":$weightsTableSize,
        "mlir::IntegerAttr":$weightsElemBitSize,
        "VPUIP::SparsityCompressionAttr":$weightsCompression,
        "mlir::IntegerAttr":$channelOffset
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$weightsPtr, "mlir::IntegerAttr":$sparsityPtr, "mlir::ArrayAttr":$offsets, "mlir::IntegerAttr":$weightsTableSize,
                 "mlir::IntegerAttr":$weightsElemBitSize, "VPUIP::SparsityCompressionAttr":$weightsCompression, "mlir::IntegerAttr":$channelOffset), [{
                return $_get(weightsPtr.getContext(), weightsPtr, sparsityPtr, offsets, weightsTableSize, weightsElemBitSize, weightsCompression, channelOffset);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// SwizzleConstant
//

def Const_SwizzleConstantAttr : Const_Attr<"SwizzleConstant",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement",
                                                                    "supportsSubByteStorageType"]>]> {
    let summary = "Swizzles the constant based on swizzle key";

    let description = [{
        This transformation applies swizzling transformation to a constant content.
        Configuration of transformation depends on architecture and swizzleKey.
        Constant content will be size aligned (alignSize flag) if its default size
        does not meet HW requirements for this transformation.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$swizzleKey,
        "mlir::IntegerAttr":$arch
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$swizzleKey, "mlir::IntegerAttr": $arch), [{
                return $_get(swizzleKey.getContext(), swizzleKey, arch);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

def Const_FuseAttr : Const_Attr<"Fuse",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Fuse constants";

    let parameters = (ins
        "mlir::RankedTensorType":$fusedType,
        "Const::ContentAttr":$weightsTable,
        OptionalParameter<"Const::ContentAttr">:$weights,
        OptionalParameter<"Const::ContentAttr">:$sparsity,
        OptionalParameter<"Const::ContentAttr">:$activations
    );

    let storageNamespace = "details";
    let assemblyFormat = [{
        `<`
        $fusedType
        `,` `weightsTable` `=` `<` $weightsTable `>`
        (`,` `weights` `=` `<` $weights^ `>`)?
        (`,` `sparsity` `=` `<` $sparsity^ `>`)?
        (`,` `activations` `=` `<` $activations^ `>`)?
        `>`
    }];
}

//
// ExpandDilatedAttr
//

def Const_ExpandDilatedAttr : Const_Attr<"ExpandDilated",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Expand constant content with zeros according to dilations";

    let parameters = (ins
        "mlir::ArrayAttr":$dilations
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$dilations), [{
                return $_get(dilations.getContext(), dilations);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// GetSparsityMapAttr
//

def Const_GetSparsityMapAttr : Const_Attr<"GetSparsityMap",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement"]>
    ]> {
    let summary = "Generate sparsity map";

    let storageNamespace = "details";
}

//
// Sparsify
//

def Const_SparsifyAttr : Const_Attr<"Sparsify", [
        DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement"]>
    ]> {
    let summary = "Remove zeros / zero-points from constant";

    let parameters = (ins
        "mlir::BoolAttr":$compressOutputType,
        "mlir::ElementsAttr":$numActualElements
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::BoolAttr":$compressOutputType), [{
                return $_get(compressOutputType.getContext(), compressOutputType, nullptr);
            }]
        >,

        AttrBuilderWithInferredContext<
            (ins "mlir::BoolAttr":$compressOutputType, "mlir::ElementsAttr":$numActualElements), [{
                return $_get(compressOutputType.getContext(), compressOutputType, numActualElements);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// ChangeShapeAndElemTypeAttr
//

def Const_ChangeShapeAndElemTypeAttr : Const_Attr<"ChangeShapeAndElemType",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Change the shape and element type of a constant";

    let description = [{
        This transformation atomically changes the shape and the element type of a constant.
        It can only be applied to constants that have a per-axis quantized element type
        and the only difference between the original element type and the new one must be
        the quantization dimension. The transformation is needed to avoid any intermediary
        invalid state when the reshape also changes the quantization dimention.
    }];

    let parameters = (ins
        "mlir::ArrayAttr":$shape,
        "mlir::Type":$elemType
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$shape, "mlir::Type":$elemType), [{
                return $_get(shape.getContext(), shape, elemType);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

def Const_ScalarMultInverseAttr : Const_Attr<"ScalarMultInverse",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Apply scalar multiplicative inverse to the constant content";
}

#endif
