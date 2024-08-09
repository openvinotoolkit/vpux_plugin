//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IERT
#define VPUX_COMPILER_DIALECT_IERT

include "mlir/IR/OpBase.td"

def IERT_Dialect : Dialect {
    let summary = "InferenceEngine RunTime Dialect";

    let description = [{
The **IERT Dialect** represents bufferized version of **IE Dialect**.

It has the following properties:

* Works with fixed operation set (like **IE Dialect**).
* Represents execution scheduling and memory allocation.
* Works with `MemRefType`.
* Includes transformations and optimizations closer to HW level (memory re-usage, parallel resources usage, etc.).

**TBD:** It operates with `MemRefType`, but in contrast to MLIR uses SSA value semantic (inspired by PlaidML approach).
It combines both memory effects and buffer aliasing for this:

* Each layer operation takes as its operands both input and output buffers.
* The layer marks input buffer as read-only and output buffer as write-only via memory effects inferface.
* The layer returns new buffer Value, which is an alias for output buffer.

```MLIR
#NHWC = affine_map<(n, c, h, w) -> (n, h, w, c)>

func.func @main(%input: memref<1x3x240x240xf16, #NHWC>, %output: memref<1x3x240x240xf16, #NHWC>) -> memref<1x3x240x240xf16, #NHWC> {
    %1 = IERT.SoftMax(%input, %output) {axisInd = 1} // %1 is an alias for %output
    return %1
}
```

The memory allocation/deallocation is defined as separate operations (dynamic or static).

The **IERT Dialect** uses the following scheme to represent scheduling information:

* Operations order defines scheduling.
* Each IERT operation is assumed as blocking: next operation will not start until previous is finished.
* Concurrent execution is defined as asynchronous regions (**Async Dialect**).

```MLIR
%11_t, %11_f = async.execute { IERT.executor = @NCE }
    [%7_t, %8_9_t](%8_9_f#0 as %8, %8_9_f#1 as %9)
{
    %11_0_t, %11_0_f = async.execute { IERT.executor = @DPU }
    {
        %11_0 = IERT.Convolution(%7, %8, %9) to %10_0 { strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1] }
        async.yield %11_0
    }

    %11_1_t, %11_1_f = async.execute { IERT.executor = @DPU }
    {
        %11_1 = IERT.Convolution(%7, %8, %9) to %10_1 { strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1] }
        async.yield %11_1
    }

    %11:2 = async.await %11_0_f, %11_1_f
    %11 = IERT.FakeConcat(%11#0, %11#1) to %10
    async.yield %11
}
```

[./IERT/_ops_interfaces.md]
    }];

    let name = "IERT";

    let cppNamespace = "vpux::IERT";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::IE::IEDialect",
        "mlir::func::FuncDialect",
        "mlir::memref::MemRefDialect",
        "mlir::quant::QuantizationDialect",
        "mlir::async::AsyncDialect"
    ];

    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 1;

    let extraClassDeclaration = [{
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];

    let usePropertiesForAttributes = 1;
}

#endif
