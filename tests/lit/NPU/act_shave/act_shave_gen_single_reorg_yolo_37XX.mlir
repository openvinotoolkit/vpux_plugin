// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-NPU37XX
//
// This file generates a blob with reorgYolo activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "Parameter_198" : tensor<1x64x26x26xf16>
    }
    outputsInfo : {
        IE.DataInfo "ReorgYolo_199" : tensor<1x256x13x13xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_ReorgYolo(memref<*xf16>, memref<*xf16>, i64)
        attributes {
            VPU.kernel_code = "reorg_yolo.cpp",
            VPU.kernel_entry = "reorg_yolo"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}



func.func @main(%arg0: memref<1x64x26x26xf16, @DDR>, %arg1: memref<1x256x13x13xf16, @DDR>) -> memref<1x256x13x13xf16, @DDR> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x26x26xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x256x13x13xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x64x26x26xf16, @DDR>) outputs(%in_tile0_cmx : memref<1x64x26x26xf16, [@CMX_NN, 0]>) -> memref<1x64x26x26xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_ReorgYolo            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg2: memref<1x64x26x26xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg3: memref<1x256x13x13xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<1x256x13x13xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run{attrs = [2]}(%arg2, %arg3)
                    : memref<1x64x26x26xf16, [@CMX_NN, 0]>
                    , memref<1x256x13x13xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<1x256x13x13xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x256x13x13xf16, @DDR>) -> memref<1x256x13x13xf16, @DDR>
    }
    return %arg1: memref<1x256x13x13xf16, @DDR>

}


}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "Parameter_198",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         64,
// CHECK:         26,
// CHECK:         26
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           692224,
// CHECK:           10816,
// CHECK:           416,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "ReorgYolo_199",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         256,
// CHECK:         13,
// CHECK:         13
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           692224,
// CHECK:           2704,
// CHECK:           208,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 5,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Parameter_198",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         64,
// CHECK:         26,
// CHECK:         26
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           692224,
// CHECK:           10816,
// CHECK:           416,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "ReorgYolo_199",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         256,
// CHECK:         13,
// CHECK:         13
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           692224,
// CHECK:           2704,
// CHECK:           208,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ]
