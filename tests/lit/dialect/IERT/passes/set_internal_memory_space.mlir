// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --set-internal-memory-space="memory-space=DDR" %s | FileCheck %s

//
// The 'set-internal-memory-space' pass:
//
//   * Updates only Function bodies.
//   * Updates `memref.alloc` Operation result Type.
//

func @MultipleAllocs(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16>
    %1 = IERT.SoftMax {axisInd = 1} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>) -> memref<1x1000xf16>
    %2 = memref.alloc() : memref<1x1000xf16>
    %3 = IERT.SoftMax {axisInd = 1} inputs(%1 : memref<1x1000xf16>) outputs(%2 : memref<1x1000xf16>) -> memref<1x1000xf16>
    %4 = IERT.SoftMax {axisInd = 1} inputs(%3 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    return %4 : memref<1x1000xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x1000xf16, "DDR">
    // CHECK: [[VAR1:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg0 : memref<1x1000xf16>) outputs([[VAR0]] : memref<1x1000xf16, "DDR">) -> memref<1x1000xf16, "DDR">
    // CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x1000xf16, "DDR">
    // CHECK: [[VAR3:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR1]] : memref<1x1000xf16, "DDR">) outputs([[VAR2]] : memref<1x1000xf16, "DDR">) -> memref<1x1000xf16, "DDR">
    // CHECK: [[VAR4:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR3]] : memref<1x1000xf16, "DDR">) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    // CHECK: return [[VAR4]] : memref<1x1000xf16>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>

func @ReshapeInGraph(%arg0: memref<1x512x1x1xf32>, %arg1: memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32> {
    %0 = IERT.GenericReshape inputs(%arg0 : memref<1x512x1x1xf32>) -> memref<1x512xf32>
    %1 = memref.alloc() : memref<1x512xf32>
    %2 = IERT.SoftMax {axisInd = 1} inputs(%0 : memref<1x512xf32>) outputs(%1 : memref<1x512xf32>) -> memref<1x512xf32>
    %3 = IERT.GenericReshape inputs(%2 : memref<1x512xf32>) -> memref<1x512x1x1xf32>
    %4 = IERT.Copy inputs(%3 : memref<1x512x1x1xf32>) outputs(%arg1 : memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32>
    memref.dealloc %1 : memref<1x512xf32>
    return %4 : memref<1x512x1x1xf32>

    // CHECK: [[VAR0:%.*]] =  IERT.GenericReshape inputs(%arg0 : memref<1x512x1x1xf32>) -> memref<1x512xf32>
    // CHECK: [[VAR1:%.*]] =  memref.alloc() : memref<1x512xf32, "DDR">
    // CHECK: [[VAR2:%.*]] =  IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR0]] : memref<1x512xf32>) outputs([[VAR1]] : memref<1x512xf32, "DDR">) -> memref<1x512xf32, "DDR">
    // CHECK: [[VAR3:%.*]] =  IERT.GenericReshape inputs([[VAR2]] : memref<1x512xf32, "DDR">) -> memref<1x512x1x1xf32, "DDR">
    // CHECK: [[VAR4:%.*]] =  IERT.Copy inputs([[VAR3]] : memref<1x512x1x1xf32, "DDR">) outputs(%arg1 : memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32>
    // CHECK: memref.dealloc [[VAR1]] : memref<1x512xf32, "DDR">
    // CHECK: return [[VAR4]] : memref<1x512x1x1xf32>
}
