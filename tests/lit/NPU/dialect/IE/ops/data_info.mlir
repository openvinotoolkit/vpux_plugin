//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @ParserPrintDirectMLDataInfoTest {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x64x64x64xf16, {order = #NHWC}> directml <dimension_count = 4, data_type = #IE.directml_tensor_data_type<FLOAT16>, byte_offset = 0, size = [1, 64, 64, 64], memory_type = #IE.directml_tensor_memory_type<GLOBAL>, layout_type = #IE.directml_tensor_layout_type<STRIDED>, dimension_block_sizes = [1, 64, 64, 64], start_gutter_sizes = [0, 0, 0, 0], end_gutter_sizes = [0, 0, 0, 0], element_stride = [0, 1, 4096, 64], block_strides = [0, 0, 0, 0], first_memory_tile = 0, memory_tile_count = 1, alignment_of_modifiable_padding_in_bytes = 64, base_alignment_in_bytes = 0, physical_size_in_elements = 150528, order = #NHWC>
    DataInfo "input1" : tensor<1x64x64x64xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output0" : tensor<1x16x128x128xf16, {order = #NHWC}> directml <dimension_count = 4, data_type = #IE.directml_tensor_data_type<FLOAT16>, byte_offset = 0, size = [1, 16, 128, 128], memory_type = #IE.directml_tensor_memory_type<GLOBAL>, layout_type = #IE.directml_tensor_layout_type<STRIDED>, dimension_block_sizes = [1, 16, 128, 128], start_gutter_sizes = [0, 0, 0, 0], end_gutter_sizes = [0, 0, 0, 0], element_stride = [0, 1, 2048, 16], block_strides = [0, 0, 0, 0], first_memory_tile = 0, memory_tile_count = 1, alignment_of_modifiable_padding_in_bytes = 64, base_alignment_in_bytes = 0, physical_size_in_elements = 150528, order = #NHWC>
  }
  func.func @main() {
    return
  }
}

// CHECK: DataInfo "input0"{{.*}}directml <
// CHECK: DataInfo "input1"
// CHECK-NOT: directml
// CHECK: DataInfo "output0"{{.*}}directml <
