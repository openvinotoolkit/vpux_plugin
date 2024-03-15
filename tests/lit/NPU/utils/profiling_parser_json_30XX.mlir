//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
//
// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %data_path_npu%/profiling-30XX.mlir.txt
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-30XX.bin -f json | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK: {"traceEvents":[
// CHECK: {"name": "process_name", "ph": "M", "pid":0, "args": {"name" : "DMA"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":0, "args": {"sort_index" : "0"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":0, "tid":0, "args": {"name" : "DMA"}},
// CHECK: {"name": "process_name", "ph": "M", "pid":1, "args": {"name" : "Cluster (0)"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":1, "args": {"sort_index" : "1"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":1, "tid":0, "args": {"name" : "DPU"}},
// CHECK: {"name": "process_name", "ph": "M", "pid":2, "args": {"name" : "Cluster (1)"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":2, "args": {"sort_index" : "2"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":2, "tid":0, "args": {"name" : "DPU"}},
// CHECK: {"name": "process_name", "ph": "M", "pid":3, "args": {"name" : "Cluster (2)"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":3, "args": {"sort_index" : "3"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":3, "tid":0, "args": {"name" : "DPU"}},
// CHECK: {"name": "process_name", "ph": "M", "pid":4, "args": {"name" : "Cluster (3)"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":4, "args": {"sort_index" : "4"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":4, "tid":0, "args": {"name" : "DPU"}},
// CHECK: {"name": "process_name", "ph": "M", "pid":5, "args": {"name" : "UPA"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":5, "args": {"sort_index" : "5"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":5, "tid":0, "args": {"name" : "SW"}},
// CHECK: {"name": "process_name", "ph": "M", "pid":6, "args": {"name" : "Layers"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":6, "args": {"sort_index" : "6"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":6, "tid":0, "args": {"name" : "Layers"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":6, "tid":1, "args": {"name" : "Layers"}},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/_expand_copy_1_13", "cat":"DMA", "ph":"X", "ts":0.000, "dur":17.495, "pid":0, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DMA", "ph":"X", "ts":65.593, "dur":4.032, "pid":0, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]", "cat":"DMA", "ph":"X", "ts":69.845, "dur":1.355, "pid":0, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0", "cat":"DMA", "ph":"X", "ts":228.856, "dur":2.584, "pid":0, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1", "cat":"DMA", "ph":"X", "ts":231.592, "dur":2.852, "pid":0, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_2", "cat":"DMA", "ph":"X", "ts":234.590, "dur":2.724, "pid":0, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_3", "cat":"DMA", "ph":"X", "ts":237.465, "dur":2.452, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]", "cat":"DMA", "ph":"X", "ts":240.062, "dur":0.561, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/_cluster_0", "cat":"DMA", "ph":"X", "ts":299.670, "dur":2.024, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/_cluster_1", "cat":"DMA", "ph":"X", "ts":301.903, "dur":2.022, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/_cluster_2", "cat":"DMA", "ph":"X", "ts":304.283, "dur":2.151, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/_cluster_3", "cat":"DMA", "ph":"X", "ts":306.666, "dur":1.445, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool", "cat":"DMA", "ph":"X", "ts":308.392, "dur":6.504, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":315.040, "dur":36.390, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":351.580, "dur":36.390, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":388.115, "dur":36.397, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":424.656, "dur":18.852, "pid":0, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool", "cat":"DMA", "ph":"X", "ts":443.653, "dur":7.104, "pid":0, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0", "cat":"DPU", "ph":"X", "ts":239.918, "dur":42.810, "pid":1, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/cluster_0", "cat":"DPU", "ph":"X", "ts":289.829, "dur":6.862, "pid":1, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1", "cat":"DPU", "ph":"X", "ts":240.033, "dur":43.785, "pid":2, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/cluster_1", "cat":"DPU", "ph":"X", "ts":289.559, "dur":7.210, "pid":2, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2", "cat":"DPU", "ph":"X", "ts":240.226, "dur":45.585, "pid":3, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/cluster_2", "cat":"DPU", "ph":"X", "ts":289.675, "dur":8.785, "pid":3, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3", "cat":"DPU", "ph":"X", "ts":240.359, "dur":48.818, "pid":4, "tid":0},
// CHECK: {"name":"pool1?t_MaxPool/cluster_3", "cat":"DPU", "ph":"X", "ts":289.636, "dur":5.385, "pid":4, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"UPA", "ph":"X", "ts":23.175, "dur":37.684, "pid":5, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"UPA", "ph":"X", "ts":69.627, "dur":154.490, "pid":5, "tid":0},
// CHECK: {"name":"output?t_Output", "cat":"UPA", "ph":"X", "ts":454.155, "dur":63.481, "pid":5, "tid":0},
// CHECK: {"name":"conv1/WithoutBiases", "cat":"Layer", "ph":"X", "ts":0.000, "dur":289.177, "pid":6, "tid":0, "args":{"Layer type": "Convolution"}},
// CHECK: {"name":"pool1", "cat":"Layer", "ph":"X", "ts":240.062, "dur":210.695, "pid":6, "tid":1, "args":{"Layer type": "MaxPool"}},
// CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":454.155, "dur":63.481, "pid":6, "tid":0, "args":{"Layer type": "Convert"}}
// CHECK: ],
// CHECK: "displayTimeUnit": "ns"
// CHECK: }
