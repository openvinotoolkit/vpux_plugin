//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %data_path_npu%/profiling-30XX.mlir.txt
// RUN: prof_parser -b %t -m | FileCheck %s
// REQUIRES: arch-VPUX30XX

//CHECK: {
//CHECK: platform: {
//CHECK: device: 1
//CHECK: },
//CHECK: profilingBuffer: {
//CHECK: sections: [ {
//CHECK: type: 1,
//CHECK: size: 2048
//CHECK: }, {
//CHECK: type: 2,
//CHECK: offset: 2048,
//CHECK: size: 72
//CHECK: }, {
//CHECK: type: 4,
//CHECK: offset: 2176,
//CHECK: size: 144
//CHECK: } ],
//CHECK: size: 2368
//CHECK: },
//CHECK: dmaTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_1_13",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_1_13",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution",
//CHECK: waitBarriers: [ 0 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 1 ],
//CHECK: dataIndex: 1
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 2
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0",
//CHECK: waitBarriers: [ 2 ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 3
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 4
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_2",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_2",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 5
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_3",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_3",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 3 ],
//CHECK: dataIndex: 6
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: dataIndex: 7
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 8
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 9
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_2",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_2",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 10
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_3",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_3",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 11
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 12
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 13
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 14
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 15
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 16
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: isProfBegin: true
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ],
//CHECK: dataIndex: 17
//CHECK: } ],
//CHECK: dpuTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0",
//CHECK: taskId: 1,
//CHECK: numVariants: 20,
//CHECK: maxVariants: 20,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 1,
//CHECK: numVariants: 20,
//CHECK: maxVariants: 20,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2",
//CHECK: clusterId: 2,
//CHECK: taskId: 1,
//CHECK: numVariants: 20,
//CHECK: maxVariants: 20,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3",
//CHECK: clusterId: 3,
//CHECK: taskId: 1,
//CHECK: numVariants: 20,
//CHECK: maxVariants: 20,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_0",
//CHECK: taskId: 2,
//CHECK: numVariants: 12,
//CHECK: maxVariants: 12,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 2,
//CHECK: numVariants: 12,
//CHECK: maxVariants: 12,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_2",
//CHECK: clusterId: 2,
//CHECK: taskId: 2,
//CHECK: numVariants: 12,
//CHECK: maxVariants: 12,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_3",
//CHECK: clusterId: 3,
//CHECK: taskId: 2,
//CHECK: numVariants: 6,
//CHECK: maxVariants: 12,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [  ]
//CHECK: } ],
//CHECK: swTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 0 ],
//CHECK: taskType: "Convert",
//CHECK: clusterSize: 3
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution",
//CHECK: waitBarriers: [ 1 ],
//CHECK: updateBarriers: [ 2 ],
//CHECK: taskType: "Permute",
//CHECK: clusterSize: 3,
//CHECK: dataIndex: 1
//CHECK: }, {
//CHECK: name: "output?t_Output",
//CHECK: waitBarriers: [ 6 ],
//CHECK: updateBarriers: [  ],
//CHECK: taskType: "Convert",
//CHECK: clusterSize: 3,
//CHECK: dataIndex: 2
//CHECK: } ]
//CHECK: }
