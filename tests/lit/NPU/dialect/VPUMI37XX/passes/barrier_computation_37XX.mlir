//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --barrier-computation-VPUMI37XX %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module @Test {

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1x2x1000xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1x2x1000xf16>
}

func.func @main(%arg0: memref<1x1x2x1000xf16>, %arg1: memref<1x1x2x1000xf16>) -> memref<1x1x2x1000xf16> {

    %buffer = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x2x1000xf16, @DDR>
    %buffer1 = VPURT.DeclareBuffer <DDR> <4000> -> memref<1x1x2x1000xf16, @DDR>

    %barrier0 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=0:ui8 } <0,-1> -> !VPURegMapped.Index<0:0:0>

    // CHECK:       %[[VAL0:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 0 : ui8}<0, 32> -> !VPURegMapped.Index<0:0:0>

    %barrier1 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <1,-1> -> !VPURegMapped.Index<0:0:1>

    // CHECK:       %[[VAL1:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, 33> -> !VPURegMapped.Index<0:0:1>

    %barrier2 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <2,-1> -> !VPURegMapped.Index<0:0:2>

    // CHECK:       %[[VAL2:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<2, 34> -> !VPURegMapped.Index<0:0:2>

    %barrier3 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <3,-1> -> !VPURegMapped.Index<0:0:3>

    // CHECK:       %[[VAL3:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<3, 35> -> !VPURegMapped.Index<0:0:3>

    %barrier4 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <4,-1> -> !VPURegMapped.Index<0:0:4>

    // CHECK:       %[[VAL4:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<4, 36> -> !VPURegMapped.Index<0:0:4>

    %barrier5 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <5,-1> -> !VPURegMapped.Index<0:0:5>

    // CHECK:       %[[VAL5:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<5, 37> -> !VPURegMapped.Index<0:0:5>

    %barrier6 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <6,-1> -> !VPURegMapped.Index<0:0:6>

    // CHECK:       %[[VAL6:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<6, 38> -> !VPURegMapped.Index<0:0:6>

    %barrier7 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <7,-1> -> !VPURegMapped.Index<0:0:7>

    // CHECK:       %[[VAL7:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<7, 39> -> !VPURegMapped.Index<0:0:7>

    %barrier8 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <8,-1> -> !VPURegMapped.Index<0:0:8>

    // CHECK:       %[[VAL8:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<8, 40> -> !VPURegMapped.Index<0:0:8>

    %barrier9 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <9,-1> -> !VPURegMapped.Index<0:0:9>

    // CHECK:       %[[VAL9:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<9, 41> -> !VPURegMapped.Index<0:0:9>

    %barrier10 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <10,-1> -> !VPURegMapped.Index<0:0:10>

    // CHECK:       %[[VAL10:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<10, 42> -> !VPURegMapped.Index<0:0:10>

    %barrier11 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <11,-1> -> !VPURegMapped.Index<0:0:11>

    // CHECK:       %[[VAL11:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<11, 43> -> !VPURegMapped.Index<0:0:11>

    %barrier12 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <12,-1> -> !VPURegMapped.Index<0:0:12>

    // CHECK:       %[[VAL12:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<12, 44> -> !VPURegMapped.Index<0:0:12>

    %barrier13 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <13,-1> -> !VPURegMapped.Index<0:0:13>

    // CHECK:       %[[VAL13:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<13, 45> -> !VPURegMapped.Index<0:0:13>

    %barrier14 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <14,-1> -> !VPURegMapped.Index<0:0:14>

    // CHECK:       %[[VAL14:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<14, 46> -> !VPURegMapped.Index<0:0:14>

    %barrier15 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <15,-1> -> !VPURegMapped.Index<0:0:15>

    // CHECK:       %[[VAL15:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<15, 47> -> !VPURegMapped.Index<0:0:15>

    %barrier16 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <16,-1> -> !VPURegMapped.Index<0:0:16>

    // CHECK:       %[[VAL16:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<16, 48> -> !VPURegMapped.Index<0:0:16>

    %barrier17 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <17,-1> -> !VPURegMapped.Index<0:0:17>

    // CHECK:       %[[VAL17:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<17, 49> -> !VPURegMapped.Index<0:0:17>

    %barrier18 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <18,-1> -> !VPURegMapped.Index<0:0:18>

    // CHECK:       %[[VAL18:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<18, 50> -> !VPURegMapped.Index<0:0:18>

    %barrier19 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <19,-1> -> !VPURegMapped.Index<0:0:19>

    // CHECK:       %[[VAL19:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<19, 51> -> !VPURegMapped.Index<0:0:19>

    %barrier20 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <20,-1> -> !VPURegMapped.Index<0:0:20>

    // CHECK:       %[[VAL20:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<20, 52> -> !VPURegMapped.Index<0:0:20>

    %barrier21 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <21,-1> -> !VPURegMapped.Index<0:0:21>

    // CHECK:       %[[VAL21:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<21, 53> -> !VPURegMapped.Index<0:0:21>

    %barrier22 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <22,-1> -> !VPURegMapped.Index<0:0:22>

    // CHECK:       %[[VAL22:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<22, 54> -> !VPURegMapped.Index<0:0:22>

    %barrier23 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <23,-1> -> !VPURegMapped.Index<0:0:23>

    // CHECK:       %[[VAL23:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<23, 55> -> !VPURegMapped.Index<0:0:23>

    %barrier24 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <24,-1> -> !VPURegMapped.Index<0:0:24>

    // CHECK:       %[[VAL24:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<24, 56> -> !VPURegMapped.Index<0:0:24>

    %barrier25 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <25,-1> -> !VPURegMapped.Index<0:0:25>

    // CHECK:       %[[VAL25:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<25, 57> -> !VPURegMapped.Index<0:0:25>

    %barrier26 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <26,-1> -> !VPURegMapped.Index<0:0:26>

    // CHECK:       %[[VAL26:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<26, 58> -> !VPURegMapped.Index<0:0:26>

    %barrier27 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <27,-1> -> !VPURegMapped.Index<0:0:27>

    // CHECK:       %[[VAL27:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<27, 59> -> !VPURegMapped.Index<0:0:27>

    %barrier28 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <28,-1> -> !VPURegMapped.Index<0:0:28>

    // CHECK:       %[[VAL28:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<28, 60> -> !VPURegMapped.Index<0:0:28>

    %barrier29 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <29,-1> -> !VPURegMapped.Index<0:0:29>

    // CHECK:       %[[VAL29:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<29, 61> -> !VPURegMapped.Index<0:0:29>

    %barrier30 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <30,-1> -> !VPURegMapped.Index<0:0:30>

    // CHECK:       %[[VAL30:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<30, 62> -> !VPURegMapped.Index<0:0:30>

    %barrier31 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <31,-1> -> !VPURegMapped.Index<0:0:31>

    // CHECK:       %[[VAL31:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<31, 63> -> !VPURegMapped.Index<0:0:31>

    %barrier32 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <0,-1> -> !VPURegMapped.Index<0:0:32>

    // CHECK:       %[[VAL32:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, 64> previousSameId(%2 : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:32>

    %barrier33 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <1,-1> -> !VPURegMapped.Index<0:0:33>

    // CHECK:       %[[VAL33:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, 65> previousSameId(%3 : !VPURegMapped.Index<0:0:1>) -> !VPURegMapped.Index<0:0:33>

    %barrier34 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <2,-1> -> !VPURegMapped.Index<0:0:34>

    // CHECK:       %[[VAL34:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<2, -1> previousSameId(%4 : !VPURegMapped.Index<0:0:2>) -> !VPURegMapped.Index<0:0:34>

    %barrier35 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <3,-1> -> !VPURegMapped.Index<0:0:35>

    // CHECK:       %[[VAL35:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<3, -1> previousSameId(%5 : !VPURegMapped.Index<0:0:3>) -> !VPURegMapped.Index<0:0:35>

    %barrier36 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <4,-1> -> !VPURegMapped.Index<0:0:36>

    // CHECK:       %[[VAL36:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<4, -1> previousSameId(%6 : !VPURegMapped.Index<0:0:4>) -> !VPURegMapped.Index<0:0:36>

    %barrier37 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <5,-1> -> !VPURegMapped.Index<0:0:37>

    // CHECK:       %[[VAL37:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<5, -1> previousSameId(%7 : !VPURegMapped.Index<0:0:5>) -> !VPURegMapped.Index<0:0:37>

    %barrier38 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <6,-1> -> !VPURegMapped.Index<0:0:38>

    // CHECK:       %[[VAL38:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<6, -1> previousSameId(%8 : !VPURegMapped.Index<0:0:6>) -> !VPURegMapped.Index<0:0:38>

    %barrier39 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <7,-1> -> !VPURegMapped.Index<0:0:39>

    // CHECK:       %[[VAL39:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<7, -1> previousSameId(%9 : !VPURegMapped.Index<0:0:7>) -> !VPURegMapped.Index<0:0:39>

    %barrier40 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <8,-1> -> !VPURegMapped.Index<0:0:40>

    // CHECK:       %[[VAL40:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<8, -1> previousSameId(%10 : !VPURegMapped.Index<0:0:8>) -> !VPURegMapped.Index<0:0:40>

    %barrier41 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <9,-1> -> !VPURegMapped.Index<0:0:41>

    // CHECK:       %[[VAL41:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<9, -1> previousSameId(%11 : !VPURegMapped.Index<0:0:9>) -> !VPURegMapped.Index<0:0:41>

    %barrier42 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <10,-1> -> !VPURegMapped.Index<0:0:42>

    // CHECK:       %[[VAL42:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<10, -1> previousSameId(%12 : !VPURegMapped.Index<0:0:10>) -> !VPURegMapped.Index<0:0:42>

    %barrier43 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <11,-1> -> !VPURegMapped.Index<0:0:43>

    // CHECK:       %[[VAL43:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<11, -1> previousSameId(%13 : !VPURegMapped.Index<0:0:11>) -> !VPURegMapped.Index<0:0:43>

    %barrier44 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <12,-1> -> !VPURegMapped.Index<0:0:44>

    // CHECK:       %[[VAL44:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<12, -1> previousSameId(%14 : !VPURegMapped.Index<0:0:12>) -> !VPURegMapped.Index<0:0:44>

    %barrier45 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <13,-1> -> !VPURegMapped.Index<0:0:45>

    // CHECK:       %[[VAL45:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<13, -1> previousSameId(%15 : !VPURegMapped.Index<0:0:13>) -> !VPURegMapped.Index<0:0:45>

    %barrier46 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <14,-1> -> !VPURegMapped.Index<0:0:46>

    // CHECK:       %[[VAL46:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<14, -1> previousSameId(%16 : !VPURegMapped.Index<0:0:14>) -> !VPURegMapped.Index<0:0:46>

    %barrier47 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <15,-1> -> !VPURegMapped.Index<0:0:47>

    // CHECK:       %[[VAL47:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<15, -1> previousSameId(%17 : !VPURegMapped.Index<0:0:15>) -> !VPURegMapped.Index<0:0:47>

    %barrier48 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <16,-1> -> !VPURegMapped.Index<0:0:48>

    // CHECK:       %[[VAL48:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<16, -1> previousSameId(%18 : !VPURegMapped.Index<0:0:16>) -> !VPURegMapped.Index<0:0:48>

    %barrier49 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <17,-1> -> !VPURegMapped.Index<0:0:49>

    // CHECK:       %[[VAL49:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<17, -1> previousSameId(%19 : !VPURegMapped.Index<0:0:17>) -> !VPURegMapped.Index<0:0:49>

    %barrier50 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <18,-1> -> !VPURegMapped.Index<0:0:50>

    // CHECK:       %[[VAL50:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<18, -1> previousSameId(%20 : !VPURegMapped.Index<0:0:18>) -> !VPURegMapped.Index<0:0:50>

    %barrier51 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <19,-1> -> !VPURegMapped.Index<0:0:51>

    // CHECK:       %[[VAL51:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<19, -1> previousSameId(%21 : !VPURegMapped.Index<0:0:19>) -> !VPURegMapped.Index<0:0:51>

    %barrier52 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <20,-1> -> !VPURegMapped.Index<0:0:52>

    // CHECK:       %[[VAL52:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<20, -1> previousSameId(%22 : !VPURegMapped.Index<0:0:20>) -> !VPURegMapped.Index<0:0:52>

    %barrier53 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <21,-1> -> !VPURegMapped.Index<0:0:53>

    // CHECK:       %[[VAL53:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<21, -1> previousSameId(%23 : !VPURegMapped.Index<0:0:21>) -> !VPURegMapped.Index<0:0:53>

    %barrier54 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <22,-1> -> !VPURegMapped.Index<0:0:54>

    // CHECK:       %[[VAL54:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<22, -1> previousSameId(%24 : !VPURegMapped.Index<0:0:22>) -> !VPURegMapped.Index<0:0:54>

    %barrier55 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <23,-1> -> !VPURegMapped.Index<0:0:55>

    // CHECK:       %[[VAL55:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<23, -1> previousSameId(%25 : !VPURegMapped.Index<0:0:23>) -> !VPURegMapped.Index<0:0:55>

    %barrier56 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <24,-1> -> !VPURegMapped.Index<0:0:56>

    // CHECK:       %[[VAL56:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<24, -1> previousSameId(%26 : !VPURegMapped.Index<0:0:24>) -> !VPURegMapped.Index<0:0:56>

    %barrier57 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <25,-1> -> !VPURegMapped.Index<0:0:57>

    // CHECK:       %[[VAL57:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<25, -1> previousSameId(%27 : !VPURegMapped.Index<0:0:25>) -> !VPURegMapped.Index<0:0:57>

    %barrier58 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <26,-1> -> !VPURegMapped.Index<0:0:58>

    // CHECK:       %[[VAL58:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<26, -1> previousSameId(%28 : !VPURegMapped.Index<0:0:26>) -> !VPURegMapped.Index<0:0:58>

    %barrier59 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <27,-1> -> !VPURegMapped.Index<0:0:59>

    // CHECK:       %[[VAL59:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<27, -1> previousSameId(%29 : !VPURegMapped.Index<0:0:27>) -> !VPURegMapped.Index<0:0:59>

    %barrier60 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <28,-1> -> !VPURegMapped.Index<0:0:60>

    // CHECK:       %[[VAL60:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<28, -1> previousSameId(%30 : !VPURegMapped.Index<0:0:28>) -> !VPURegMapped.Index<0:0:60>

    %barrier61 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <29,-1> -> !VPURegMapped.Index<0:0:61>

    // CHECK:       %[[VAL61:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<29, -1> previousSameId(%31 : !VPURegMapped.Index<0:0:29>) -> !VPURegMapped.Index<0:0:61>

    %barrier62 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <30,-1> -> !VPURegMapped.Index<0:0:62>

    // CHECK:       %[[VAL62:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<30, -1> previousSameId(%32 : !VPURegMapped.Index<0:0:30>) -> !VPURegMapped.Index<0:0:62>

    %barrier63 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <31,-1> -> !VPURegMapped.Index<0:0:63>

    // CHECK:       %[[VAL63:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<31, -1> previousSameId(%33 : !VPURegMapped.Index<0:0:31>) -> !VPURegMapped.Index<0:0:63>

    %barrier64 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <0,-1> -> !VPURegMapped.Index<0:0:64>

    // CHECK:       %[[VAL64:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> previousSameId(%34 : !VPURegMapped.Index<0:0:32>) -> !VPURegMapped.Index<0:0:64>

    %barrier65 = VPUMI37XX.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <1,-1> -> !VPURegMapped.Index<0:0:65>

    // CHECK:       %[[VAL65:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> previousSameId(%35 : !VPURegMapped.Index<0:0:33>) -> !VPURegMapped.Index<0:0:65>

    %d64 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier64 : !VPURegMapped.Index<0:0:64>) updates(%barrier65 : !VPURegMapped.Index<0:0:65>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:64>
    %d63 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d64 : !VPURegMapped.Index<0:0:64>) waits(%barrier63 : !VPURegMapped.Index<0:0:63>) updates(%barrier64 : !VPURegMapped.Index<0:0:64>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:63>
    %d62 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d63 : !VPURegMapped.Index<0:0:63>) waits(%barrier62 : !VPURegMapped.Index<0:0:62>) updates(%barrier63 : !VPURegMapped.Index<0:0:63>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:62>
    %d61 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d62 : !VPURegMapped.Index<0:0:62>) waits(%barrier61 : !VPURegMapped.Index<0:0:61>) updates(%barrier62 : !VPURegMapped.Index<0:0:62>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:61>
    %d60 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d61 : !VPURegMapped.Index<0:0:61>) waits(%barrier60 : !VPURegMapped.Index<0:0:60>) updates(%barrier61 : !VPURegMapped.Index<0:0:61>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:60>
    %d59 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d60 : !VPURegMapped.Index<0:0:60>) waits(%barrier59 : !VPURegMapped.Index<0:0:59>) updates(%barrier60 : !VPURegMapped.Index<0:0:60>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:59>
    %d58 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d59 : !VPURegMapped.Index<0:0:59>) waits(%barrier58 : !VPURegMapped.Index<0:0:58>) updates(%barrier59 : !VPURegMapped.Index<0:0:59>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:58>
    %d57 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d58 : !VPURegMapped.Index<0:0:58>) waits(%barrier57 : !VPURegMapped.Index<0:0:57>) updates(%barrier58 : !VPURegMapped.Index<0:0:58>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:57>
    %d56 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d57 : !VPURegMapped.Index<0:0:57>) waits(%barrier56 : !VPURegMapped.Index<0:0:56>) updates(%barrier57 : !VPURegMapped.Index<0:0:57>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:56>
    %d55 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d56 : !VPURegMapped.Index<0:0:56>) waits(%barrier55 : !VPURegMapped.Index<0:0:55>) updates(%barrier56 : !VPURegMapped.Index<0:0:56>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:55>
    %d54 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d55 : !VPURegMapped.Index<0:0:55>) waits(%barrier54 : !VPURegMapped.Index<0:0:54>) updates(%barrier55 : !VPURegMapped.Index<0:0:55>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:54>
    %d53 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d54 : !VPURegMapped.Index<0:0:54>) waits(%barrier53 : !VPURegMapped.Index<0:0:53>) updates(%barrier54 : !VPURegMapped.Index<0:0:54>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:53>
    %d52 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d53 : !VPURegMapped.Index<0:0:53>) waits(%barrier52 : !VPURegMapped.Index<0:0:52>) updates(%barrier53 : !VPURegMapped.Index<0:0:53>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:52>
    %d51 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d52 : !VPURegMapped.Index<0:0:52>) waits(%barrier51 : !VPURegMapped.Index<0:0:51>) updates(%barrier52 : !VPURegMapped.Index<0:0:52>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:51>
    %d50 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d51 : !VPURegMapped.Index<0:0:51>) waits(%barrier50 : !VPURegMapped.Index<0:0:50>) updates(%barrier51 : !VPURegMapped.Index<0:0:51>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:50>
    %d49 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d50 : !VPURegMapped.Index<0:0:50>) waits(%barrier49 : !VPURegMapped.Index<0:0:49>) updates(%barrier50 : !VPURegMapped.Index<0:0:50>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:49>
    %d48 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d49 : !VPURegMapped.Index<0:0:49>) waits(%barrier48 : !VPURegMapped.Index<0:0:48>) updates(%barrier49 : !VPURegMapped.Index<0:0:49>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:48>
    %d47 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d48 : !VPURegMapped.Index<0:0:48>) waits(%barrier47 : !VPURegMapped.Index<0:0:47>) updates(%barrier48 : !VPURegMapped.Index<0:0:48>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:47>
    %d46 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d47 : !VPURegMapped.Index<0:0:47>) waits(%barrier46 : !VPURegMapped.Index<0:0:46>) updates(%barrier47 : !VPURegMapped.Index<0:0:47>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:46>
    %d45 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d46 : !VPURegMapped.Index<0:0:46>) waits(%barrier45 : !VPURegMapped.Index<0:0:45>) updates(%barrier46 : !VPURegMapped.Index<0:0:46>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:45>
    %d44 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d45 : !VPURegMapped.Index<0:0:45>) waits(%barrier44 : !VPURegMapped.Index<0:0:44>) updates(%barrier45 : !VPURegMapped.Index<0:0:45>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:44>
    %d43 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d44 : !VPURegMapped.Index<0:0:44>) waits(%barrier43 : !VPURegMapped.Index<0:0:43>) updates(%barrier44 : !VPURegMapped.Index<0:0:44>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:43>
    %d42 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d43 : !VPURegMapped.Index<0:0:43>) waits(%barrier42 : !VPURegMapped.Index<0:0:42>) updates(%barrier43 : !VPURegMapped.Index<0:0:43>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:42>
    %d41 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d42 : !VPURegMapped.Index<0:0:42>) waits(%barrier41 : !VPURegMapped.Index<0:0:41>) updates(%barrier42 : !VPURegMapped.Index<0:0:42>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:41>
    %d40 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d41 : !VPURegMapped.Index<0:0:41>) waits(%barrier40 : !VPURegMapped.Index<0:0:40>) updates(%barrier41 : !VPURegMapped.Index<0:0:41>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:40>
    %d39 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d40 : !VPURegMapped.Index<0:0:40>) waits(%barrier39 : !VPURegMapped.Index<0:0:39>) updates(%barrier40 : !VPURegMapped.Index<0:0:40>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:39>
    %d38 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d39 : !VPURegMapped.Index<0:0:39>) waits(%barrier38 : !VPURegMapped.Index<0:0:38>) updates(%barrier39 : !VPURegMapped.Index<0:0:39>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:38>
    %d37 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d38 : !VPURegMapped.Index<0:0:38>) waits(%barrier37 : !VPURegMapped.Index<0:0:37>) updates(%barrier38 : !VPURegMapped.Index<0:0:38>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:37>
    %d36 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d37 : !VPURegMapped.Index<0:0:37>) waits(%barrier36 : !VPURegMapped.Index<0:0:36>) updates(%barrier37 : !VPURegMapped.Index<0:0:37>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:36>
    %d35 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d36 : !VPURegMapped.Index<0:0:36>) waits(%barrier35 : !VPURegMapped.Index<0:0:35>) updates(%barrier36 : !VPURegMapped.Index<0:0:36>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:35>
    %d34 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d35 : !VPURegMapped.Index<0:0:35>) waits(%barrier34 : !VPURegMapped.Index<0:0:34>) updates(%barrier35 : !VPURegMapped.Index<0:0:35>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:34>
    %d33 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d34 : !VPURegMapped.Index<0:0:34>) waits(%barrier33 : !VPURegMapped.Index<0:0:33>) updates(%barrier34 : !VPURegMapped.Index<0:0:34>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:33>
    %d32 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d33 : !VPURegMapped.Index<0:0:33>) waits(%barrier32 : !VPURegMapped.Index<0:0:32>) updates(%barrier33 : !VPURegMapped.Index<0:0:33>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:32>
    %d31 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d32 : !VPURegMapped.Index<0:0:32>) waits(%barrier31 : !VPURegMapped.Index<0:0:31>) updates(%barrier32 : !VPURegMapped.Index<0:0:32>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:31>
    %d30 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d31 : !VPURegMapped.Index<0:0:31>) waits(%barrier30 : !VPURegMapped.Index<0:0:30>) updates(%barrier31 : !VPURegMapped.Index<0:0:31>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:30>
    %d29 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d30 : !VPURegMapped.Index<0:0:30>) waits(%barrier29 : !VPURegMapped.Index<0:0:29>) updates(%barrier30 : !VPURegMapped.Index<0:0:30>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:29>
    %d28 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d29 : !VPURegMapped.Index<0:0:29>) waits(%barrier28 : !VPURegMapped.Index<0:0:28>) updates(%barrier29 : !VPURegMapped.Index<0:0:29>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:28>
    %d27 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d28 : !VPURegMapped.Index<0:0:28>) waits(%barrier27 : !VPURegMapped.Index<0:0:27>) updates(%barrier28 : !VPURegMapped.Index<0:0:28>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:27>
    %d26 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d27 : !VPURegMapped.Index<0:0:27>) waits(%barrier26 : !VPURegMapped.Index<0:0:26>) updates(%barrier27 : !VPURegMapped.Index<0:0:27>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:26>
    %d25 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d26 : !VPURegMapped.Index<0:0:26>) waits(%barrier25 : !VPURegMapped.Index<0:0:25>) updates(%barrier26 : !VPURegMapped.Index<0:0:26>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:25>
    %d24 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d25 : !VPURegMapped.Index<0:0:25>) waits(%barrier24 : !VPURegMapped.Index<0:0:24>) updates(%barrier25 : !VPURegMapped.Index<0:0:25>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:24>
    %d23 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d24 : !VPURegMapped.Index<0:0:24>) waits(%barrier23 : !VPURegMapped.Index<0:0:23>) updates(%barrier24 : !VPURegMapped.Index<0:0:24>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:23>
    %d22 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d23 : !VPURegMapped.Index<0:0:23>) waits(%barrier22 : !VPURegMapped.Index<0:0:22>) updates(%barrier23 : !VPURegMapped.Index<0:0:23>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:22>
    %d21 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d22 : !VPURegMapped.Index<0:0:22>) waits(%barrier21 : !VPURegMapped.Index<0:0:21>) updates(%barrier22 : !VPURegMapped.Index<0:0:22>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:21>
    %d20 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d21 : !VPURegMapped.Index<0:0:21>) waits(%barrier20 : !VPURegMapped.Index<0:0:20>) updates(%barrier21 : !VPURegMapped.Index<0:0:21>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:20>
    %d19 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d20 : !VPURegMapped.Index<0:0:20>) waits(%barrier19 : !VPURegMapped.Index<0:0:19>) updates(%barrier20 : !VPURegMapped.Index<0:0:20>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:19>
    %d18 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d19 : !VPURegMapped.Index<0:0:19>) waits(%barrier18 : !VPURegMapped.Index<0:0:18>) updates(%barrier19 : !VPURegMapped.Index<0:0:19>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:18>
    %d17 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d18 : !VPURegMapped.Index<0:0:18>) waits(%barrier17 : !VPURegMapped.Index<0:0:17>) updates(%barrier18 : !VPURegMapped.Index<0:0:18>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:17>
    %d16 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d17 : !VPURegMapped.Index<0:0:17>) waits(%barrier16 : !VPURegMapped.Index<0:0:16>) updates(%barrier17 : !VPURegMapped.Index<0:0:17>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:16>
    %d15 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d16 : !VPURegMapped.Index<0:0:16>) waits(%barrier15 : !VPURegMapped.Index<0:0:15>) updates(%barrier16 : !VPURegMapped.Index<0:0:16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:15>
    %d14 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d15 : !VPURegMapped.Index<0:0:15>) waits(%barrier14 : !VPURegMapped.Index<0:0:14>) updates(%barrier15 : !VPURegMapped.Index<0:0:15>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:14>
    %d13 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d14 : !VPURegMapped.Index<0:0:14>) waits(%barrier13 : !VPURegMapped.Index<0:0:13>) updates(%barrier14 : !VPURegMapped.Index<0:0:14>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:13>
    %d12 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d13 : !VPURegMapped.Index<0:0:13>) waits(%barrier12 : !VPURegMapped.Index<0:0:12>) updates(%barrier13 : !VPURegMapped.Index<0:0:13>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:12>
    %d11 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d12 : !VPURegMapped.Index<0:0:12>) waits(%barrier11 : !VPURegMapped.Index<0:0:11>) updates(%barrier12 : !VPURegMapped.Index<0:0:12>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:11>
    %d10 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d11 : !VPURegMapped.Index<0:0:11>) waits(%barrier10 : !VPURegMapped.Index<0:0:10>) updates(%barrier11 : !VPURegMapped.Index<0:0:11>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:10>
    %d9 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d10 : !VPURegMapped.Index<0:0:10>) waits(%barrier9 : !VPURegMapped.Index<0:0:9>) updates(%barrier10 : !VPURegMapped.Index<0:0:10>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:9>
    %d8 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d9 : !VPURegMapped.Index<0:0:9>) waits(%barrier8 : !VPURegMapped.Index<0:0:8>) updates(%barrier9 : !VPURegMapped.Index<0:0:9>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:8>
    %d7 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d8 : !VPURegMapped.Index<0:0:8>) waits(%barrier7 : !VPURegMapped.Index<0:0:7>) updates(%barrier8 : !VPURegMapped.Index<0:0:8>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:7>
    %d6 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d7 : !VPURegMapped.Index<0:0:7>) waits(%barrier6 : !VPURegMapped.Index<0:0:6>) updates(%barrier7 : !VPURegMapped.Index<0:0:7>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:6>
    %d5 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d6 : !VPURegMapped.Index<0:0:6>) waits(%barrier5 : !VPURegMapped.Index<0:0:5>) updates(%barrier6 : !VPURegMapped.Index<0:0:6>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>
    %d4 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d5 : !VPURegMapped.Index<0:0:5>) waits(%barrier4 : !VPURegMapped.Index<0:0:4>) updates(%barrier5 : !VPURegMapped.Index<0:0:5>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
    %d3 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d4 : !VPURegMapped.Index<0:0:4>) waits(%barrier3 : !VPURegMapped.Index<0:0:3>) updates(%barrier4 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %d2 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d3 : !VPURegMapped.Index<0:0:3>) waits(%barrier2 : !VPURegMapped.Index<0:0:2>) updates(%barrier3 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %d1 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d2 : !VPURegMapped.Index<0:0:2>) waits(%barrier1 : !VPURegMapped.Index<0:0:1>) updates(%barrier2 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %d0 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) nextDMAIdx(%d1 : !VPURegMapped.Index<0:0:1>) waits(%barrier0 : !VPURegMapped.Index<0:0:0>) updates(%barrier1 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %mi = VPUMI37XX.MappedInference dmas(%d0 : !VPURegMapped.Index<0:0:0>) dmaCount([65, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(65) -> !VPURegMapped.Index<0:0:0>



    return %arg1 : memref<1x1x2x1000xf16>
}
}
