//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --vertical-fusion-tiling %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


!qElemType = !quant.uniform<u8:f16, 0.022970187430288277:92>
!qElemType1 = !quant.uniform<u8:f16, 0.026811079885445389:79>
!qElemType2 = !quant.uniform<u8<0:254>:f16:0, {0.0015777572402803917:127,0.0014889139359391581:127,0.0021083918143445114:127,0.0019076836156094168:127,0.002125650878966324:127,0.0015942034289592832:127,0.0013417831556064876:127,0.0013941065298290704:127,0.0015372968330158022:127,0.0022980439381336602:127,0.0019483130982541662:127,0.0013906989745267732:127,0.0019419781101031566:127,0.001397803895116791:127,0.0016154645700154342:127,0.0019699626081571804:127,0.0020237670639368494:127,0.0015820100551515114:127,0.002136513708144661:127,0.0015983497063944659:127,0.0013165938572620782:127,0.0023659495856818251:127,0.0020228220721868078:127,0.0014110315503097894:127,0.0016812501460548462:127,0.0013487547870696061:127,0.0017490208618284211:127,0.0016929984796704269:127,0.0023835024495763102:127,0.0016608983278274536:127,0.0014221570857866543:127,0.0020535398186661128:127}>
!qElemType3 = !quant.uniform<u8<0:254>:f16:0, {0.0050877895880871871:127,0.0036900820225242554:127,0.0029946140886291744:127,0.0046036679913678503:127,0.0030745821674977702:127,0.0040151899255166839:127,0.0026243648191136637:127,0.002185353378611287:127,0.0053012159865672192:127,0.0019454215689906924:127,0.0043949132829200566:127,0.0021944085913380299:127,0.0045137020546620289:127,0.0027246975053952433:127,0.0026398010141267551:127,0.0035128525392277036:127,0.0055316268928407688:127,0.0037249932138938603:127,0.0031018390899568093:127,0.0031733273521183042:127,0.0048826055263909767:127,0.0055176643874701552:127,0.0026252196999046748:127,0.0023261205417903388:127,0.0029696068895144726:127,0.0052650167247441813:127,0.0054816298597440945:127,0.0024053254934746451:127,0.0044391516625411865:127,0.0043663044614116039:127,0.0028751257836349365:127,0.0051064467805577076:127}>
!qElemType4 = !quant.uniform<u8<0:254>:f16:0, {0.0021634723727158673:127,0.0025575409724017768:127,0.0038781130877066786:127,0.0032656551815393401:127,0.0020378891407974121:127,0.0035840688258644165:127,0.0028856976295080711:127,0.0023135664894824892:127,0.0034765291401720423:127,0.0033485044644573541:127,0.0035138308532594695:127,0.0018750902001313336:127,0.0018750902001313336:127,0.0018750902001313336:127,0.0018750902001313336:127,0.0018750902001313336:127}>
!qElemType5 = !quant.uniform<u8:f16, 0.016268887239343981:73>
!qElemType6 = !quant.uniform<u8:f16, 0.011832303860608269:83>
!qElemType7 = !quant.uniform<u8:f16, 0.020894650851978974:128>

func.func @TileEltwiseWithTwoVFLinkedInputs(%arg0: tensor<1x16x256x256x!qElemType1, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {

   %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
   %cst_2 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
   %cst_3 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
   %cst_4 = const.Declare tensor<16x32x3x3x!qElemType4, {order = #NHWC}> = dense<1.0> : tensor<16x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType4>, #const.Reorder<#NHWC>]
   %cst_5 = const.Declare tensor<32x32x3x3x!qElemType3, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType6>, #const.Reorder<#NHWC>]
   %cst_6 = const.Declare tensor<32x16x3x3x!qElemType2, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]


   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType1, {order = #NHWC}>, %cst_6 as %arg2: tensor<32x16x3x3x!qElemType2, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>, %cst_5 as %arg4: tensor<32x32x3x3x!qElemType3, {order = #NHWC}>, %cst_2 as %arg5: tensor<32x1x1x4xsi32>, %cst_4 as %arg6: tensor<16x32x3x3x!qElemType4, {order = #NHWC}>, %cst_3 as %arg7: tensor<16x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 6, 1]} -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.300048828125 : f64, lrelu_mult = 1229 : i64, lrelu_shift = 12 : i64>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]}
         -> tensor<1x32x256x256x!qElemType5, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4, %arg5)
         {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.300048828125 : f64, lrelu_mult = 1229 : i64, lrelu_shift = 12 : i64>,
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]}
         -> tensor<1x32x256x256x!qElemType6, {order = #NHWC}>
      %3 = VPU.NCE.Eltwise(%1, %2)
         {is_inplace = true, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPETask<mode = <NOOP>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [34118], in2_quant_mult = [24814], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [24503], quant_post_shift = 0 : i64, quant_shift = [30]>}
         -> tensor<1x32x256x256x!qElemType7, {order = #NHWC}>
      %4 = VPU.NCE.Convolution(%3, %arg6, %arg7)
         {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.300048828125 : f64, lrelu_mult = 1229 : i64, lrelu_shift = 12 : i64>,
         rawFilterShape = [16, 32, 3, 3], strides = [1, 1]}
         -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    return %0 : tensor<1x16x256x256x!qElemType, {order = #NHWC}>


    // CHECK: [[SLICEARG0TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 46, 256]
    // CHECK: [[CONV0TILE0:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE0]], %cst_4, %cst)
    // CHECK: [[CONV1TILE0:%.+]] = VPU.NCE.Convolution([[CONV0TILE0]], %cst_3, %cst_0)
    // CHECK: [[SLICETILE0:%.+]] = VPU.Slice [[CONV0TILE0]] [0, 0, 0, 0] [1, 32, 44, 256]
    // CHECK: [[ELTWISETILE0:%.+]] = VPU.NCE.Eltwise([[SLICETILE0]], [[CONV1TILE0]])
    // CHECK: [[CONV2TILE0:%.+]] = VPU.NCE.Convolution([[ELTWISETILE0]], %cst_2, %cst_1)
    // CHECK: [[SLICEARG0TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 40, 0] [1, 16, 49, 256]
    // CHECK: [[CONV0TILE1:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE1]], %cst_4, %cst)
    // CHECK: [[CONV1TILE1:%.+]] = VPU.NCE.Convolution([[CONV0TILE1]], %cst_3, %cst_0)
    // CHECK: [[SLICETILE1:%.+]] = VPU.Slice [[CONV0TILE1]] [0, 0, 1, 0] [1, 32, 45, 256]
    // CHECK: [[ELTWISETILE1:%.+]] = VPU.NCE.Eltwise([[SLICETILE1]], [[CONV1TILE1]])
    // CHECK: [[CONV2TILE1:%.+]] = VPU.NCE.Convolution([[ELTWISETILE1]], %cst_2, %cst_1)
    // CHECK: [[SLICEARG0TILE2:%.+]] = VPU.Slice %arg0 [0, 0, 83, 0] [1, 16, 49, 256]
    // CHECK: [[CONV0TILE2:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE2]], %cst_4, %cst)
    // CHECK: [[CONV1TILE2:%.+]] = VPU.NCE.Convolution([[CONV0TILE2]], %cst_3, %cst_0)
    // CHECK: [[SLICETILE2:%.+]] = VPU.Slice [[CONV0TILE2]] [0, 0, 1, 0] [1, 32, 45, 256]
    // CHECK: [[ELTWISETILE2:%.+]] = VPU.NCE.Eltwise([[SLICETILE2]], [[CONV1TILE2]])
    // CHECK: [[CONV2TILE2:%.+]] = VPU.NCE.Convolution([[ELTWISETILE2]], %cst_2, %cst_1)
    // CHECK: [[SLICEARG0TILE3:%.+]] = VPU.Slice %arg0 [0, 0, 126, 0] [1, 16, 49, 256]
    // CHECK: [[CONV0TILE3:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE3]], %cst_4, %cst)
    // CHECK: [[CONV1TILE3:%.+]] = VPU.NCE.Convolution([[CONV0TILE3]], %cst_3, %cst_0)
    // CHECK: [[SLICETILE3:%.+]] = VPU.Slice [[CONV0TILE3]] [0, 0, 1, 0] [1, 32, 45, 256]
    // CHECK: [[ELTWISETILE3:%.+]] = VPU.NCE.Eltwise([[SLICETILE3]], [[CONV1TILE3]])
    // CHECK: [[CONV2TILE3:%.+]] = VPU.NCE.Convolution([[ELTWISETILE3]], %cst_2, %cst_1)
    // CHECK: [[SLICEARG0TILE4:%.+]] = VPU.Slice %arg0 [0, 0, 169, 0] [1, 16, 48, 256]
    // CHECK: [[CONV0TILE4:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE4]], %cst_4, %cst)
    // CHECK: [[CONV1TILE4:%.+]] = VPU.NCE.Convolution([[CONV0TILE4]], %cst_3, %cst_0)
    // CHECK: [[SLICETILE4:%.+]] = VPU.Slice [[CONV0TILE4]] [0, 0, 1, 0] [1, 32, 44, 256]
    // CHECK: [[ELTWISETILE4:%.+]] = VPU.NCE.Eltwise([[SLICETILE4]], [[CONV1TILE4]])
    // CHECK: [[CONV2TILE4:%.+]]  = VPU.NCE.Convolution([[ELTWISETILE4]], %cst_2, %cst_1)
    // CHECK: [[SLICEARG0TILE5:%.+]] = VPU.Slice %arg0 [0, 0, 211, 0] [1, 16, 45, 256]
    // CHECK: [[CONV0TILE5:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE5]], %cst_4, %cst)
    // CHECK: [[CONV1TILE5:%.+]] = VPU.NCE.Convolution([[CONV0TILE5]], %cst_3, %cst_0)
    // CHECK: [[SLICETILE5:%.+]] = VPU.Slice [[CONV0TILE5]] [0, 0, 1, 0] [1, 32, 43, 256]
    // CHECK: [[ELTWISETILE5:%.+]] = VPU.NCE.Eltwise([[SLICETILE5]], [[CONV1TILE5]])
    // CHECK: [[CONV2TILE5:%.+]] = VPU.NCE.Convolution(%34, %cst_2, %cst_1)

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV2TILE0]], [[CONV2TILE1]], [[CONV2TILE2]], [[CONV2TILE3]], [[CONV2TILE4]], [[CONV2TILE5]])
    // CHECK-SAME: {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]]}

    // CHECK: return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @TileEltwiseChain(%arg0: tensor<1x64x56x56xf16, {order = #NHWC}>) -> tensor<1x256x56x56xf16, {order = #NHWC}> {

   %cst_1 = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
   %cst_2 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
   %cst_3 = const.Declare tensor<64x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x3x3xf16>, [#const.Reorder<#NHWC>]
   %cst_4 = const.Declare tensor<256x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x64x1x1xf16>, [#const.Reorder<#NHWC>]
   %cst_5 = const.Declare tensor<64x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x256x1x1xf16>, [#const.Reorder<#NHWC>]
   %cst_6 = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
   

   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x56x56xf16, {order = #NHWC}>, 
                            %cst_4 as %arg2: tensor<256x64x1x1xf16, {order = #NHWC}>, 
                            %cst_1 as %arg3: tensor<256x1x1x4xsi32>, 
                            %cst_6 as %arg4: tensor<64x64x1x1xf16, {order = #NHWC}>, 
                            %cst_2 as %arg5: tensor<64x1x1x4xsi32>, 
                            %cst_3 as %arg6: tensor<64x64x3x3xf16, {order = #NHWC}>, 
                            %cst_2 as %arg7: tensor<64x1x1x4xsi32>, 
                            %cst_4 as %arg8: tensor<256x64x1x1xf16, {order = #NHWC}>, 
                            %cst_1 as %arg9: tensor<256x1x1x4xsi32>, 
                            %cst_5 as %arg10: tensor<64x256x1x1xf16, {order = #NHWC}>, 
                            %cst_2 as %arg11: tensor<64x1x1x4xsi32>, 
                            %cst_3 as %arg12: tensor<64x64x3x3xf16, {order = #NHWC}>, 
                            %cst_2 as %arg13: tensor<64x1x1x4xsi32>, 
                            %cst_4 as %arg14: tensor<256x64x1x1xf16, {order = #NHWC}>, 
                            %cst_1 as %arg15: tensor<256x1x1x4xsi32>, 
                            %cst_5 as %arg16: tensor<64x256x1x1xf16, {order = #NHWC}>, 
                            %cst_2 as %arg17: tensor<64x1x1x4xsi32>, 
                            %cst_3 as %arg18: tensor<64x64x3x3xf16, {order = #NHWC}>, 
                            %cst_2 as %arg19: tensor<64x1x1x4xsi32>, 
                            %cst_4 as %arg20: tensor<256x64x1x1xf16, {order = #NHWC}>, 
                            %cst_1 as %arg21: tensor<256x1x1x4xsi32>) 
   attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x256x56x56xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [256, 64, 1, 1], strides = [1, 1]} -> tensor<1x256x56x56xf16, {order = #NHWC}> 
      %2 = VPU.NCE.Convolution(%arg1, %arg4, %arg5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 64, 1, 1], strides = [1, 1]} -> tensor<1x64x56x56xf16, {order = #NHWC}> 
      %3 = VPU.NCE.Convolution(%2, %arg6, %arg7) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} -> tensor<1x64x56x56xf16, {order = #NHWC}> 
      %4 = VPU.NCE.Convolution(%3, %arg8, %arg9) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [256, 64, 1, 1], strides = [1, 1]} -> tensor<1x256x56x56xf16, {order = #NHWC}> 
      %5 = VPU.NCE.Eltwise(%4, %1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x256x56x56xf16, {order = #NHWC}> 
      %6 = VPU.NCE.Convolution(%5, %arg10, %arg11) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 256, 1, 1], strides = [1, 1]} -> tensor<1x64x56x56xf16, {order = #NHWC}> 
      %7 = VPU.NCE.Convolution(%6, %arg12, %arg13) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} -> tensor<1x64x56x56xf16, {order = #NHWC}> 
      %8 = VPU.NCE.Convolution(%7, %arg14, %arg15) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [256, 64, 1, 1], strides = [1, 1]} -> tensor<1x256x56x56xf16, {order = #NHWC}> 
      %9 = VPU.NCE.Eltwise(%8, %5) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x256x56x56xf16, {order = #NHWC}> 
      %10 = VPU.NCE.Convolution(%9, %arg16, %arg17) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 256, 1, 1], strides = [1, 1]} -> tensor<1x64x56x56xf16, {order = #NHWC}> 
      %11 = VPU.NCE.Convolution(%10, %arg18, %arg19) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} -> tensor<1x64x56x56xf16, {order = #NHWC}> 
      %12 = VPU.NCE.Convolution(%11, %arg20, %arg21) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [256, 64, 1, 1], strides = [1, 1]} -> tensor<1x256x56x56xf16, {order = #NHWC}> 
      %13 = VPU.NCE.Eltwise(%12, %9) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x256x56x56xf16, {order = #NHWC}> 
      VPU.Yield %13
    }


    return %0 : tensor<1x256x56x56xf16, {order = #NHWC}>


    // CHECK: [[SLICEARG0TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 64, 56, 31]
    // CHECK: [[SLICE0TILE0:%.+]] = VPU.Slice [[SLICEARG0TILE0]] [0, 0, 0, 0] [1, 64, 56, 30]
    // CHECK: [[CONV0TILE0:%.+]] = VPU.NCE.Convolution([[SLICE0TILE0]]
    // CHECK: [[SLICE1ARG0TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 64, 56, 31]
    // CHECK: [[CONV1TILE0:%.+]] = VPU.NCE.Convolution([[SLICE1ARG0TILE0]],
    // CHECK: [[CONV2TILE0:%.+]] = VPU.NCE.Convolution([[CONV1TILE0]],
    // CHECK: [[CONV3TILE0:%.+]] = VPU.NCE.Convolution([[CONV2TILE0]],
    // CHECK: [[ELTWISE0TILE0:%.+]] = VPU.NCE.Eltwise([[CONV3TILE0]], [[CONV0TILE0]])
    // CHECK: [[CONV4TILE0:%.+]] = VPU.NCE.Convolution([[ELTWISE0TILE0]],
    // CHECK: [[CONV5TILE0:%.+]] = VPU.NCE.Convolution([[CONV4TILE0]],
    // CHECK: [[CONV6TILE0:%.+]] = VPU.NCE.Convolution([[CONV5TILE0]],
    // CHECK: [[SLICE0TILE0:%.+]] = VPU.Slice [[ELTWISE0TILE0]] [0, 0, 0, 0] [1, 256, 56, 29]
    // CHECK: [[ELTWISE1TILE0:%.+]] = VPU.NCE.Eltwise([[CONV6TILE0]], [[SLICE0TILE0]])
    // CHECK: [[CONV7TILE0:%.+]] = VPU.NCE.Convolution([[ELTWISE1TILE0]],
    // CHECK: [[CONV8TILE0:%.+]] = VPU.NCE.Convolution([[CONV7TILE0]],
    // CHECK: [[CONV9TILE0:%.+]] = VPU.NCE.Convolution([[CONV8TILE0]],
    // CHECK: [[SLICE1TILE0:%.+]] = VPU.Slice [[ELTWISE1TILE0]] [0, 0, 0, 0] [1, 256, 56, 28]
    // CHECK: [[ELTWISE2TILE0:%.+]] = VPU.NCE.Eltwise([[CONV9TILE0]], [[SLICE1TILE0]])

    // CHECK: [[SLICEARG0TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 25] [1, 64, 56, 31]
    // CHECK: [[SLICE0TILE1:%.+]] = VPU.Slice [[SLICEARG0TILE1]] [0, 0, 0, 1] [1, 64, 56, 30]
    // CHECK: [[CONV0TILE1:%.+]] = VPU.NCE.Convolution([[SLICE0TILE1]]
    // CHECK: [[SLICE1ARG0TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 25] [1, 64, 56, 31]
    // CHECK: [[CONV1TILE1:%.+]] = VPU.NCE.Convolution([[SLICE1ARG0TILE1]],
    // CHECK: [[CONV2TILE1:%.+]] = VPU.NCE.Convolution([[CONV1TILE1]],
    // CHECK: [[CONV3TILE1:%.+]] = VPU.NCE.Convolution([[CONV2TILE1]],
    // CHECK: [[ELTWISE0TILE1:%.+]] = VPU.NCE.Eltwise([[CONV3TILE1]], [[CONV0TILE1]])
    // CHECK: [[CONV4TILE1:%.+]] = VPU.NCE.Convolution([[ELTWISE0TILE1]],
    // CHECK: [[CONV5TILE1:%.+]] = VPU.NCE.Convolution([[CONV4TILE1]],
    // CHECK: [[CONV6TILE1:%.+]] = VPU.NCE.Convolution([[CONV5TILE1]],
    // CHECK: [[SLICE0TILE1:%.+]] = VPU.Slice [[ELTWISE0TILE1]] [0, 0, 0, 1] [1, 256, 56, 29]
    // CHECK: [[ELTWISE1TILE1:%.+]] = VPU.NCE.Eltwise([[CONV6TILE1]], [[SLICE0TILE1]])
    // CHECK: [[CONV7TILE1:%.+]] = VPU.NCE.Convolution([[ELTWISE1TILE1]],
    // CHECK: [[CONV8TILE1:%.+]] = VPU.NCE.Convolution([[CONV7TILE1]],
    // CHECK: [[CONV9TILE1:%.+]] = VPU.NCE.Convolution([[CONV8TILE1]],
    // CHECK: [[SLICE1TILE1:%.+]] = VPU.Slice [[ELTWISE1TILE1]] [0, 0, 0, 1] [1, 256, 56, 28]
    // CHECK: [[ELTWISE2TILE1:%.+]] = VPU.NCE.Eltwise([[CONV9TILE1]], [[SLICE1TILE1]])
    

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[ELTWISE2TILE0]], [[ELTWISE2TILE1]])
    // CHECK-SAME: {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 28]]}

    // CHECK: return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @main(%arg0: tensor<1x128x68x120xf16, {order = #NHWC}>, %arg1: tensor<1x32x272x480xf16, {order = #NHWC}>,
%arg2: tensor<1x64x136x240xf16, {order = #NHWC}>) -> tensor<1x32x272x480xf16, {order = #NHWC}> {

    %cst = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<1x64x275x483xi1, {order = #NHWC}> = dense<1> : tensor<1x64x275x483xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    %cst_2 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_3 = const.Declare tensor<1x128x139x243xi1, {order = #NHWC}> = dense<1> : tensor<1x128x139x243xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]

    %cst_4 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<128x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<128x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_6 = const.Declare tensor<64x128x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_7 = const.Declare tensor<32x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_8 = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_9 = const.Declare tensor<32x64x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<32x64x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_10 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_11 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_12 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_13 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
  
   %storage_0 = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 128, 68, 120], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>, seDepth = 1 : i64, seSize = 128 : i64} -> tensor<1x1x139x243xi32, {order = #NHWC}>
   %storage_1 = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 136, 240], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>, seDepth = 1 : i64, seSize = 64 : i64} -> tensor<1x1x275x483xi32, {order = #NHWC}>
   %vf_0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x128x68x120xf16, {order = #NHWC}>, 
                             %cst_4 as %arg4: tensor<64x128x1x1xf16, {order = #NHWC}>, 
                             %cst_12 as %arg5: tensor<64x1x1x4xsi32>, 
                             %cst_5 as %arg6: tensor<128x64x3x3xf16, {order = #NHWC}>, 
                             %cst_13 as %arg7: tensor<128x1x1x4xsi32>, 
                             %cst_3 as %arg8: tensor<1x128x139x243xi1, {order = #NHWC}>, 
                             %storage_0 as %arg9: tensor<1x1x139x243xi32, {order = #NHWC}>, 
                             %cst_6 as %arg10: tensor<64x128x4x4xf16, {order = #NHWC}>, 
                             %cst_2 as %arg11: tensor<64x1x1x4xsi32>, 
                             %arg2 as %arg12: tensor<1x64x136x240xf16, {order = #NHWC}>, 
                             %cst_7 as %arg13: tensor<32x64x1x1xf16, {order = #NHWC}>, 
                             %cst_11 as %arg14: tensor<32x1x1x4xsi32>, 
                             %cst_8 as %arg15: tensor<64x32x3x3xf16, {order = #NHWC}>, 
                             %cst_12 as %arg16: tensor<64x1x1x4xsi32>, 
                             %cst_1 as %arg17: tensor<1x64x275x483xi1, {order = #NHWC}>, 
                             %storage_1 as %arg18: tensor<1x1x275x483xi32, {order = #NHWC}>, 
                             %cst_9 as %arg19: tensor<32x64x4x4xf16, {order = #NHWC}>, 
                             %cst as %arg20: tensor<32x1x1x4xsi32>, 
                             %arg1 as %arg21: tensor<1x32x272x480xf16, {order = #NHWC}>, 
                             %cst_10 as %arg22: tensor<32x32x3x3xf16, {order = #NHWC}>, 
                             %cst_11 as %arg23: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 5]} -> tensor<1x32x272x480xf16, {order = #NHWC}> {
      %0 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      %1 = VPU.NCE.Convolution(%0, %arg6, %arg7) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <LRELUX>, clamp_low = 0 : i64, clamp_high = 218 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [128, 64, 3, 3], strides = [1, 1]} -> tensor<1x128x68x120xf16, {order = #NHWC}> 
      %2 = VPU.NCE.Eltwise(%1, %arg3) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [29920], quant_shift = [30], quant_post_shift = 0 : i64, in1_quant_mult = [19228], in2_quant_mult = [34108], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x128x68x120xf16, {order = #NHWC}> 
      %3 = VPU.GroupSparseTensor(%2, %arg8, %arg9) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>} -> !VPU.SparseTensor<data=tensor<1x128x68x120xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x139x243xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x139x243xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>
      %4 = VPU.NCE.Convolution(%3, %arg10, %arg11) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <LRELUX>, clamp_low = 0 : i64, clamp_high = 218 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 128, 4, 4], strides = [1, 1]} -> tensor<1x64x136x240xf16, {order = #NHWC}> 
      %5 = VPU.NCE.Eltwise(%4, %arg12) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [25541], quant_shift = [30], quant_post_shift = 0 : i64, in1_quant_mult = [19228], in2_quant_mult = [40092], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x64x136x240xf16, {order = #NHWC}> 
      %6 = VPU.NCE.Convolution(%5, %arg13, %arg14) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <LRELUX>, clamp_low = 0 : i64, clamp_high = 223 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 64, 1, 1], strides = [1, 1]} -> tensor<1x32x136x240xf16, {order = #NHWC}> 
      %7 = VPU.NCE.Convolution(%6, %arg15, %arg16) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <LRELUX>, clamp_low = 0 : i64, clamp_high = 218 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 32, 3, 3], strides = [1, 1]} -> tensor<1x64x136x240xf16, {order = #NHWC}> 
      %8 = VPU.NCE.Eltwise(%7, %5) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [20266], quant_shift = [30], quant_post_shift = 0 : i64, in1_quant_mult = [19228], in2_quant_mult = [42038], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x64x136x240xf16, {order = #NHWC}> 
      %9 = VPU.GroupSparseTensor(%8, %arg17, %arg18) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>} -> !VPU.SparseTensor<data=tensor<1x64x136x240xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x275x483xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x275x483xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>
      %10 = VPU.NCE.Convolution(%9, %arg19, %arg20) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <LRELUX>, clamp_low = 0 : i64, clamp_high = 218 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 64, 4, 4], strides = [1, 1]} -> tensor<1x32x272x480xf16, {order = #NHWC}> 
      %11 = VPU.NCE.Eltwise(%10, %arg21) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [22972], quant_shift = [30], quant_post_shift = 0 : i64, in1_quant_mult = [19228], in2_quant_mult = [40714], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x32x272x480xf16, {order = #NHWC}> 
      %12 = VPU.NCE.Convolution(%11, %arg22, %arg23) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <LRELUX>, clamp_low = 0 : i64, clamp_high = 218 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x272x480xf16, {order = #NHWC}> 
      VPU.Yield %12 
    }

    return %vf_0 : tensor<1x32x272x480xf16, {order = #NHWC}>


    // CHECK: [[SLICE_ARG0_0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 128, 68, 27] 
    // CHECK: VPU.NCE.Convolution
    // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution
    // CHECK: [[SLICE_ARG0_1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 128, 68, 27] 
    // CHECK: [[SLICE_0:%.+]] = VPU.Slice [[SLICE_ARG0_1]] [0, 0, 0, 0] [1, 128, 68, 26] 
    // CHECK: [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[SLICE_0]])  
    // CHECK: [[GROUP_SPARSE_0:%.+]] = VPU.GroupSparseTensor([[ELTWISE_0]]
    // CHECK: [[CONV_2:%.+]] = VPU.NCE.Convolution([[GROUP_SPARSE_0]], %cst_5, %cst_1)  
    // CHECK: [[SLICE_ARG2:%.+]] = VPU.Slice %arg2 [0, 0, 0, 0] [1, 64, 136, 50] 
    // CHECK: [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[CONV_2]], [[SLICE_ARG2]])  
    // CHECK: [[CONV_3:%.+]] = VPU.NCE.Convolution([[ELTWISE_1]], %cst_6, %cst_10)  
    // CHECK: [[CONV_4:%.+]] = VPU.NCE.Convolution([[CONV_3]], %cst_7, %cst_11)  
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[ELTWISE_1]] [0, 0, 0, 0] [1, 64, 136, 49] 
    // CHECK: [[ELTWISE_2:%.+]] = VPU.NCE.Eltwise([[CONV_4]], [[SLICE_1]]) 
    // CHECK: [[GROUP_SPARSE_1:%.+]] = VPU.GroupSparseTensor([[ELTWISE_2]]
    // CHECK: [[CONV_5:%.+]] = VPU.NCE.Convolution([[GROUP_SPARSE_1]], %cst_8, %cst)
    // CHECK: [[SLICE_ARG1:%.+]] = VPU.Slice %arg1 [0, 0, 0, 0] [1, 32, 272, 97] 
    // CHECK: [[ELTWISE_3:%.+]] = VPU.NCE.Eltwise([[CONV_5]], [[SLICE_ARG1]]) 
    // CHECK: VPU.NCE.Convolution([[ELTWISE_3]], %cst_9, %cst_10) 
        
}
