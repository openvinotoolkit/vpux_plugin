//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

// clang-format off

#include "ngraph_mcm_frontend/frontend.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include "ngraph_mcm_frontend/passes/add_io_convert_ops.hpp"
#include "ngraph_mcm_frontend/passes/convert_to_mcm_conv.hpp"
#include "ngraph_mcm_frontend/passes/convert_to_mcm_model.hpp"
#include "ngraph_mcm_frontend/passes/convert_to_mcm_fc.hpp"
#include "ngraph_mcm_frontend/passes/merge_result_convert.hpp"
#include "ngraph_mcm_frontend/passes/replace_add_with_eltwise.hpp"
#include "ngraph_mcm_frontend/passes/replace_scaleshift_with_mcm_scale.hpp"
#include "ngraph_mcm_frontend/passes/align_eltwise_scales.hpp"
#include "ngraph_mcm_frontend/passes/align_concat_scales.hpp"
#include <file_utils.h>
#include <vpu/utils/logger.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/convert_opset1_to_legacy/convert_mul_add_to_scaleshift_or_power.hpp>
#include <transformations/convert_opset1_to_legacy/convert_mul_or_add_finally.hpp>
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp>
#include <transformations/convert_reduce_to_pooling.hpp>
#include <transformations/lin_op_sequence_fusoin.hpp>
#include <transformations/convert_opset1_to_legacy/convert_convolutions.hpp>
#include <transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp>
#include <transformations/convert_opset1_to_legacy/convert_prelu_to_relu_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <transformations/convert_opset1_to_legacy/convert_power_to_power_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp>
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>


#include "transformations/common_optimizations/algebraic_simplification.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/depth_to_space_fusion.hpp"
#include "transformations/optimize_strided_slice.hpp"
#include "transformations/convert_scatter_elements_to_scatter.hpp"
#include "transformations/convert_pad_to_group_conv.hpp"
#include "transformations/remove_filtering_boxes_by_size.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/mish_fusion.hpp"
#include "transformations/softplus_fusion.hpp"
#include "transformations/softplus_to_mish_fusion.hpp"
#include "transformations/swish_fusion.hpp"
#include "transformations/hswish_fusion.hpp"
#include "transformations/normalize_l2_fusion.hpp"
#include "transformations/convert_quantize_dequantize.hpp"
#include "transformations/bidirectional_sequences_decomposition.hpp"
#include <generic_ie.hpp>

#include <include/mcm/compiler/compilation_unit.hpp>
#include <memory>
#include <string>
#include <vector>
#include <utility>

std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReference(
    const std::string& tensorName,
    const InferenceEngine::TensorDesc& tensorInfo);

namespace {
    std::map<std::string, std::string> MapInputOutputInfoToNgraphOps(const std::shared_ptr<ngraph::Function>& func,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo) {
        // Due to historical reasons, ICNNNetwork::getOutputsInfo() does not match excatly
        // to ngraph::op::v0::Result::get_friendly_name(), actual get_friendly_name() may be have arbitary different.
        // Instead getOutputsInfo() returns names of nodes, who produces input to ngraph::op::v0::Result,
        // This expected to be fixed in 2021.2
        // Below ngraph function is changed and Result producers are replaced, making impossible to match.
        // Therefore Ngraph Results must be mached to OutputsInfo Here.
        // There is no API to extract actual mapping from CNNNetwork
        // See how results are converted to outputInfo in convert_function_to_cnn_network.cpp
        std::map<std::string, std::string> ioMap;
        // TBD Do we need inputs too?
        for (const auto& inputInfo : inputsInfo) {
            bool isFound = false;
            for (auto&& paramOp : func->get_parameters()) {
                IE_ASSERT(1 == paramOp->get_output_size());
                auto name = paramOp->output(0).get_tensor().get_name();
                if (name.empty())
                    name = ngraph::op::util::create_ie_output_name(paramOp->output(0));
                if (name == inputInfo.first) {
                    ioMap[inputInfo.first] = paramOp->get_friendly_name();
                    isFound = true;
                    break;
                }
            }
            if (!isFound)
                THROW_IE_EXCEPTION << "Input not found: " << inputInfo.first;
        }

        for (const auto& outputInfo : outputsInfo) {
            bool isFound = false;
            for (auto&& resultOp : func->get_results()) {
                IE_ASSERT(1 == resultOp->get_input_size());
                const auto &input = resultOp->input_value(0);
                auto name = input.get_tensor().get_name();
                if (name.empty())
                    name = ngraph::op::util::create_ie_output_name(input);
                if (name == outputInfo.first) {
                    ioMap[outputInfo.first] = resultOp->get_friendly_name();
                    isFound = true;
                    break;
                }
            }
            if (!isFound)
                THROW_IE_EXCEPTION << "Ouput not found: " << outputInfo.first;
        }

        return ioMap;
    }
}

std::vector<char> compileNGraph(
        const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo,
        const vpu::MCMConfig& config) {
    const auto log = std::make_shared<vpu::Logger>("KMB nGraph Parser", config.logLevel(), vpu::consoleOutput());

    log->info("Parse nGraph %v", netName);

    //
    // Configure MCM Compiler
    //

    mv::CompilationUnit mcmCompiler(netName);

    {
        log->debug("Configure MCM Compiler");
        VPU_LOGGER_SECTION(log);

        bool layoutNCHW = true;
        auto compDescName = config.mcmCompilationDesciptor();
        for (const auto& netInput : inputsInfo) {
            if (netInput.second->getLayout() != InferenceEngine::Layout::NCHW) {
                layoutNCHW = false;
                break;
            }
        }
        if (layoutNCHW) {
            compDescName = "release_kmb_with_CM_Conv";
        }

        const auto targetPath = ie::getIELibraryPath() + "/" + config.mcmTargetDesciptorPath() + "/" + config.mcmTargetDesciptor() + ".json";
        const auto compDescPath = ie::getIELibraryPath() + "/" + config.mcmCompilationDesciptorPath() + "/" + compDescName + ".json";

        IE_ASSERT(mcmCompiler.loadTargetDescriptor(targetPath));
        IE_ASSERT(mcmCompiler.loadCompilationDescriptor(compDescPath));

        auto& mcmCompDesc = mcmCompiler.compilationDescriptor();

        mcmCompDesc.setPassArg("GlobalConfigParams", "verbose", cvtLogLevelToMCM(config.mcmLogLevel()));

        if (config.referenceMode()) {
            mcmCompDesc.setPassArg("GlobalConfigParams", "ReferenceMode", true);
        }

        std::function<void(MVCNN::GraphFileT&)> metaInfoSerializer =
            [&inputsInfo, &outputsInfo](MVCNN::GraphFileT& graphFileInstance) {
            if (graphFileInstance.header == nullptr) {
                THROW_IE_EXCEPTION << "metaInfoSerializer: graph file header points to null";
            }

            for (const auto& inInfo : inputsInfo) {
                graphFileInstance.header->in_tensor_desc.push_back(
                    buildTensorReference(inInfo.first, inInfo.second->getTensorDesc()));
            }

            for (const auto& outInfo : outputsInfo) {
                graphFileInstance.header->out_tensor_desc.push_back(
                    buildTensorReference(outInfo.first, outInfo.second->getTensorDesc()));
            }
        };
        mcmCompDesc.setPassArg("GenerateBlobKmb", "metaInfoSerializer", metaInfoSerializer);

        IE_ASSERT(mcmCompiler.initialize());
    }

    //
    // Convert nGraph to MCM Model
    //

    {
        log->debug("Convert nGraph to MCM Model");

        auto& mcmModel = mcmCompiler.model();
        NodeOutputToMcmMap mcmOutputsMap;

        ngraph::pass::Manager passManager;
        passManager.register_pass<ngraph::pass::ConvertPriorBox>();
        passManager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
        passManager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
        passManager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();

        passManager.register_pass<ngraph::pass::ConstantFolding>();
        passManager.register_pass<ngraph::pass::ConvertConvolutions>();
        passManager.register_pass<ngraph::pass::LinOpSequenceFusion>();
        passManager.register_pass<ngraph::pass::ConvertMatMulToFCorGemm>();
        passManager.register_pass<ngraph::pass::ConvFusion>();
        passManager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
        passManager.register_pass<ngraph::pass::ConvertMulAddToScaleShiftOrPower>();
        passManager.register_pass<ngraph::pass::ConvertMulOrAddFinally>();
        passManager.register_pass<ngraph::pass::ConvertReduceToPooling>();
        passManager.register_pass<ngraph::pass::ConvertPReLUToReLUIE>();
        passManager.register_pass<ngraph::pass::ConvertInterpolateToInterpOrResampleMatcher>();

        passManager.register_pass<ngraph::pass::ConvertPowerToPowerIEMatcher>();
        passManager.register_pass<ngraph::pass::ConstantFolding>();

        // TBD Should be ngraph::pass too in order to be applied in between other passes.
        const auto ioMap = MapInputOutputInfoToNgraphOps(func, inputsInfo, outputsInfo);


        passManager.register_pass<ConvertToMcmConv>();
        passManager.register_pass<ConvertToMcmFC>();
        passManager.register_pass<ReplaceScaleShiftWithMcmScale>();
        passManager.register_pass<ReplaceAddWithMcmEltwise>();
        passManager.register_pass<AlignEltwiseScales>();
        passManager.register_pass<AlignConcatScales>();
        passManager.register_pass<ConvertToMcmModel>(mcmModel, mcmOutputsMap, inputsInfo, outputsInfo, ioMap);

        const auto start = std::chrono::high_resolution_clock::now();
        passManager.run_passes(func);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto process_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
        log->info("Plugin processing time: %v ms", process_time.count());
    }

    //
    // Run MCM Compiler
    //

    {
        log->debug("Run MCM Compiler");
        try {
            const auto start = std::chrono::high_resolution_clock::now();
            mcmCompiler.run();
            const auto end = std::chrono::high_resolution_clock::now();
            const auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
            log->info("Compiler processing time: %v ms", compile_time.count());
        } catch (std::string& str) {
            log->error("MCM Compiler error: %v", str);
            throw std::logic_error(str);
        } catch (std::exception& ex) {
            log->error("MCM Compiler exception: %v", ex.what());
            throw;
        } catch (...) {
            log->error("MCM Compiler general exception");
            throw;
        }
    }

    //
    // Return compiled blob
    //

    const auto memBlob = mcmCompiler.getBlob();
    IE_ASSERT(memBlob != nullptr);
    std::vector<char> blob;
    std::copy(memBlob->begin(), memBlob->end(), std::back_inserter(blob));

    if (blob.empty()) {
        THROW_IE_EXCEPTION << "Blob created by mcmCompiler is empty!";
    }

    return blob;
}

// clang-format on
