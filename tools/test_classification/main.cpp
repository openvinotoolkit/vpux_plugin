// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <condition_variable>
#include <mutex>
#include <algorithm>

#include "openvino/openvino.hpp"

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>

#include <sys/stat.h>

#include "test_classification.hpp"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // -------- Parsing and validation of input arguments --------
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    const std::vector<std::string> allowedPrecision = {"u8", "f16", "f32"};
    if (!FLAGS_ip.empty()) {
        // input precision is u8, f16 or f32 only
        if (std::find(allowedPrecision.cbegin(), allowedPrecision.cend(), FLAGS_ip) == allowedPrecision.cend())
            throw std::logic_error("Parameter -ip " + FLAGS_ip + " is not supported");
    }

    if (!FLAGS_op.empty()) {
        // output precision is u8, f16 or f32 only
        if (std::find(allowedPrecision.cbegin(), allowedPrecision.cend(), FLAGS_op) == allowedPrecision.cend())
            throw std::logic_error("Parameter -op " + FLAGS_op + " is not supported");
    }

    return true;
}

template <typename T> void writeToFile(const std::vector<T>& input, const std::string& dst) {
    std::ofstream dumper(dst, std::ios_base::binary);
    if (dumper.good() && !input.empty()) {
        dumper.write(reinterpret_cast<const char*>(&input[0]), input.size() * sizeof(T));
    }
    dumper.close();
}

void dumpTensor(const ov::runtime::Tensor& inputTensor, const std::string& dst) {
    std::ofstream dumper(dst, std::ios_base::binary);
    if (dumper.good()) {
        const auto* inputRaw = inputTensor.data();
        dumper.write(reinterpret_cast<const char*>(inputRaw), inputTensor.get_byte_size());
    }
    dumper.close();
}

template <class T_data>
std::vector<T_data> generateSequence(std::size_t dataSize, ov::element::Type dtype) {
    std::vector<T_data> result(dataSize);
    if (dtype == ov::element::u8) {
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<T_data>(i);
    } else if (dtype == ov::element::f32) {
        float LO = -10, HI = 10;
        float nummax = RAND_MAX;
        for (std::size_t i = 0; i < result.size(); ++i)
             result[i] = LO + (static_cast<float>(std::rand()) /(nummax/(HI-LO)));
    }
    return result;
}

int main(int argc, char *argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        
        const std::string model_path = FLAGS_m;
        const std::string image_path = FLAGS_i;

        // -------- 1. Initialize OpenVINO Runtime Core --------
        slog::info << "Creating Inference Engine" << slog::endl;
        ov::runtime::Core core;

        // Print device version
        slog::info << core.get_versions("VPUX") << slog::endl;

        // -------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) --------
        slog::info << "Loading network files" << slog::endl;

        // Read network model
        auto model = core.read_model(model_path);
        printInputAndOutputsInfo(*model);

        // -------- 3. Set up input --------
        slog::info << "Preparing input blobs" << slog::endl;

        // Taking information about all topology inputs
        auto inputs = model->inputs();
        bool multiInput = false;
        if (inputs.size() > 1){
            multiInput = true;
            slog::info << "\t multiple inputs detected" << slog::endl;
            slog::info << "\t inputs feeding order: ";
            for (auto& input : inputs) {
                slog::info << input.get_any_name() << ",";
            }
            slog::info << slog::endl;
        }

        // input precision
        ov::element::Type inPrecision;
        std::vector<ov::Shape> shapes;
        std::vector<size_t> totalSizes;
        
        if (FLAGS_ip == "f16") {
            inPrecision = ov::element::f16;
        } else if (FLAGS_ip == "f32") {
            inPrecision = ov::element::f32;
        } else {
            inPrecision = ov::element::u8;
        }

        // apply preprocessing
        ov::preprocess::PrePostProcessor preproc(model);
        preproc.input().tensor().set_element_type(inPrecision);

        // set input layout
        ov::Layout input_layout;
        for (auto& input : inputs) {
            ov::Shape input_shape = input.get_shape();
            if (input_shape.size() == 4)
                input_layout = "NHWC";
            else if (input_shape.size() == 1)
                input_layout = "C";
            else
                input_layout = "NC";

            shapes.push_back(input_shape);
            totalSizes.push_back(std::accumulate(input_shape.cbegin(), input_shape.cend(), 1,
                                 std::multiplies<size_t>()));
        }

        preproc.input().network().set_layout(input_layout);

        // validImageNames is needed for classificationresult function. keeping it for now
        std::vector<std::string> validImageNames = {};
        std::vector<std::shared_ptr<unsigned char>> imagesData = {};
        std::vector<std::vector<uint8_t>> inputSeqs_u8;
        bool inputChannelMajor = false;

        // parsing the inputs
        std::vector<std::string> inputNames;
        if (!multiInput) inputNames.push_back(FLAGS_i);
        else {
            std::stringstream inputNameStream(FLAGS_i);
            while(inputNameStream.good()) {
                std::string inputNameItem;
                getline(inputNameStream, inputNameItem, ',');
                inputNames.push_back(inputNameItem);
            }
            if (inputNames.size() != inputs.size())
                throw std::logic_error("Inputs number doesn't match the required inputs.");
        }

        // generate an input or use provided
        if (FLAGS_i.empty()) {
            // auto generate an input
            slog::info << "No image provided, generating a random input..." << slog::endl;
            validImageNames.push_back("Autogenerated");
            for (int index = 0; index < totalSizes.size(); index++) {
                inputSeqs_u8.push_back((generateSequence<uint8_t>(totalSizes[index], ov::element::u8)));
            }
        } else {
            // Only consider the first image input.
            validImageNames.push_back(inputNames[0]);
            for (int inputIndex = 0; inputIndex < inputNames.size(); inputIndex++) {
                std::string inputName = inputNames[inputIndex];
                size_t totalSize = totalSizes[inputIndex];
                std::vector<size_t> inputDims = shapes[inputIndex];
                std::vector<uint8_t> inputSeq_u8;
                if (fileExt(inputName).compare("dat") == 0 || fileExt(inputName).compare("bin") == 0) {
                    // use a binary input.dat or .bin
                    slog::info << "Using provided binary input..." << slog::endl;
                    std::ifstream file(inputName, std::ios::in | std::ios::binary);
                    if (!file.is_open())
                        throw std::logic_error("Input: " + inputName + " cannot be read!");
                    file.seekg(0, std::ios::end);
                    size_t total = file.tellg() / sizeof(uint8_t);
                    if (inPrecision == ov::element::f32)
                        total = file.tellg() / sizeof(float);
                    if (total != totalSize) {
                        // number of entries doesn't match, either not U8 or from different network
                        throw std::logic_error("Input contains " + std::to_string(total) + " entries," +
                            "which doesn't match expected dimensions: " + std::to_string(totalSize));
                    }
                    file.seekg(0, std::ios::beg);
                    inputSeq_u8.resize(total);
                    file.read(reinterpret_cast<char *>(&inputSeq_u8[0]), total * sizeof(uint8_t));
                    inputChannelMajor = inputDims.size()==4 ? true : false;

                } else {
                    // use a provided image
                    slog::info << "Using provided image..." << slog::endl;
                    FormatReader::ReaderPtr reader(inputName.c_str());
                    if (reader.get() == nullptr) {
                        throw std::logic_error("Image: " + inputName + " cannot be read!");
                    }

                    /** Store image data **/
                    std::shared_ptr<unsigned char> data(
                            reader->getData(inputDims[3],inputDims[2]));
                    if (data != nullptr) {
                        imagesData.push_back(data);
                    }
                    // store input in vector for processing later
                    for (size_t i = 0; i < totalSize; ++i) {
                        inputSeq_u8.push_back(static_cast<uint8_t>(imagesData.at(0).get()[i]));
                    }
                }
                inputSeqs_u8.push_back(inputSeq_u8);
            }
        }
        
        // Output precision
        ov::element::Type outPrecision = ov::element::u8;
        auto outputs = model->outputs();
        if (!FLAGS_op.empty()) {
            if (FLAGS_op == "f16") outPrecision = ov::element::f16;
            else if (FLAGS_op == "f32") outPrecision = ov::element::f32;
        }
        
        preproc.output().tensor().set_element_type(outPrecision);
        model = preproc.build();

        // /** Setting batch size using image count **/
        size_t batchSize = 1;
        slog::info << "Batch size is " << batchSize << slog::endl;

        // -------- 4. Loading model to the device --------
        slog::info << "Loading model to the device" << slog::endl;
        ov::runtime::ExecutableNetwork executable_network = core.compile_model(model, "VPUX");

        // -------- 5. Create infer request --------
        ov::runtime::InferRequest infer_request = executable_network.create_infer_request();
        slog::info << "CreateInferRequest completed successfully" << slog::endl;

        // -------- 6. Prepare input --------
        std::vector<ov::runtime::Tensor> input_tensors;
        std::vector<std::size_t> input_tensors_idx;
        auto item = inputs.begin();

        for (int inputIndex = 0; inputIndex < inputs.size(); inputIndex++) {
            auto& input = inputs[inputIndex];
            const auto& inputTensorDesc = input.get_tensor();
            const auto inputShape = inputTensorDesc.get_shape();
            auto inputTensor = ov::runtime::Tensor(inPrecision, inputShape);

            size_t totalSize = totalSizes[inputIndex];
            /** Fill input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = 1;
            size_t image_size = 1;
            if (inputShape.size() == 4) {  // image input
                num_channels = inputShape[1];
                image_size = inputShape[3] * inputShape[2];
            }
            else {  // unknown kind of input
                num_channels = 1;
                image_size = totalSize;
            }
            // ASSUMPTION inputSeq_u8 is Z-Major (NHWC), BGR
            std::vector<uint8_t> input_nchw_rgb(num_channels * image_size);
            std::vector<uint8_t> input_nchw_bgr(num_channels * image_size);
            std::vector<uint8_t> input_nhwc_rgb(num_channels * image_size);
            std::vector<uint8_t> input_nhwc_bgr(num_channels * image_size);

            // write to the buffer if the input is not image format
            // For non image input, only supportU8 precision
            if (inputShape.size() != 4) {
                auto* data = inputTensor.data<uint8_t>();
                auto inputSeq_u8 = inputSeqs_u8[inputIndex];
                if (data == nullptr) {
                    throw std::logic_error("input blob buffer is null");
                }
                for(size_t pid = 0; pid < inputSeq_u8.size(); pid++)
                    data[pid] = inputSeq_u8[pid];
                item++;
                continue;
            }

            if (inPrecision == ov::element::u8) {
                auto* data = inputTensor.data<uint8_t>();
                auto inputSeq_u8 = inputSeqs_u8[inputIndex];
                if (data == nullptr) {
                    throw std::logic_error("input blob buffer is null");
                }
                if(inputChannelMajor) // Input is Channel Major, BGR
                {
                    for (size_t i = 0; i < inputSeq_u8.size(); ++i){
                        if(!FLAGS_r) // Keep Channel Major BGR
                            data[i] = inputSeq_u8[i];
                        input_nchw_bgr[i] = inputSeq_u8[i];
                        // Channel Major RGB
                        if(i < image_size) { // B
                            if(FLAGS_r)
                                data[i] = inputSeq_u8.at(i+(image_size*2));
                            input_nchw_rgb[i] = inputSeq_u8.at(i+(image_size*2));
                        }
                        else if ( i > image_size * 2) { // R
                            if(FLAGS_r)
                                data[i+(image_size*2)] = inputSeq_u8.at(i);
                            input_nchw_rgb[i+(image_size*2)] = inputSeq_u8.at(i);
                        }
                        else { // G
                            if(FLAGS_r)
                                data[i] = inputSeq_u8.at(i);
                            input_nchw_rgb[i] = inputSeq_u8.at(i);
                        }
                    }
                    //Create Z-Major BGR for KMB (may be required later)
                    for (size_t ch = 0; ch < num_channels; ++ch)
                    {
                        for (size_t pid = 0; pid < image_size; pid++)
                        {
                            input_nhwc_bgr[pid*num_channels + pid] = inputSeq_u8.at(ch*image_size + pid);
                        }
                    }

                    // Create Z-Major RGB. Use Z-Major BGR, just swap channels into RGB
                    for (size_t i = 1; i <  input_nhwc_bgr.size(); i=i+3)
                    {
                        input_nhwc_rgb[i-1] =  input_nhwc_bgr.at(i+1); // R <- B
                        input_nhwc_rgb[i] =  input_nhwc_bgr.at(i); // G
                        input_nhwc_rgb[i+1] = input_nhwc_bgr.at(i-1); // B <- R
                    }
                }
                else // Input is Z-Major, BGR. From bmp image, for example.
                {
                    for(size_t pid = 0; pid < inputSeq_u8.size(); pid++)
                        input_nhwc_bgr[pid] = inputSeq_u8[pid];
                    for (size_t pid = 0; pid < image_size; pid++) {
                        /** Iterate over all channels to create channel major input **/
                        for (size_t ch = 0; ch < num_channels; ++ch) {
                            int swap_ch = (num_channels-1) - ch;
                            if(!FLAGS_r) // Create channel major, BGR input for CPU Plugin
                                data[ch * image_size + pid] = inputSeq_u8.at(pid*num_channels + ch);
                            input_nchw_bgr[ch * image_size + pid] = inputSeq_u8.at(pid*num_channels + ch);
                            if(FLAGS_r) // Create channel major, RGB input for CPU Plugin
                                data[swap_ch * image_size + pid] = inputSeq_u8.at(pid*num_channels + ch);
                            input_nchw_rgb[swap_ch * image_size + pid] = inputSeq_u8.at(pid*num_channels + ch);
                        }
                    }
                    // Keep Z-major, just swap channels into RGB
                    for (size_t i = 1; i < inputSeq_u8.size(); i=i+3) {
                        input_nhwc_rgb[i-1] = inputSeq_u8.at(i+1); // R <- B
                        input_nhwc_rgb[i] = inputSeq_u8.at(i); // G
                        input_nhwc_rgb[i+1] = inputSeq_u8.at(i-1); // B <- R
                    }
                }
                writeToFile<uint8_t>(input_nhwc_bgr, "./input_cpu_nhwc_bgr.bin");
                writeToFile<uint8_t>(input_nhwc_rgb, "./input_cpu_nhwc_rgb.bin");
                writeToFile<uint8_t>(input_nchw_rgb, "./input_cpu_nchw_rgb.bin");
                writeToFile<uint8_t>(input_nchw_bgr, "./input_cpu_nchw_bgr.bin");
             }
            // TODO Update to create all 4 options for input
            else if (inPrecision == ov::element::f32) {
                std::ifstream file(inputNames[inputIndex], std::ios::in | std::ios::binary);
                if (!file.is_open())
                    throw std::logic_error("Input: " + inputNames[inputIndex] + " cannot be read!");
                file.seekg(0, std::ios::end);
                size_t total = file.tellg() / sizeof(float);
                if (total != totalSize) {
                    // number of entries doesn't match, either not FP32 or from different network
                    throw std::logic_error("Input contains " + std::to_string(total) + " entries, " +
                                           "which doesn't match expected dimensions: " + std::to_string(totalSize));
                }
                file.seekg(0, std::ios::beg);
                std::vector<float> inputSeq_fp32;
                inputSeq_fp32.resize(total);
                file.read(reinterpret_cast<char *>(&inputSeq_fp32[0]), total * sizeof(float));
                auto* data = inputTensor.data<float>();
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        data[ch * image_size + pid] = inputSeq_fp32.at(pid*num_channels + ch);
                        // std::cout << " data[" << (ch * image_size + pid) << "] = " << data[ch * image_size + pid];
                    }
                }
                writeToFile<float>(inputSeq_fp32, "./input_cpu_nhwc_bgr.bin");
                writeToFile<float>(inputSeq_fp32, "./input_cpu_nhwc_rgb.bin");
                writeToFile<float>(inputSeq_fp32, "./input_cpu_nchw_bgr.bin");
                writeToFile<float>(inputSeq_fp32, "./input_cpu_nchw_rgb.bin");
            }

            item++;
            input_tensors.emplace_back(std::move(inputTensor));
            input_tensors_idx.emplace_back(input.get_index());
        }

        for (std::size_t i = 0; i < inputs.size(); i++) {
            infer_request.set_input_tensor(input_tensors_idx[i], input_tensors[i]);
        }

        // -------- 7. Do inference --------
        size_t numIterations = 1;
        size_t curIteration = 0;
        std::condition_variable condVar;

        infer_request.set_callback(
            [&](std::exception_ptr) {
                curIteration++;
                slog::info << "Completed " << curIteration << " async request execution" << slog::endl;
                if (curIteration < numIterations) {
                    /* here a user can read output containing inference results and put new input
                       to repeat async request again */
                    infer_request.start_async();
                } else {
                    /* continue sample execution after last Asynchronous inference request execution */
                    condVar.notify_one();
                }
            }
        );

        /* Start async request for the first time */
        slog::info << "Start inference (" << numIterations << " asynchronous executions)" << slog::endl;
        infer_request.start_async();

        /* Wait all repetitions of the async request */
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [&]{ return curIteration == numIterations; });

        // -------- 8. Process output --------
        slog::info << "Processing " << outputs.size() << " output blob" << ((outputs.size() > 1) ? "s" : "") << slog::endl;

        /** Read labels from file (e.x. AlexNet.labels) **/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;

        std::ifstream inputFile;
        inputFile.open(labelFileName, std::ios::in);
        if (inputFile.is_open()) {
            std::string strLine;
            while (std::getline(inputFile, strLine)) {
                trim(strLine);
                labels.push_back(strLine);
            }
        }

        std::vector<ov::runtime::Tensor> outputTensors;
        for (auto output : outputs) {
            const auto index = output.get_index();
            outputTensors.emplace_back(infer_request.get_output_tensor(index));
        }

        for (unsigned i = 0 ; i < outputTensors.size(); i++)
        {
            /** Validating -nt value **/
            const size_t resultsCnt = outputTensors[i].get_size() / batchSize;
            if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
                slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " \
                          << resultsCnt+1 << " and more than 0)\n            will be used maximal value : " << resultsCnt << slog::endl;
                FLAGS_nt = resultsCnt;
            }

            // save the results file for validation
            if (i == 0) dumpTensor(outputTensors[i], "./output_cpu.bin"); //TODO: remove when validator updated
            dumpTensor(outputTensors[i], "./output_cpu" + std::to_string(i) + ".bin");
            ClassificationResult classificationResult(outputTensors[i], validImageNames,
                                                      batchSize, FLAGS_nt,
                                                      labels);
            if (outputTensors.size() > 1)
                std::cout << "Output " << i << ":" << std::endl;
            classificationResult.show();

            std::string results_filename;
            if (outputTensors.size() == 1)
                results_filename = "./inference_results.txt";
            else
                results_filename = "./inference_results" + std::to_string(i) + ".txt";
            std::ofstream f(results_filename);
            auto topK = classificationResult.getResults();
            for(auto i = topK.begin(); i != topK.end(); ++i) {
                f << *i << '\n';
            }
        }
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl << "This sample is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool" << slog::endl;
    return 0;
}

