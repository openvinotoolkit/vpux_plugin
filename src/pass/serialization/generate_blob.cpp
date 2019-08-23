#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/target/myriadx/nce1_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include <cmath>
#include <numeric>
#include <cmath>

static void generateBlobFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void PopulateSerialFieldsFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&);
//static void writeSerialFieldsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput);

namespace mv
{

    namespace pass
    {


        MV_REGISTER_PASS(PopulateSerialFields)
        .setFunc(PopulateSerialFieldsFcn)
        .setDescription(
            "Gathers fields for serialization"
        );

        MV_REGISTER_PASS(GenerateBlob)
        .setFunc(generateBlobFcn)
        .defineArg(json::JSONType::Bool, "enableFileOutput")
        .defineArg(json::JSONType::Bool, "enableRAMOutput")
        .setDescription(
            "Generates an executable blob file"
        );

    }

}

void generateBlobFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element& compOutput)
{   

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    mv::ControlModel cm(model);
    mv::Serializer serializer(mv::mvblob_mode);

    // Set output parameters for this serialization from config JSON object
    // note: defaults from cm.RuntimeBinary constructor are disableRam , enableFile ,  mcmCompile.blob
    bool RAMEnable = false ;
    bool fileEnable = false ;
    std::string blobFileName = "mcmCompile.blob"; 
 
    if (passDesc.hasAttr("enableRAMOutput") && passDesc.get("enableRAMOutput")) {
        RAMEnable = true ;
    }
    cm.getBinaryBuffer()->setRAMEnabled(RAMEnable) ;

    if (passDesc.hasAttr("enableFileOutput") && passDesc.get("enableFileOutput")) {
        fileEnable = true ;
    }
    cm.getBinaryBuffer()->setFileEnabled(fileEnable) ;

    if (passDesc.hasAttr("fileName")) {
        blobFileName = passDesc.get<std::string>("fileName");
    }

    cm.getBinaryBuffer()->setFileName(blobFileName) ;

    long long result = static_cast<long long>(serializer.serialize(model, td));
    compOutput.set<int64_t>("blobSize", result);

}

void fillMXDescriptors(mv::ControlModel cm, mv::DataModel dm, unsigned fp16_size, mv::Data::OpListIterator opIt, mv::NCE1HWOps hwOp)
{
    int cmxSize = 256*1024;

    mv::Data::TensorIterator input;
    mv::Data::TensorIterator output;
    mv::Data::TensorIterator taps;

    input = opIt->getInputTensor(0);
    if(hwOp == mv::NCE1HWOps::Convolution)
        taps = opIt->getInputTensor(1);
    output = opIt->getOutputTensor(0);

    auto chPerRamBlock = opIt->get<std::vector<std::size_t>>("NCE1_InputChannelsRamBlock");
    auto bottomJunk = opIt->get<std::vector<size_t>>("NCE1_JunkOutputAfter");
    auto topJunk = opIt->get<std::vector<size_t>>("NCE1_JunkOutputBefore");
    auto localLS = opIt->get<std::size_t>("NCE1_LocalLineStride");
    auto minLines = opIt->get<std::vector<std::size_t>>("NCE1_MinLines");
    auto stride = opIt->get<std::array<unsigned short, 2>>("stride")[0];
    auto padEn = opIt->get<std::array<unsigned short, 4>>("padding")[0];
    auto LPC = opIt->get<std::vector<std::size_t>>("NCE1_LinesPerChannel");
    auto localCS = opIt->get<std::vector<std::size_t>>("NCE1_LocalChannelStride");

    // Get all attrs:
    auto splits_over_H = opIt->get<size_t>("NCE1_SplitsOverHeight");
    auto DPUmodeVector = opIt->get<std::vector<size_t>>("NCE1_Modes");
    auto splits_over_iC = opIt->get<size_t>("NCE1_SplitsOverInputChannels");
    auto inputChannelsPadded = opIt->get<std::size_t>("NCE1_InputChannelsPadded");
    auto outputChannelsPadded = opIt->get<std::size_t>("NCE1_OutputChannelsPadded");
    auto inputWidthPadded = opIt->get<std::size_t>("NCE1_InputWidthPadded");
    auto outputWidthPadded = opIt->get<std::size_t>("NCE1_OutputWidthPadded");
    auto desc_count = opIt->get<std::size_t>("NCE1_DescriptorSplits");
    auto streamingMask = opIt->get<std::size_t>("NCE1_StreamingMask");
    auto outputChannelPerformed = opIt->get<std::vector<std::size_t>>("NCE1_OutputChannelsPerformed");

    auto input_lines_processed = opIt->get<std::vector<size_t>>("NCE1_InputLinesProcessed");
    auto output_lines_processed = opIt->get<std::vector<size_t>>("NCE1_OutputLinesProcessed");
    auto output_line_start = opIt->get<std::vector<size_t>>("NCE1_StartOutputLine");
    auto input_line_start = opIt->get<std::vector<size_t>>("NCE1_StartInputLine");

    unsigned radixX, radixY;
    if(hwOp == mv::NCE1HWOps::Convolution)
    {
        //ASSUMING MX WEIGHT SHAPE
        radixX = taps->getShape()[2];
        radixY = taps->getShape()[3];
    }
    else if(hwOp == mv::NCE1HWOps::FullyConnected)
    {
        //TODO
    }
    else
    {
        radixX = opIt->get<std::array<unsigned short, 2>>("kSize")[0];
        radixY = opIt->get<std::array<unsigned short, 2>>("kSize")[1];
    }

// below code adds IDs for conv/pooling and FC
    if (hwOp == mv::NCE1HWOps::Convolution)
        opIt->set<unsigned>("SerialID", 33);
    else if ((hwOp == mv::NCE1HWOps::AveragePooling) || (hwOp == mv::NCE1HWOps::MaxPooling))
        opIt->set<unsigned>("SerialID", 34);
    else if (hwOp == mv::NCE1HWOps::FullyConnected)
        opIt->set<unsigned>("SerialID", 33);


    opIt->set<unsigned>("streamingMask", streamingMask );

    std::size_t total_size = input->getShape().totalSize();
    total_size *= inputChannelsPadded;
    total_size /= input->getShape()[2];
    opIt->set<unsigned>("inputSize", total_size*fp16_size);

    opIt->set<unsigned>("outputSize",
        output->getShape().totalSize()*fp16_size);

    opIt->set<unsigned>("concatOffset", 0); // Not Supported...
    opIt->set<unsigned>("unloadCMX", 0); // Not Supported...
    opIt->set<unsigned>("overwriteInput", 0); // Not Supported...
    opIt->set<unsigned>("CMXSize", cmxSize);  // Magic Number...
    opIt->set<unsigned>("reluSHVAcc", 0); // Not Supported...
    if (splits_over_iC > 1)
        if(opIt->hasAttr("postOpType") && opIt->get<std::string>("postOpType") == "ReLu")
            opIt->set<unsigned>("reluSHVAcc", 1);
    opIt->set<unsigned>("shvNegSlope", 0); // Not Supported...
    opIt->set<unsigned>("shvPosSlope", 1065353216); // Magic Number...
       // New fields in blob format in release_R6_181129_1401 vs wddm_vpu_pre_alpha or cpp_api_NORELUELTFUSION
    opIt->set<unsigned>("reluXOnShaveAccumulation", 0); // Not Supported...
    opIt->set<unsigned>("x_of_relux", 0); // Not Supported...
    opIt->set<unsigned>("desc_count", desc_count);

    std::vector<unsigned> desc;
    std::vector<cnnConvolutionPoolStructure> descriptors = std::vector<cnnConvolutionPoolStructure>(desc_count);

    int i = -1;
    for (unsigned h = 0; h < splits_over_H; ++h)
    {
        for (unsigned ic = 0; ic < splits_over_iC; ++ic)
        {
            for (unsigned oc = 0; oc < DPUmodeVector.size(); ++oc)
            {
                //BCONV CTOR
                ++i;

                auto output_channels_performed_so_far = std::accumulate(outputChannelPerformed.begin(), outputChannelPerformed.begin()+oc, 0);

                // Relations to other Descriptors
                if (i+1 == (int)desc_count)
                    descriptors[i].Line0.linkAddress = 0; // Last.
                else
                    descriptors[i].Line0.linkAddress = 32*4*(oc+1);

                descriptors[i].Line0.id = 0;

                // Layer Meta Information - Layout & DataTypes
                if(hwOp == mv::NCE1HWOps::Convolution)
                    descriptors[i].Line0.type = NCE1_CONV;
                else if(hwOp == mv::NCE1HWOps::FullyConnected)
                    descriptors[i].Line0.type = NCE1_FCL;
                else
                    descriptors[i].Line0.type = NCE1_POOL;

                if( input->getOrder().isRowInterleaved())
                    descriptors[i].Line0.interleavedInput = 1;
                else
                    descriptors[i].Line0.interleavedInput = 0;

                if( output->getOrder().isRowInterleaved()){
                    descriptors[i].Line0.interleavedOutput = 1;
                    descriptors[i].rsvd3_interleaved = 1;
                }
                else
                    descriptors[i].Line0.interleavedOutput = 0;

                descriptors[i].Line0.cm = NCE1_DTYPE_FP16;
                descriptors[i].Line0.dm = NCE1_DTYPE_FP16;

                if(hwOp == mv::NCE1HWOps::Convolution)
                {
                    descriptors[i].kernelWidth = radixX - 1;
                    descriptors[i].kernelHeight = radixY - 1;
                }
                else
                {
                    descriptors[i].kernelWidth = 0;
                    descriptors[i].kernelHeight = 0;
                }
                descriptors[i].chStride = stride -1;  // Stride of Kernel (Square only)

                if (padEn > 0)
                    descriptors[i].padEn = 1;
                else
                    descriptors[i].padEn = 0;

                if(hwOp == mv::NCE1HWOps::Convolution)
                    descriptors[i].padType = 0;   // Zero Padding
                else if(hwOp == mv::NCE1HWOps::FullyConnected)
                    descriptors[i].padType = 0;   // Zero Padding
                if ((hwOp == mv::NCE1HWOps::AveragePooling) || (hwOp == mv::NCE1HWOps::MaxPooling)) 
		{
                    auto inputSize = input->getShape();
                    auto outputSize = output->getShape();
                    int kernelDimX = radixX;
                    int kernelDimY = radixY;
                    auto kernelStride = stride;
                    int valid_out_x = std::ceil((float)(inputSize[0] - kernelDimX + 1) / (float)kernelStride);
                    int valid_out_y = std::ceil((float)(inputSize[1] - kernelDimY + 1) / (float)kernelStride);
                    int pad_along_x = (outputSize[0] - 1) * kernelStride + kernelDimX - inputSize[0];
                    int pad_along_y = (outputSize[1] - 1) * kernelStride + kernelDimY - inputSize[1];

                    auto padEnable = (outputSize[0] != valid_out_x || outputSize[1] != valid_out_y);
                    auto pad_left = pad_along_x / 2;
                    auto pad_right = pad_along_x - pad_left;
                    auto pad_top = pad_along_y / 2;
                    auto pad_bottom = pad_along_y - pad_top;
                    descriptors[i].padEn = padEnable;
                    if (padEnable) {
                        descriptors[i].padType = (pad_left > 0) ? 0x08 : 0x00;
                        descriptors[i].padType |= (pad_right > 0) ? 0x01 : 0x00;
                        descriptors[i].padType |= (pad_top > 0) ? 0x04 : 0x00;
                        descriptors[i].padType |= (pad_bottom > 0) ? 0x02 : 0x00;
                    }
                }


                descriptors[i].inputWidth = input->getShape()[0] -1;

                unsigned int current_height;
       
                current_height = input_lines_processed[h];

                descriptors[i].inputHeight =  current_height - 1;

                if(hwOp == mv::NCE1HWOps::Convolution)
                {
                    descriptors[i].outputChannels = outputChannelPerformed[oc] - 1;
                    descriptors[i].inputChannels = inputChannelsPadded -1;
                }
                else if(hwOp == mv::NCE1HWOps::FullyConnected)
                {

                }
                else
                {
                    descriptors[i].outputChannels = outputChannelPerformed[oc] - 1;
                    descriptors[i].inputChannels =  descriptors[i].outputChannels;
                }

                // Myriad X DPU Assignment & Execution Configuration

                descriptors[i].Line0.mode = DPUmodeVector[oc];
                descriptors[i].Line0.it = 0;  // Interrupt Trigger
                descriptors[i].Line0.disInt = 0;  // 0 - Interrupts Enabled, 1 - Interrupts disabled.
                descriptors[i].chPerRamBlock = chPerRamBlock[oc] - 1;        // Input Channels per Ram Block


                // Myriad X Compensation Fields
                descriptors[i].topOutputJunk = topJunk[h];
                descriptors[i].bottomOutputJunk = bottomJunk[h];

 
 // Calculate localLineStride, channel stride, lines per ch 
                // TODO: verify this works for all cases
                auto inputDimX = input->getShape()[0];
                auto inputDimY = input_lines_processed[h];
                auto inputDimZ = inputChannelsPadded;
                // TODO: verify this works for all cases
                if ((hwOp == mv::NCE1HWOps::AveragePooling) || (hwOp == mv::NCE1HWOps::MaxPooling)) {
                    inputDimZ = outputChannelPerformed[i];
                }
                auto noOfBlocks = 1 << DPUmodeVector[oc];
                auto sizeOfBlock = (128 * 1024) >> DPUmodeVector[oc];
                auto bytesPerPixel = fp16_size;
                auto pixelsPerCMXLine = 128 / (bytesPerPixel * 8);

                auto localLineStride = (inputDimX + (pixelsPerCMXLine - 1)) / pixelsPerCMXLine;
                descriptors[i].localLs = localLineStride;


                // Calculate localChanStride
                auto chanPerBlock = inputDimZ / noOfBlocks;
                if (chanPerBlock == 0)
                    chanPerBlock = 1;
                auto availableBytesPerChan = sizeOfBlock / chanPerBlock;
                auto bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel;
                auto linesPerChan = availableBytesPerChan / bytesPerLine;

                if (linesPerChan > inputDimY) {
                    linesPerChan = inputDimY;
                }
                auto localChanStride = linesPerChan * localLineStride;
                descriptors[i].localCs = localChanStride;

                // Calculate linesPerCh
                if (hwOp == mv::NCE1HWOps::Convolution) {
                    //if(self.poolEn == 1)      // not used in WDDM
                    //  minLines = self.kerDimY + self.poolKerDimY
                    //else
                    minLines[oc] = std::min((int)radixY+1,(int)linesPerChan);
                }
                else if ((hwOp == mv::NCE1HWOps::AveragePooling) || (hwOp == mv::NCE1HWOps::MaxPooling)) {
                    if (inputDimX / stride <= 4)
                        minLines[oc] = radixY + 2 * stride + 3;
                    else {
                        minLines[oc] = radixY + 1;
                        minLines[oc] = radixY + stride + 2;
                    }
                    descriptors[i].minLines = minLines[oc];
                }

                descriptors[i].linesPerCh = linesPerChan - 1;
                //---------------------------------------------------------------------------------- code from Python compiler
                descriptors[i].linesPerCh = std::min(LPC[oc] - 1, input_lines_processed[h] - 1);
                descriptors[i].localCs = (descriptors[i].linesPerCh + 1) * descriptors[i].localLs;

     
                // python compiler calculates rud. 
                //---------------------------------------------
                                // Calculate rud
                if ((hwOp == mv::NCE1HWOps::Convolution) || (hwOp == mv::NCE1HWOps::FullyConnected)) {
     
                    auto scheduledInputDimZ = inputChannelsPadded / splits_over_iC;

                       if (((mv::round_up(input->getShape()[0], 8)) * input->getShape()[1] * scheduledInputDimZ * bytesPerPixel) < 128 * 1024) { 
                        auto tileIndex = oc;
                        if (tileIndex == 0) {
                            descriptors[i].rud = 0;
                        }

                        else if (DPUmodeVector[tileIndex - 1] != DPUmodeVector[tileIndex])
                            descriptors[i].rud = 0;
                        else
                            descriptors[i].rud = 1;
                    }
                }

                descriptors[i].minLines = minLines[ic] - 1;     // Minimum lines of data required to carry out function

                descriptors[i].coeffLpb = (descriptors[i].chPerRamBlock+1) * (descriptors[i].kernelWidth+1) * (descriptors[i].kernelHeight+1) - 1;
                descriptors[i].css = (descriptors[i].kernelWidth + 1) * (descriptors[i].kernelHeight + 1) -1 ;
                descriptors[i].outputX = output->getShape()[0];

                // Myriad X - Splitting groups
                descriptors[i].sohGroup = h;
                descriptors[i].sodGroup = ic;

                // Fused ReLU
               
                if(opIt->hasAttr("postOpType") && opIt->get<std::string>("postOpType") == "ReLu" && opIt->get<unsigned>("reluSHVAcc") == 0)
                {
                    descriptors[i].t0 = 0;
                    descriptors[i].a0 = 0;
                    descriptors[i].a1 = 1;
                    descriptors[i].reluxEn = 0;
                    descriptors[i].reluEn = 1;
                }
                else
                {
                    descriptors[i].t0 = 0;
                    descriptors[i].a0 = 0;
                    descriptors[i].a1 = 0;
                    descriptors[i].reluxEn = 0;
                    descriptors[i].reluEn = 0;
                }

                // Fused Pooling, TODO
                if (0)
                {
                    descriptors[i].Line0.type = NCE1_CONV_POOL;
                }
                descriptors[i].avgPoolX = 0;
                if(hwOp == mv::NCE1HWOps::Convolution)
                {
                    descriptors[i].poolType = 0;
                    descriptors[i].poolEn = 0;
                    descriptors[i].poolKernelHeight = 0;
                    descriptors[i].poolKernelWidth = 0;
                }
                else if(hwOp == mv::NCE1HWOps::FullyConnected)
                {
                    //TODO
                }
                else
                {
                    if(hwOp == mv::NCE1HWOps::AveragePooling)
                        descriptors[i].poolType = 1;
                    if(hwOp == mv::NCE1HWOps::MaxPooling)
                        descriptors[i].poolType = 0;
                    descriptors[i].poolEn = 1;
                    descriptors[i].poolKernelHeight = radixY - 1;
                    descriptors[i].poolKernelWidth = radixX - 1;
                }

                // this is needed for wddm but no harm in being here
                                // Calculate Line 7: 1/AV
                if (descriptors[i].reluxEn) {
                    auto x_of_relux = 0;
                    descriptors[i].avgPoolX = x_of_relux;   // Not Supported...;
                }
                else if (descriptors[i].poolEn && descriptors[i].poolType == 1) {
                    float chemicalX = 1.0 / ((float)(radixX * radixY));
                    mv_num_convert cvtr;
                    uint16_t fp16_val = cvtr.fp32_to_fp16(chemicalX);
                    descriptors[i].avgPoolX = fp16_val;
                }
                //------------------------------------------------------

                // Reserved fields of the hw descriptor. Leave as zero or live in eternal fear.
                descriptors[i].Line0.rsvd1 = 0;
                descriptors[i].rsvd2 = 0;
                descriptors[i].rsvd3 = 0;
                descriptors[i].rsvd4 = 0;
                descriptors[i].rsvd5 = 0;
                descriptors[i].rsvd6 = 0;
                descriptors[i].rsvd7 = 0;
                descriptors[i].rsvd9 = 0;
                descriptors[i].rsvd10 = 0;
                descriptors[i].rsvd13 = 0;
                descriptors[i].rsvd8 = 0;

                // Palette for Weights Lookup (Currently Unsupported).
                descriptors[i].p0 = 0;
                descriptors[i].p1 = 0;
                descriptors[i].p2 = 0;
                descriptors[i].p3 = 0;
                descriptors[i].p4 = 0;
                descriptors[i].p5 = 0;
                descriptors[i].p6 = 0;
                descriptors[i].p7 = 0;
                descriptors[i].p8 = 0;
                descriptors[i].p9 = 0;
                descriptors[i].p10 = 0;
                descriptors[i].p11 = 0;
                descriptors[i].p12 = 0;
                descriptors[i].p13 = 0;
                descriptors[i].p14 = 0;
                descriptors[i].p15 = 0;

                auto input_width = inputWidthPadded;
                auto output_channels = outputChannelsPadded;

                auto inputBlobTensor = mv::convertStrides(input, cm, dm);
                auto outputBlobTensor = mv::convertStrides(output, cm, dm);

                /* this->descriptors[i].dataBaseAddr = fp16_size * input_width * output_channels_performed_so_far;    // TODO: Calculate 3f0 (1008)
                        this->descriptors[i].dataBaseAddr += this->input->getShape()[2] * 2 * input_width * this->input_line_start[h] ;    // TODO: Calculate 3f0 (1008)
                        this->descriptors[i].outBaseAddr = fp16_size * outputWidthPadded * output_channels_performed_so_far;  // TODO: Calculate 3f0 (1008)
                        this->descriptors[i].outBaseAddr += outputChannelsPadded * 2 * outputWidthPadded * this->output_line_start[h];    // TODO: Calculate 3f0 (1008)
                 */

                if(hwOp == mv::NCE1HWOps::Convolution)
                {
                    auto inputStride_2 = inputBlobTensor.strideZ;
                    auto inputStartIndex = input_line_start[h];
                    auto inputDimZ = inputChannelsPadded;
                    auto scheduledInputDimZ = inputDimZ;
                    auto sodIndex = descriptors[i].sodGroup;
                    auto split_by_input_sz = sodIndex * scheduledInputDimZ * inputStride_2;
                    auto split_by_height_sz = (scheduledInputDimZ * splits_over_iC) * inputStride_2 * inputStartIndex;
                    std::cout << "split_by_input_sz: " << sodIndex << "," << scheduledInputDimZ << "," << inputStride_2 << std::endl;
                    std::cout << "split_by_height_size: " << scheduledInputDimZ  << "," <<  splits_over_iC << "," <<  inputStride_2  << "," <<  inputStartIndex;
                    auto offset = split_by_input_sz + split_by_height_sz;
                    descriptors[i].dataBaseAddr = offset;
                }

                else if(hwOp == mv::NCE1HWOps::FullyConnected)
                {
                    //TODO
                }
                else
                    descriptors[i].dataBaseAddr = (fp16_size * input_width * output_channels_performed_so_far) + (input->getShape()[2] * fp16_size * input_width * input_line_start[h]);

                //Only case handled for now
                if( input->getOrder().isRowInterleaved() )
                {
                    descriptors[i].dataLnStr = inputBlobTensor.strideY;
                    descriptors[i].dataChStr = inputBlobTensor.strideZ;
                }
                descriptors[i].coeffBaseAddr = 0;
                descriptors[i].biasBaseAddr = 0;
                descriptors[i].scaleBaseAddr = 0;

                if ((hwOp == mv::NCE1HWOps::Convolution) || (hwOp == mv::NCE1HWOps::FullyConnected)) {

                    // Update taps base address
                    // https://github.com/movidius/mdk/blob/release_R6_181129_1401/projects/Fathom/src2/Controllers/HwCompiler.py
               
                    auto sodIndex = descriptors[i].sodGroup;
                    //auto bytesPerPixel = fp16_size;
                    auto scheduledOutputDimZ = output->getShape()[2];
                    auto scheduledInputDimZ = inputChannelsPadded;
                    auto taps_processed = output_channels_performed_so_far;

                    auto taps_total = mv::round_up(scheduledOutputDimZ, 8);
                    auto split_by_input_taps_sz = bytesPerPixel * taps_total * scheduledInputDimZ * radixY * radixX;
                    std::cout << "split_by_input_taps_sz: " << split_by_input_taps_sz << std::endl;
                    std::cout << "descriptors[i].coeffBaseAddr: " << sodIndex << "," << split_by_input_taps_sz << "," << bytesPerPixel << "," << taps_processed << "," << scheduledInputDimZ << "," << radixY << "," << radixX << std::endl;
                    descriptors[i].coeffBaseAddr = sodIndex * split_by_input_taps_sz + bytesPerPixel * taps_processed * scheduledInputDimZ *  radixY * radixX;

                    // Update bias base address
                    if (sodIndex == 0 && opIt->hasAttr("bias"))
                        descriptors[i].biasBaseAddr = output_channels_performed_so_far * bytesPerPixel;
                }

                // Update output base address
                // Note: only interleaved supported right now
                auto offset = output_line_start[h] * output->getShape()[2] * outputBlobTensor.strideZ + output_channels_performed_so_far * outputBlobTensor.strideZ;
                descriptors[i].outBaseAddr = offset;
                
                // Note: scale tensor patching is handled by FW
//-----------------------------------------------------------------------------------------------
                //HACK FOR CONCAT
                /* commenting this out as this seems causing tinyyolo4th conv to fail. 
                if(hwOp == mv::NCE1HWOps::Convolution)
                    descriptors[i].outBaseAddr = outputBlobTensor.strideZ * output_line_start[h] * output_channels; //output:427
                    BlobTensor.strideZ should be fp16_size * outputWidthPadded
                else if(hwOp == mv::NCE1HWOps::FullyConnected)
                {
                    //TODO
                }
                else
                    descriptors[i].outBaseAddr = (fp16_size * outputWidthPadded * output_channels_performed_so_far) + (outputChannelsPadded * fp16_size * outputWidthPadded * output_line_start[h]);
                */

                //Only case handled for now
                if( output->getOrder().isRowInterleaved() )
                {
                    descriptors[i].outLnStr = outputBlobTensor.strideY;
                    descriptors[i].outChStr = outputBlobTensor.strideZ;
                }

                if(hwOp == mv::NCE1HWOps::Convolution)
                {
                    auto weight_4dshape = taps->getShape();

                    descriptors[i].coeffChStrIn = weight_4dshape[2]*weight_4dshape[3]*weight_4dshape[4]* fp16_size;
                    int inChans = inputChannelsPadded;

                    descriptors[i].coeffChStrOut = radixX * radixY * inChans * fp16_size * 8; // (fp16)
                }

                for(unsigned j = 0; j != 32; j++)
                    desc.push_back(((unsigned *) &descriptors[i])[j]);
            }
        }
    }

    opIt->set<std::vector<unsigned>>("descriptors", desc);
}

void PopulateSerialFieldsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType(opIt->getOpType());
        std::cout << "Populating Serial fields for Op{" << opType << "}" << std::endl;
        //Short term fix: Big if-else acting like a switch
        //Long term solution: Move everything to Target Descriptor

        if(opType == "Add")
            opIt->set<unsigned>("SerialID", 12);

        else if(opType == "Identity")
            opIt->set<unsigned>("SerialID", 19);

        else if(opType == "AveragePool")
        {
            auto fp16_size = 2;

            if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible") )
            {
                fillMXDescriptors(cm, dm, fp16_size, opIt, mv::NCE1HWOps::AveragePooling);
            }
            else
            {
                opIt->set<unsigned>("SerialID", 2);

                opIt->set<unsigned>("radixX",  opIt->get<std::array<short unsigned, 2>>("kSize")[0]);
                opIt->set<unsigned>("radixY",  opIt->get<std::array<short unsigned, 2>>("kSize")[1]);
                opIt->set<unsigned>("strideX",  opIt->get<std::array<unsigned short, 2>>("stride")[0]);
                opIt->set<unsigned>("strideY",  opIt->get<std::array<unsigned short, 2>>("stride")[1]);
                opIt->set<unsigned>("padX",  opIt->get<std::array<unsigned short, 4>>("padding")[0]);
                opIt->set<unsigned>("padY",  opIt->get<std::array<unsigned short, 4>>("padding")[2]);
                opIt->set<unsigned>("padStyle",  2);

            }
        }
        else if(opType == "BatchNormalization")
        {

        }
        else if(opType == "Bias")
        {

        }
        else if(opType == "Concat")
        {

        }
        else if(opType == "Constant")
        {

        }
        else if(opType == "Conv")
        {
            auto fp16_size = 2;

            if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible"))
            {
                fillMXDescriptors(cm, dm, fp16_size, opIt, mv::NCE1HWOps::Convolution);
            }
            else
            {
                opIt->set<unsigned>("SerialID", 0);
                opIt->set<unsigned>("radixX",  opIt->getInputTensor(1)->getShape()[0]);
                opIt->set<unsigned>("radixY",  opIt->getInputTensor(1)->getShape()[1]);
                opIt->set<unsigned>("strideX",  opIt->get<std::array<unsigned short, 2>>("stride")[0]);
                opIt->set<unsigned>("strideY",  opIt->get<std::array<unsigned short, 2>>("stride")[1]);
                opIt->set<unsigned>("padX",  opIt->get<std::array<unsigned short, 4>>("padding")[0]);
                opIt->set<unsigned>("padY",  opIt->get<std::array<unsigned short, 4>>("padding")[2]);
                opIt->set<unsigned>("padStyle",  2);
                opIt->set<unsigned>("dilation",  1);
            }
        }
        else if(opType == "Conversion")
            opIt->set<unsigned>("SerialID", 37);

        else if(opType == "DepthwiseConv")
        {
            //auto fp16_size = 2;

            opIt->set<unsigned>("SerialID", 8);

            opIt->set<unsigned>("radixX",  opIt->getInputTensor(1)->getShape()[0]);
            opIt->set<unsigned>("radixY",  opIt->getInputTensor(1)->getShape()[1]);
            opIt->set<unsigned>("strideX",  opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            opIt->set<unsigned>("strideY",  opIt->get<std::array<unsigned short, 2>>("stride")[1]);
            opIt->set<unsigned>("padX",  opIt->get<std::array<unsigned short, 4>>("padding")[0]);
            opIt->set<unsigned>("padY",  opIt->get<std::array<unsigned short, 4>>("padding")[2]);
            opIt->set<unsigned>("padStyle",  2);
            opIt->set<unsigned>("dilation",  1);
        }
        else if(opType == "Divide")
            opIt->set<unsigned>("SerialID", 13);

        else if(opType == "Dropout")
        {

        }
        else if(opType == "FullyConnected")
        {
            //auto fp16_size = 2;

            if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible") )
            {
                //TODO
            }
            else
            {
                opIt->set<unsigned>("SerialID", 4);
            }
        }
        else if(opType == "Input")
        {

        }
        else if(opType == "MatMul")
            opIt->set<unsigned>("SerialID", 8);

        else if(opType == "MaxPool")
        {
            auto fp16_size = 2;

            if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible") )
            {
                fillMXDescriptors(cm, dm, fp16_size, opIt, mv::NCE1HWOps::MaxPooling);
            }
            else
            {
                opIt->set<unsigned>("SerialID", 1);

                opIt->set<unsigned>("radixX",  opIt->get<std::array<short unsigned, 2>>("kSize")[0]);
                opIt->set<unsigned>("radixY",  opIt->get<std::array<short unsigned, 2>>("kSize")[1]);
                opIt->set<unsigned>("strideX",  opIt->get<std::array<unsigned short, 2>>("stride")[0]);
                opIt->set<unsigned>("strideY",  opIt->get<std::array<unsigned short, 2>>("stride")[1]);
                opIt->set<unsigned>("padX",  opIt->get<std::array<unsigned short, 4>>("padding")[0]);
                opIt->set<unsigned>("padY",  opIt->get<std::array<unsigned short, 4>>("padding")[2]);
                opIt->set<unsigned>("padStyle",  2);

            }
        }
        else if(opType == "Multiply")
            opIt->set<unsigned>("SerialID", 13);
        else if(opType == "Output")
        {

        }
        else if(opType == "Prelu")
            opIt->set<unsigned>("SerialID", 10);
        else if(opType == "Relu")
        {
            opIt->set<unsigned>("opX", 0);
            opIt->set<unsigned>("strideX", 0);
            opIt->set<unsigned>("strideY", 0);
            opIt->set<unsigned>("SerialID", 6);
        }
        else if(opIt->getOpType() == "Elu")
        {
            opIt->set<unsigned>("alpha", opIt->get<unsigned>("alpha")); 
            opIt->set<unsigned>("strideX", 0);
            opIt->set<unsigned>("strideY", 0);
            opIt->set<unsigned>("SerialID", 23);
        }
        else if(opIt->getOpType() == "LeakyRelu")
        {
            std::cout << opIt->getOpType() << std::endl;

            opIt->set<unsigned>("alpha", opIt->get<double>("alpha"));
            opIt->set<unsigned>("strideX", 0);
            opIt->set<unsigned>("strideY", 0);
            opIt->set<unsigned>("SerialID", 42);

    
        }
        else if(opIt->getOpType() == "LocalResponseNormalization")
        {
            opIt->set<unsigned>("size", opIt->get<unsigned>("size")); 
            opIt->set<unsigned>("bias", opIt->get<unsigned>("bias")); 
            opIt->set<unsigned>("SerialID", 11);
        }
        else if(opIt->getOpType() == "Reshape")
        {

        }
        else if(opIt->getOpType() == "Sigmoid")
        {
            opIt->set<unsigned>("SerialID", 20);
        }
        else if(opIt->getOpType() == "Tanh")
        {
            opIt->set<unsigned>("SerialID", 21);
        }
        else if(opIt->getOpType() == "Scale")
        {
            opIt->set<unsigned>("SerialID", 15);
        }
        else if(opIt->getOpType() == "Softmax")
        {
            opIt->set<unsigned>("axis", 1);
            opIt->set<unsigned>("SerialID", 3);
        }
        else if(opType == "Subtract")
            opIt->set<unsigned>("SerialID", 12);

        else
            std::cout << "Unsupported serialization operation " << opType << std::endl;
    }
}
