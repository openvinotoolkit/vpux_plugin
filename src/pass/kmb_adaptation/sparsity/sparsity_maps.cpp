#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "mcm/utils/custom_math.hpp"
#include <math.h>

static void generateSparsityMapsPopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void generateSparsityMapsUnpopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void setSparsityAttrForUnpopulatedFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateSparsityMapsPopulatedTensors)
        .setFunc(generateSparsityMapsPopulatedTensorsFcn)
        .setDescription(
            "Generates sparsity maps for populated tensors."
        );

        MV_REGISTER_PASS(GenerateSparsityMapsUnpopulatedTensors)
        .setFunc(generateSparsityMapsUnpopulatedTensorsFcn)
        .setDescription(
            "Generates sparsity maps for unpopulated tensors."
        );

        MV_REGISTER_PASS(SetSparsityAttrForUnpopulatedTensors)
        .setFunc(setSparsityAttrForUnpopulatedFnc)
        .setDescription(
            "sets needs sparsity attr for unpopulated tensors."
        );
    }
}

mv::Data::TensorIterator createFakeSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& sparsityMapName, const mv::Shape& sparsityShape, const std::vector<int64_t>& sparsityMapData)
{
    auto sparsityMap = om.constantInt(sparsityMapData, sparsityShape, mv::DType("UInt8"), mv::Order("NHWC"), {{},{},{},{}},sparsityMapName);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    unsigned newSize = dpuTaskOp->addInputTensor(sparsityMap);
    om.defineFlow(sparsityMap, dpuTaskOp, newSize - 1);

    return sparsityMap;
}


std::vector<int8_t> createBitPattern(uint16_t kernelW, uint16_t kernelH, uint16_t windowsSize, uint16_t inputChannels)
{
    std::vector<int8_t> bitpattern;
    bitpattern.reserve(windowsSize*kernelH*inputChannels);
    for(size_t c = 0; c < inputChannels; c++)
        for(size_t y = 0; y < kernelH; y++)
            for(size_t x = 0; x < windowsSize; x++)
                if (x < kernelW)
                    bitpattern.emplace_back(1);
                else
                    bitpattern.emplace_back(0);
    return bitpattern;
}

// The sparsity maps relative to populated tensors have to be generated BEFORE the dma passes.
// As they have to be DMAed into CMX.
static void generateSparsityMapsPopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            bool weightsSparsity = dpuTask->hasAttr("weightsSparsity") ? dpuTask->get<bool>("weightsSparsity") : false;
            std::string taskOp = dpuTask->get<std::string>("taskOp");
            pass.log(mv::Logger::MessageType::Debug, " taskOp "  + dpuTask->get<std::string>("taskOp"));
            bool isChannelMajorConv = taskOp == "ChannelMajorConvolution";
            bool isPooling = taskOp == "MaxPool";
            bool isDepthWiseConv = taskOp == "DepthwiseConv";

            // NOTE by Marco: Checking for Elementwise is necessary, as it is the only operation
            // that does not support neither Real Sparsity or Fake Sparsity (aka Activation Window)
            // being it the hackiest operation ever
            bool isElementWise = taskOp == "Eltwise";

            //for max pooling and deptwise convolution (and CM conv, if enabled) we need to
            //generate sparsity data even if those layers do not support sparsity.
            if (isPooling || isDepthWiseConv || isChannelMajorConv)
            {
                uint16_t kernelW, kernelH;

                auto strides = dpuTask->get<std::array<unsigned short, 2>>("stride");
                auto inputChannels = dpuTask->getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
                auto outputChannels = dpuTask->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];

                // Using the check in this way, instead of on the operation type
                // makes this pass work on both aligned and unaligned convolutions.
                if (dpuTask->hasAttr("kSize"))
                {
                    auto kernelShape = dpuTask->get<std::array<unsigned short, 2>>("kSize");
                    kernelW = kernelShape[0];
                    kernelH = kernelShape[1];
                }
                else
                {
                    auto weightsShape = dpuTask->getInputTensor(1)->getShape();
                    kernelW = weightsShape[mv::KERNEL_WIDTH];
                    kernelH = weightsShape[mv::KERNEL_HEIGHT];
                }

                mv::DType dataType = mv::DType("UInt8");

                if (isPooling || isDepthWiseConv)
                {
                    if (dpuTask->hasAttr("floatPrecision") && dpuTask->get<bool>("floatPrecision"))
                    {
                        dataType = mv::DType("Float16");
                    }
                }

                auto windowsSize = getWindowSize(kernelW, strides[0], dataType);

                pass.log(mv::Logger::MessageType::Debug, "windowSize " + std::to_string(windowsSize));
                pass.log(mv::Logger::MessageType::Debug, "OutputChannels " + std::to_string(outputChannels));

                std::vector<int8_t> bitpattern;
                std::vector<uint8_t> perChannelSparsity;
                std::vector<std::size_t> ndims(4);
                if (isPooling || isDepthWiseConv)
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, 1));
                    perChannelSparsity.resize(static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)) * 16);//allocate once
                    ndims = {16 * static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)), 1, 1, inputChannels};
                }
                else //isChannelMajorConvolution
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, inputChannels));
                    auto windowSparsitySize = static_cast<std::size_t>(std::ceil(windowsSize/8.0)); //how many bytes we need per window
                    auto NumberOfRowsSparistyBytes = static_cast<std::size_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0 ));
                    perChannelSparsity.resize(NumberOfRowsSparistyBytes * 16);//allocate once
                    ndims = {16, NumberOfRowsSparistyBytes, 1, outputChannels};
                }

                int channelLenght = bitpattern.size();

                //populate sparsity
                pass.log(mv::Logger::MessageType::Debug, " perChannelSize = " + std::to_string(perChannelSparsity.size()) );
                for (size_t i = 0; i < bitpattern.size(); i++)
                {
                    //pass.log(mv::Logger::MessageType::Debug, " i = " + std::to_string(i));
                    //pass.log(mv::Logger::MessageType::Debug, " perChannelIDx = " + std::to_string(((i/128)*16 + (i%128)/8)) );
                    perChannelSparsity.at((i/128)*16 + (i%128)/8) |= bitpattern[i] << (i%8); //use at to check boundaries - just in case
                }

                //Create Tensor with Sparsity Data
                mv::Shape sparsityShape(ndims);
                std::vector<int64_t> data(sparsityShape.totalSize(), 0);
                //mv::Tensor sparsityTensor("backup", sparsityShape, mv::DType("UInt8"), mv::Order("WHCN"), data);
                auto sparsityTensor = mv::Tensor(dpuTask->getName() + "_sparse_dw", sparsityShape, mv::DType("UInt8"), mv::Order("NHWC"), data);

                // NOTE: This loop can probably be simplified without using an auxiliary tensor
                for(unsigned kx = 0; kx < sparsityShape[0]; ++kx)
                    for(unsigned ky = 0; ky < sparsityShape[1]; ++ky)
                        for(unsigned ic = 0; ic < sparsityShape[2]; ++ic)
                            for(unsigned oc = 0; oc < sparsityShape[3]; ++oc)
                                sparsityTensor.at({kx, ky, ic, oc}) = static_cast<int64_t>(perChannelSparsity[ky*sparsityShape[0] + kx]);

                std::string opName = dpuTask->getName();

                auto fakeSparsityMap = createFakeSparsityMap(om, dpuTask, mv::createFakeSparsityMapName(opName), sparsityShape, sparsityTensor.getIntData());
                fakeSparsityMap->set<int>("channelLength", channelLenght);

                dpuTask->set<bool>("fakeSparsity", true);
                dpuTask->set<size_t>("fakeSparsityIndex", dpuTask->inputSlots()-1);
            }
            else if(weightsSparsity && !isElementWise)
            {
                // Here only in the case of ZMajorConvolution
                auto weightsTensor = dpuTask->getInputTensor(1);

                // NOTE: Facultative, but doesn't cause overload
                weightsTensor->setOrder(mv::Order("NHWC"));

                if(weightsTensor->setSparse())
                    dm.defineTensor(weightsTensor->getSparsityMap());
            }
            //NOTE: Here is handled a specific case and this is why is treated seperately
            if (dpuTask->getOpType() == "DPUTask" && dpuTask->get<std::string>("taskOp") == "Conv")
            {
                if (dpuTask->hasAttr("activationSparsityCompilerSolving") && dpuTask->get<bool>("activationSparsityCompilerSolving"))
                {
                    auto inputTensor = dpuTask->getInputTensor(0);
                    //every element of sparsity map describes 8 elements of normal tensor
                    auto mapShape = mv::Shape({{inputTensor->getShape()[mv::IO_WIDTH_DIMENSION]},
                                               {inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]},
                                               {inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION]/8},
                                               {1}});
                    std::vector<int64_t> unpopulatedSparsityMapData(mapShape.totalSize(), 255);
                    mv::QuantizationParams quantParams = {{},{},{},{}};
                    std::string unpopulatedSparsityMapName = dpuTask->getName() + "activation_map";
                    auto unpopulatedSparsityMap = om.constantInt(unpopulatedSparsityMapData, mapShape, mv::DType("UInt8"), mv::Order("NHWC"), quantParams, unpopulatedSparsityMapName);
                    om.getSourceOp(unpopulatedSparsityMap)->set<unsigned>("opId", dpuTask->get<unsigned>("opId"));
                    unsigned newInputsSize = dpuTask->addInputTensor(unpopulatedSparsityMap);
                    om.defineFlow(unpopulatedSparsityMap, dpuTask, newInputsSize - 1);
                    dpuTask->set<size_t>("unpopulatedSparsityMapIndex", newInputsSize - 1);

                    mv::Shape storageElementShape = mv::Shape({{inputTensor->getShape()[mv::IO_WIDTH_DIMENSION]},
                                                               {inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]},
                                                               {1},
                                                               {1}});
                    std::vector<int64_t> storageElementData(storageElementShape.totalSize(), 0);
                    std::string storageElementName = dpuTask->getName() + "storage_element_map";
                    auto storageElement = om.constantInt(storageElementData, storageElementShape, mv::DType("Int32"), mv::Order("NHWC"), quantParams, storageElementName);
                    om.getSourceOp(storageElement)->set<unsigned>("opId", dpuTask->get<unsigned>("opId"));
                    unsigned newSize = dpuTask->addInputTensor(storageElement);
                    om.defineFlow(storageElement, dpuTask, newSize - 1);
                    dpuTask->set<size_t>("storageElementIndex", newSize - 1);
                }

                // Here we generate the sparsity maps required for dilated convolution
                // Should the sparsity map be size of the full input tensor or just the "sub conv" input tensor ?
                // I think it should be the full input tensor size ?
                if (dpuTask->hasAttr("activationSparsityCompilerSolvingForDilatedConv") && dpuTask->get<bool>("activationSparsityCompilerSolvingForDilatedConv"))
                {
                    auto inputTensor = dpuTask->getInputTensor(0);
                    //every element of sparsity map describes 8 elements of normal tensor

                    // if the sparsity map should only be the size of the "sub conv input tensor" then change this in future
                    auto mapShape = mv::Shape({{inputTensor->getShape()[mv::IO_WIDTH_DIMENSION]},
                                               {inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]},
                                               {inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION]/8},
                                               {1}});

                    std::vector<int64_t> unpopulatedSparsityMapData(mapShape.totalSize(), 255); // 255 converts to all 1's in SM
                    mv::QuantizationParams quantParams = {{},{},{},{}};
                    std::string unpopulatedSparsityMapName = dpuTask->getName() + "activation_map";
                    auto unpopulatedSparsityMap = om.constantInt(unpopulatedSparsityMapData, mapShape, mv::DType("UInt8"), mv::Order("NHWC"), quantParams, unpopulatedSparsityMapName);
                    om.getSourceOp(unpopulatedSparsityMap)->set<unsigned>("opId", dpuTask->get<unsigned>("opId"));
                    unsigned newInputsSize = dpuTask->addInputTensor(unpopulatedSparsityMap);
                    om.defineFlow(unpopulatedSparsityMap, dpuTask, newInputsSize - 1);
                    dpuTask->set<size_t>("unpopulatedSparsityMapIndex", newInputsSize - 1);

                    // Here we generate a storage element pointer table of all 0's for the dilated conv case
                    // The logic to generate SEPs for dilated conv should be added in weight_tables.cpp - function populateActivationStorageElementMapForDilatedConvolution()

                    mv::Shape storageElementShape = mv::Shape({{inputTensor->getShape()[mv::IO_WIDTH_DIMENSION]},
                                                               {inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]},
                                                               {1},
                                                               {1}});
                    std::vector<int64_t> storageElementData(storageElementShape.totalSize(), 0);
                    std::string storageElementName = dpuTask->getName() + "storage_element_map";
                    auto storageElement = om.constantInt(storageElementData, storageElementShape, mv::DType("Int32"), mv::Order("NHWC"), quantParams, storageElementName);
                    om.getSourceOp(storageElement)->set<unsigned>("opId", dpuTask->get<unsigned>("opId"));
                    unsigned newSize = dpuTask->addInputTensor(storageElement);
                    om.defineFlow(storageElement, dpuTask, newSize - 1);
                    dpuTask->set<size_t>("storageElementIndex", newSize - 1);
                }
            }
        }
    }
}

bool checkA0FloatSparsityBug(mv::Data::FlowListIterator flow)
{
    auto source = flow.source();
    auto sink = flow.sink();
    auto tensor = flow->getTensor();

    if(!tensor->isPopulated())
    {
        if((sink->hasAttr("floatPrecision") && sink->get<bool>("floatPrecision")) &&
                (source->hasAttr("mixedToFloat") && source->get<bool>("mixedToFloat")))
           return true;
        if((source->hasAttr("floatPrecision") && source->get<bool>("floatPrecision")) &&
                (sink->hasAttr("floatPrecision") && sink->get<bool>("floatPrecision")))
           return true;
    }
    return false;
}

bool compilerSolvesSparsity(mv::Data::FlowListIterator flow)
{
    auto source = flow.source();
    auto sink = flow.sink();
    auto tensor = flow->getTensor();

    if(!tensor->isPopulated())
    {
        if((sink->hasAttr("floatPrecision") && sink->get<bool>("floatPrecision")) &&
                (source->hasAttr("mixedToFloat") && source->get<bool>("mixedToFloat")) &&
                (sink->hasAttr("activationSparsityCompilerSolving")
                 && sink->get<bool>("activationSparsityCompilerSolving")))
           return true;
        if((source->hasAttr("floatPrecision") && source->get<bool>("floatPrecision")) &&
                (sink->hasAttr("floatPrecision") && sink->get<bool>("floatPrecision")) &&
                (sink->hasAttr("activationSparsityCompilerSolving")
                 && sink->get<bool>("activationSparsityCompilerSolving")))
           return true;
    }
    return false;
}

// Result of chat with Alessandro:
// An activation tensor can be sparse if and only if
// 1) It is an output of a DPUTask. (ODU populates storage element and sparsity map).
// 2) It is the input of a ZMajorConv (Only layer that supports IDU)

// An activation tensor MUST be sparse if it's:
// 1) SplitOverH
// 2) Involved in a ZMajorConvolution with kernel > 1 (HW bug)
// 3) In KMB-A0 float DPU task means that needs to consume sparse tensors.

// In the future, these two conditions could change. We have to sync with runtime.

// Eltwise, being the hackiest operation ever, potentially can support sparsity input, sharing the IDU with ZMajorConv, but the runtime currently doesn't allow it.

static void setSparsityAttrForUnpopulatedFnc(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::DataModel dm(model);
    mv::OpModel om(model);

    for(auto tensor = dm.tensorBegin(); tensor != dm.tensorEnd(); ++tensor)
    {
        // Probably an inner tensor, skip
        if(!tensor->hasAttr("flows"))
            continue;

        // Populated tensor, skip
        if(tensor->isPopulated())
            continue;

        auto flows = tensor->get<std::set<std::string>>("flows");

        bool tensorSparsifiable = true;
        bool tensorNeedsSparsity = false;
        bool inputActivationSparsity = false;
        bool outputActivationSparsity = false;

        // Conditions for activation sparsity
        // 1) The source of the activation tensor must be a DPUTask (Otherwise there's no ODU to write SM, SE)
        // 2) The sink of the activation tensor must be a ZMajor Convolution, the only operation
        //    with an IDU capable of handling sparsity data
        // 3) Runtime doesn't support SOK and activation sparsity neither in input nor output
        // 4) HACK-Configuration...idu will have 1 case that the compiler generates sparsity info
        for(auto& flowStr: flows)
        {
            auto flow = dm.getDataFlow(flowStr);
            if(checkA0SOHSparsityBug(flow))
            {
                tensorNeedsSparsity = true;
                break;
            }
            else if (checkA0FloatSparsityBug(flow) && !compilerSolvesSparsity(flow))
            {
                tensorNeedsSparsity = true;
                break;
            }
        }
        for(auto& flowStr: flows)
        {
            auto flow = dm.getDataFlow(flowStr);
            auto source = flow.source();
            auto sink = flow.sink();
            outputActivationSparsity |= source->hasAttr("outputActivationSparsity") ? source->get<bool>("outputActivationSparsity") : false;
            inputActivationSparsity |= sink->hasAttr("inputActivationSparsity") ? sink->get<bool>("inputActivationSparsity") : false;
            if(source->getOpType() != "DPUTask" ||
               source->get<std::string>("splitStrategy") == "SplitOverK" ||
               sink->getOpType() != "DPUTask" ||
               sink->get<std::string>("taskOp") != "Conv")
            {
                tensorSparsifiable = false;
                break;
            }
        }

        if(tensorNeedsSparsity && !tensorSparsifiable)
            throw std::runtime_error("Wrong strategy generated: tensor " + tensor->getName() + " needs sparsity but it can't be sparsified");
        if((tensorSparsifiable && inputActivationSparsity && outputActivationSparsity) || tensorNeedsSparsity)
        {
            tensor->set<bool>("needs_sparse", true);

            //Now that we know tensor will be sparse, mark the input as need subtensor aligned
            auto sourceOp = om.getSourceOp(tensor);

            if (sourceOp->getOpType() == "DPUTask" && (sourceOp->get<std::string>("taskOp") == "Conv" || sourceOp->get<std::string>("taskOp") == "DepthwiseConv"))//TODO More??
            {
                sourceOp->getInputTensor()[0]->set<bool>("needs_splits_aligned", true);
                // Handle Align op's input tensor split-alignment
                auto parentOp = om.getSourceOp(sourceOp->getInputTensor()[0]);
                if (parentOp->getOpType() == "Align")
                    parentOp->getInputTensor()[0]->set<bool>("needs_splits_aligned", true);
            }
        }

    }
}
static void generateSparsityMapsUnpopulatedTensorsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::DataModel dm(model);

    for(auto tensor = dm.tensorBegin(); tensor != dm.tensorEnd(); ++tensor)
    {
        // Probably an inner tensor, skip
        if(!tensor->hasAttr("flows"))
            continue;

        // Populated tensor, skip
        if(tensor->isPopulated())
            continue;

        if((tensor->hasAttr("needs_sparse") && tensor->get<bool>("needs_sparse")))
        {
            tensor->setSparse();
        }
    }
}
