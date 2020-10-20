#include "math.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/tensor/tiling.hpp"
#include "include/mcm/pass/pass_utils.hpp"


static void streamingOperationsFcn(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

static void streamBinaryDataWeightsFcn(const mv::pass::PassEntry&,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

static void streamCopyOperationsFcn(const mv::pass::PassEntry&,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(StreamingOperations)
        .setFunc(streamingOperationsFcn)
        .setDescription(
                "Generates New Ops according to Streaming Strategies that the graph provides");

        MV_REGISTER_PASS(StreamBinaryDataWeights)
        .setFunc(streamBinaryDataWeightsFcn)
        .setDescription(
            "The StreamOverK on Costant Operastions creates Constant + Slice, which is new smaller/fused Constants"
        );

        MV_REGISTER_PASS(StreamCopyOperations)
        .setFunc(streamCopyOperationsFcn)
        .setDescription(
            "This pass will handle the copy+slice pattern"
        );
    }
}

mv::Data::OpListIterator operationsReplacement(mv::Data::OpListIterator parentOpIt,
        mv::Data::TensorIterator sourceTensor,
        mv::OpModel & om,
        mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    while(opIt.parentsSize() > 1)
    {
        auto paramOp = opIt.leftmostParent();
        ++paramOp;
        om.removeOp(paramOp);
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        //no need to trigger a cascade, we know what we are doing
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}

struct opStreamingSplitDef
{
    std::string axis ;
    size_t numSplits ;
};

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling);
mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling);
mv::Data::TensorIterator solveBatchTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling);

std::map<std::string, std::function<mv::Data::TensorIterator(mv::ComputationModel&, mv::Data::OpListIterator, mv::Tiling&)>>
streamSplit =
{
    {"W",solveSpatialTiling},
    {"H",solveSpatialTiling},
    {"K",solveWeightsTiling},
    {"C",solveWeightsTiling}, //NOTE::Only Convolution/Depthwise is supported for SoK now
    {"N",solveBatchTiling}
};

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model,
        mv::Data::OpListIterator op,
        mv::Tiling& tiling)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    //solve SOW/H location
    //TODO:: stop hardcoding index....
    auto inputTensor = op->getInputTensor(0);
    auto kernelTensor = op->getInputTensor(1);
    auto outputTensor = op->getOutputTensor(0);
    mv::Shape kernelShape  = kernelTensor->getShape();
    auto kernelOp = om.getSourceOp(kernelTensor);

    auto inputQuantParams  = inputTensor->getQuantParams();
    auto outputQuantParams = outputTensor->getQuantParams();

    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    // Attributes query based on blacklist
    // Weights K || C (depthwise ops) stream, need only overwrite shape and bias
    auto attrsToCopy = op->getAttrs({"shape", "bias"});
    std::string splitStrategy = op->get<std::string>("splitStrategy");
    bool mixedToFloat = false;
    
    if(op->hasAttr("mixedToFloat"))
        mixedToFloat = op->get<bool>("mixedToFloat");

    std::vector<mv::Data::TensorIterator> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> newTensors(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    size_t biasStartIndex = 0;
    size_t biasEndIndex = 0;

    bool isDilatedConv = op->hasAttr("DilatedSubConv") && op->get<bool>("DilatedSubConv");
    bool avoidCmxConcat = op->hasAttr("avoidCmxConcat") && op->get<bool>("avoidCmxConcat");

    //todo::find a better location for this. Should not be slice.. but something like Copy layer... will do with dummy slice for speed
    //aslo.. have no idea why it's not working for the scenarion stream->concat->copySlice->stream when all is in CMX ... need debug.
    mv::Data::TensorIterator copyInput;
    if(om.getSourceOp(inputTensor)->getOpType() != "Slice")
    {
        copyInput = om.slice(inputTensor->getName() + op->getName() + "_KStreamCopyIn_",
                             inputTensor,
                             mv::Shape({0,0,0,0}),
                             inputTensor->getShape());
        copyInput->setQuantParams(inputQuantParams);
        auto copyInputOp = om.getSourceOp(copyInput);
        copyInputOp->set<unsigned>("opId", opId);
        copyInputOp->set<std::string>("splitStrategy", splitStrategy);
    }
    else
    {
        copyInput = inputTensor;
    }

    //NOTE: the idea here is that the n-1 first splits will be symmetrical on h/w
    //so in order to concatenate later for the dilation case we will need to know
    //the dim of the n-1 first streams and this should be stored in the last stream
    std::size_t symmetrical_first_dimension = 0;
    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator slice;
        auto kernelSliceShape = childTiles[split].getKernelShape();
        auto kernelSliceStart = childTiles[split].getKernelStart();
        kernelSliceShape[mv::KERNEL_HEIGHT] = kernelShape[mv::KERNEL_HEIGHT]; //the tiling does not contain KERNEL W/H Info
        kernelSliceShape[mv::KERNEL_WIDTH] = kernelShape[mv::KERNEL_WIDTH];

        if (isDilatedConv &&
                kernelOp->hasAttr("dilationConvKernelSliced")
                && kernelOp->get<bool>("dilationConvKernelSliced")) //already handled this dilated Conv, nothing to do
        {
            //find the proper slice
            bool sliceFound = false;
            for (auto sinkFlow = kernelOp.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                auto sinkOp = sinkFlow.sink();
                if (sinkOp->getOpType() == "Slice" && sinkOp->hasAttr("dilatedConvKernelSliceIdx")
                        && sinkOp->get<unsigned>("dilatedConvKernelSliceIdx") == split)
                {
                    slice = sinkOp->getOutputTensor(0);
                    sliceFound = true;
                    break;
                }
            }
            if (!sliceFound)
                std::cout << "ERROR: Slice for dilatedConv weights hasn't been found although kernel was marked as already Sliced!" << std::endl;
            assert(sliceFound);
        }
        else
        {

            //todo:: clean this if-then-else quantParams logic
            if (kernelTensor->hasAttr("quantParams"))
            {
                auto sliceQuantParams = kernelTensor->get<mv::QuantizationParams>("quantParams");
                if (kernelTensor->get<mv::QuantizationParams>("quantParams").getScale().size() > 1)
                {
                    std::size_t outputChannelsofSlice = 0, starting_point = 0;
                    if (op->getOpType() == "Conv")
                    {
                        outputChannelsofSlice = childTiles[split].getSize()[mv::KERNEL_OUTPUT_CHANNELS];
                        starting_point = childTiles[split].getStartCoord()[mv::KERNEL_OUTPUT_CHANNELS];
                    }
                    else if (op->getOpType() == "DepthwiseConv")
                    {
                        outputChannelsofSlice = childTiles[split].getSize()[mv::KERNEL_INPUT_CHANNELS];
                        starting_point = childTiles[split].getStartCoord()[mv::KERNEL_INPUT_CHANNELS];
                    }
                    std::vector<double> scales(outputChannelsofSlice);
                    std::vector<int64_t> zeros(outputChannelsofSlice);
                    for (std::size_t i = starting_point; i < starting_point + outputChannelsofSlice; i++)
                    {
                        scales.at(i - starting_point) = sliceQuantParams.getScale()[i];
                        zeros.at(i - starting_point) = sliceQuantParams.getZeroPoint()[i];
                    }
                    sliceQuantParams = mv::QuantizationParams(zeros,
                                                                scales,
                                                                sliceQuantParams.getMin(),
                                                                sliceQuantParams.getMax());
                }

                slice = om.slice(kernelTensor->getName() + inputTensor->getName() + "_sliceK" + std::to_string(split),
                                kernelTensor,
                                kernelSliceStart,
                                kernelSliceShape);
                slice->setQuantParams(sliceQuantParams);
            }
            else
            {
                slice = om.slice(kernelTensor->getName() + "_sliceK" + std::to_string(split),
                                kernelTensor,
                                kernelSliceStart,
                                kernelSliceShape);
            }
            om.getSourceOp(slice)->set<unsigned>("opId", opId);

            if(isDilatedConv) //first time streaming if we are here, mark slice index for other subConvs
            {
                om.getSourceOp(slice)->set<unsigned>("dilatedConvKernelSliceIdx", split);
            }
        }
        std::string streamingOpName = op->getName() + "_streamK" + std::to_string(split);
        mv::Data::TensorIterator newTensor;
        //todo:: clean this if-then-else conv/DpthwiseConv logic... it's just bloatware code

        if (op->getOpType() == "Conv")
        {
            //todo:: place it in a more generic location

            newTensor = om.conv(streamingOpName,
                                copyInput,
                                slice,
                                op->get("stride"),
                                op->get("padding"),
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"));
            newTensor->setQuantParams(outputQuantParams);
            newTensor->setDType(outputTensor->getDType());
            newTensor->setOrder(mv::Order("NHWC"));

            if (split != number_of_splits - 1)
                symmetrical_first_dimension = newTensor->getShape()[mv::IO_CHANNEL_DIMENSION];

            if ((op->hasAttr("DilatedSubConv") && op->get<bool>("DilatedSubConv")) || (op->hasAttr("DeconvSubConv") && op->get<bool>("DeconvSubConv")))
            {
                om.getSourceOp(newTensor)->set<unsigned>("streamKId", split);
                om.getSourceOp(newTensor)->set<std::size_t>("symmetrical_first_dimensionK",
                                                                symmetrical_first_dimension);
            }
        }
        else if (op->getOpType() == "DepthwiseConv")
        {
            auto sliceShape = childTiles[split].getActivationShape();
            auto sliceStart = childTiles[split].getActivationStart();

            auto sliceInput = om.slice(op->getName() + "_sliceHK_" + std::to_string(split),
                                copyInput,
                                sliceStart,
                                sliceShape);
            sliceInput->setQuantParams(inputQuantParams);

            newTensor = om.depthwiseConv(streamingOpName,
                                sliceInput,
                                slice,
                                op->get("stride"),
                                op->get("padding"),
                                op->get<unsigned>("dilationFactor"));
            newTensor->setQuantParams(outputQuantParams);
            if((op->hasAttr("asymmetricKernel")))
            {
                om.getSourceOp(newTensor)->set<unsigned>("asymmetricKernel", op->get<unsigned>("asymmetricKernel"));
            }
            auto sliceInputOp = om.getSourceOp(sliceInput);
            sliceInputOp->set<unsigned>("opId", opId);
            sliceInputOp->set<std::string>("splitStrategy", splitStrategy);
        }

        // Does more harm than good, since mixed precision is not treated correctly
        // further on
        // // Restore original out dtype, to account for mixed precision cases
        // // where we don't want the same datatype for output as the input tensors
        // newTensor->setDType(op->getOutputTensor(0)->getDType());
        om.getSourceOp(newTensor)->set<unsigned>("opId", opId);

        //todo: clean this if-then-else bias logic.... bloatware code....
        if (op->hasAttr("bias"))
        {
            auto tileSize = kernelSliceShape[axisToSplit];
            biasStartIndex = kernelSliceStart[axisToSplit];
            biasEndIndex = biasStartIndex + tileSize;

            auto biasTensorName = op->get<std::string>("bias");
            auto originalBiasTensor = dm.getTensor(biasTensorName);
            auto oiginalBiasData = originalBiasTensor->getData();

            if ( biasEndIndex > oiginalBiasData.size())
                biasEndIndex = oiginalBiasData.size();
            std::vector<mv::DataElement>::const_iterator biasFirst = oiginalBiasData.begin() + biasStartIndex;
            std::vector<mv::DataElement>::const_iterator biasLast = oiginalBiasData.begin() + biasEndIndex;
            std::vector<mv::DataElement> subBiasData(biasFirst, biasLast);
            std::string newBiasTensorName = mv::createBiasName(op->getName() + "_split_" + std::to_string(split));
            mv::Data::TensorIterator biasTensorX;
            if (originalBiasTensor->hasAttr("quantParams"))
            {
                auto biasAttrQPs = originalBiasTensor->get("quantParams");
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData, biasAttrQPs ));
            }
            else
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData));
            om.addAttr(om.getSourceOp(newTensor), "bias", biasTensorX->getName());
        }
        auto newOp = om.getSourceOp(newTensor);

        newOp->set<bool>("splitted",true);//TODO::temporary hack. To remove once the iteration conditions are updated
        newOp->setAttrs(attrsToCopy);

        slices[split] = slice;
        newTensors[split] = newTensor;

        bool enableSerialStreaming = true;
        if ((split>0)&&(enableSerialStreaming))
            cm.defineFlow(om.getSourceOp(newTensors[split-1]), om.getSourceOp(newTensors[split]));
    }

    kernelTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;

    //in case of non-symmetric stream, we neet to check if at least one op is the last in the chain
    bool atLeastOneOpIsLast = false;
    for (unsigned idx = 0 ; idx < number_of_splits ; ++idx)
    {
        auto slice = slices[idx];
        auto newTensor = newTensors[idx];
        mv::Tensor::MemoryLocation inputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        mv::Tensor::MemoryLocation outputLocation(mv::Tensor::MemoryLocation::DEFAULT);

        auto numChildStreames = tiling.childTiles()[idx].childTiles().size();

        if(numChildStreames > 1)
        {
            //todo::should not be this convoluted to get the parentTensor of a tensor .....
            //layer may have multiple inputs with different locations (eltwise). Each inputTensor will get a slice layer based on the stream
            //so, for deciding the location of the slice, we have to check each input of the slice respectively
            inputLocation.relocate(inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            atLeastOneOpIsLast = true;
            inputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        }
        slice->set<mv::Tensor::MemoryLocation>("Location",inputLocation);
        newTensor->set<mv::Tensor::MemoryLocation>("Location",outputLocation);
    }
    //todo::better solution for this... need to decide on the location of the CopyInput
    {
        if(atLeastOneOpIsLast)
            copyInput->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::NNCMX);
        else
            copyInput->set<mv::Tensor::MemoryLocation>("Location",inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
    }

    for(unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if(childTiles[split].childTiles().size() > 1)
        {
            auto newStreamAxis = childTiles[split].getAxis();
            auto newStreamFunc = streamSplit[newStreamAxis];

            out = newStreamFunc(om,om.getSourceOp(newTensors[split]),childTiles[split]);
            om.removeOp(om.getSourceOp(newTensors[split]));
        }
        else
        {
            out = newTensors[split];
        }
        final_outputs[split] = out;
    }

    auto concat = om.concat(op->getName() + "concat_",
                    final_outputs,
                    "C");
    concat->setDType(op->getOutputTensor(0)->getDType());
    concat->setQuantParams(outputQuantParams);

    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    if(op->hasAttr("schedule_for_dpu_dma_overlap"))
    {
        auto pipelineId = op->get<unsigned>("schedule_for_dpu_dma_overlap");
        om.getSourceOp(concat)->set<unsigned>("schedule_for_dpu_dma_overlap", pipelineId);
    }
    if(avoidCmxConcat)
        om.getSourceOp(concat)->set<bool>("avoid_cmx_concat", true);
        
    if(mixedToFloat)
        om.getSourceOp(concat)->set<bool>("mixedToFloat", mixedToFloat);

    concat->set<mv::Tensor::MemoryLocation>("Location",outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
    if(isDilatedConv && !kernelOp->hasAttr("dilationConvKernelSliced"))
    {
        kernelOp->set<bool>("dilationConvKernelSliced", true);
    }
    return concat;
}

mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model,
                    mv::Data::OpListIterator op,
                    mv::Tiling& tiling)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto outputTensor = op->getOutputTensor("output");
    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    // Attributes query based on blacklist
    // Spatial H || W stream, need only overwrite shape, padding
    auto attrsToCopy = op->getAttrs({"padding", "shape"});
    std::string splitStrategy = op->get<std::string>("splitStrategy");
    bool avoidCmxConcat = op->hasAttr("avoidCmxConcat") && op->get<bool>("avoidCmxConcat");

    std::vector<mv::Data::TensorIterator> slices;
    std::vector<mv::Data::TensorIterator> newTensors(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    std::array<unsigned short, 2> kernelStride;
    if (op->hasAttr("stride"))
        kernelStride = op->get<std::array<unsigned short, 2>>("stride");
    else
        kernelStride = {1,1};//fake stride

    //NOTE: assuming order of paddings: left,right,top,bottom
    std::array<unsigned short, 4> padding;
    if (op->hasAttr("padding"))
        padding = op->get<std::array<unsigned short, 4>>("padding");
    else
        padding = {0, 0, 0, 0};

    auto startPad = padding;
    auto endPad = padding;
    auto middlePad = padding;
    auto currentPad = padding;

    if (axisToSplit == mv::Shape::getAxis("W"))
    {
        startPad[1] = 0;
        endPad[0] = 0;
        middlePad[0] = 0;
        middlePad[1] = 0;
    }
    if (axisToSplit == mv::Shape::getAxis("H"))
    {
        startPad[3] = 0;
        endPad[2] = 0;
        middlePad[2] = 0;
        middlePad[3] = 0;
    }
    std::size_t symmetrical_first_dimension = 0;
    std::size_t symmetrical_first_dimension_input = 0;
    for (unsigned split = 0; split < number_of_splits; split++)
    {
        if (split == 0)
            currentPad = startPad;
        else if (split == (number_of_splits -1))
            currentPad = endPad;
        else
            currentPad = middlePad;

        mv::Data::TensorIterator newTensor;
        std::string opType = op->getOpType();
        std::string streamingOpName = op->getName() + "_streamH" + std::to_string(split);
        if (opType == "MaxPool" || opType == "Conv" || opType == "DepthwiseConv")
        {
            auto inputTensor  = op->getInputTensor(0);
            auto outputTensor = op->getOutputTensor(0);

            auto inputQuantParams  = inputTensor->getQuantParams();
            auto outputQuantParams = outputTensor->getQuantParams();

            auto sliceShape = childTiles[split].getActivationShape();
            auto sliceStart = childTiles[split].getActivationStart();

            auto slice = om.slice(op->getName() + "_sliceH" + std::to_string(split),
                                inputTensor,
                                sliceStart,
                                sliceShape);
            slice->setQuantParams(inputQuantParams);
            om.getSourceOp(slice)->set<unsigned>("opId", opId);

            if (opType == "MaxPool")
                newTensor = om.maxPool(streamingOpName,
                                slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                currentPad,
                                op->get<const bool>("exclude_pad"));

            if (opType == "DepthwiseConv")
                newTensor = om.depthwiseConv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"));

            if (opType == "Conv") {
                newTensor = om.conv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"));
                newTensor->setOrder(mv::Order("NHWC"));
            }

            newTensor->setDType(outputTensor->getDType());
            newTensor->setQuantParams(outputQuantParams);

            if (split != number_of_splits - 1)
            {
                symmetrical_first_dimension = newTensor->getShape()[mv::IO_HEIGHT_DIMENSION];
            }
            if ((op->hasAttr("DilatedSubConv") && op->get<bool>("DilatedSubConv")) || (op->hasAttr("DeconvSubConv") && op->get<bool>("DeconvSubConv")))
            {
                om.getSourceOp(newTensor)->set<unsigned>("streamHId", split);
                om.getSourceOp(newTensor)->set<std::size_t>("symmetrical_first_dimensionH"
                                                         , symmetrical_first_dimension);
            }
            symmetrical_first_dimension_input += slice->getShape()[mv::IO_HEIGHT_DIMENSION];
            if((op->hasAttr("asymmetricKernel")))
                om.getSourceOp(newTensor)->set<unsigned>("asymmetricKernel", op->get<unsigned>("asymmetricKernel"));
            slices.push_back(slice);
        }
        else if (opType == "Eltwise")
        {
            auto inputSlots = op->inputSlots();
            std::vector<mv::Data::TensorIterator> eltwiseSlices;
            auto eltwiseType = op->get<std::string>("eltwiseType");
            auto originalDType = op->getOutputTensor(0)->getDType();
            for (unsigned i = 0; i < inputSlots; i++)
            {
                auto inputTensor = op->getInputTensor(i);
                auto inputQuantParams = inputTensor->getQuantParams();

                auto sliceShape = childTiles[split].getActivationShape();
                auto sliceStart = childTiles[split].getActivationStart();

                auto slice = om.slice(op->getName() + "_sliceH" + std::to_string(split) + "_" + std::to_string(i),
                                inputTensor,
                                sliceStart,
                                sliceShape);
                slice->setQuantParams(inputQuantParams);
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
                slices.push_back(slice);
                eltwiseSlices.push_back(slice);
            }

            auto quantParams = op->getOutputTensor(0)->getQuantParams();
            newTensor = om.eltwise(op->getName() + "_streamH" + std::to_string(split),
                                eltwiseSlices,
                                eltwiseType);
            newTensor->setDType(originalDType);
            newTensor->setQuantParams(quantParams);
        }

        // Restore original out dtype, to account for mixed precision cases
        // where we don't want the same datatype for output as the input tensors
        newTensor->setDType(op->getOutputTensor(0)->getDType());
        auto newOp = om.getSourceOp(newTensor);

        newOp->setAttrs(attrsToCopy);
        newOp->set<bool>("splitted", true);//TODO::temporary hack. To remove once the iteration conditions are updated

        newTensors[split] = newTensor;

        bool enableSerialStreaming = true;
        if ((split > 0) && enableSerialStreaming)
            cm.defineFlow(om.getSourceOp(newTensors[split-1]), om.getSourceOp(newTensors[split]));
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;
    // Unlike Kstream, we may have different number of "newOutputs" than slices, since Ops can have multiple inputs
    // and the concept of stream is per OP not per tensor // todo:: find way with less duplication of code&logic

    auto numChildStreames = tiling.childTiles().size();
    for (auto newTensor : newTensors)
    {
        mv::Tensor::MemoryLocation outputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        if(numChildStreames > 1)
        {
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            //NOTE: the idea here is that if you are not aligned with 16 channels in case that you are a z-maj
            //operation later you will have added the mechanism of align crop operation dmas to solve the //16
            //so if you are the last layer do not populate the output as a location but the ddr, leaving it in comments
            //as it is used mainly for the modelCutter, normally the locations should be handled in the placement of the
            //crop,align, quantize etc...
//            if ((newTensor->getShape()[mv::IO_CHANNEL_DIMENSION] % 16 != 0) &&
//                    outputTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::OUTPUT)
//                outputLocation.relocate(mv::Tensor::MemoryLocation::DDR);
        }
        else
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        newTensor->set<mv::Tensor::MemoryLocation>("Location",outputLocation);

    }
    for (auto slice : slices)
    {
        mv::Tensor::MemoryLocation inputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        if(numChildStreames > 1)
        {
            auto sliceInputTensor = om.getSourceOp(slice)->getInputTensor(0);
            inputLocation .relocate(sliceInputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            inputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        }
        slice->set<mv::Tensor::MemoryLocation>("Location",inputLocation);
    }


    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if (childTiles[split].childTiles().size() > 1)
        {
            auto newStreamAxis = childTiles[split].getAxis();
            auto newStreamFunc = streamSplit[newStreamAxis];

            out = newStreamFunc(om, om.getSourceOp(newTensors[split]), childTiles[split]);
            om.removeOp(om.getSourceOp(newTensors[split]));
        }
        else
            out = newTensors[split];
        final_outputs[split] = out;
    }

    auto quantParams = op->getOutputTensor(0)->getQuantParams();
    auto concat = om.concat(op->getName() + "concat_",
                    final_outputs,
                    tiling.getAxis());
    concat->setDType(op->getOutputTensor(0)->getDType());
    concat->setQuantParams(quantParams);
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    if(op->hasAttr("schedule_for_dpu_dma_overlap"))
    {
        auto pipelineId = op->get<unsigned>("schedule_for_dpu_dma_overlap");
        om.getSourceOp(concat)->set<unsigned>("schedule_for_dpu_dma_overlap", pipelineId);
    }
    if(avoidCmxConcat)
        om.getSourceOp(concat)->set<bool>("avoid_cmx_concat", true);
    concat->set<mv::Tensor::MemoryLocation>("Location", outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

mv::Data::TensorIterator solveBatchTiling(mv::ComputationModel& model,
                    mv::Data::OpListIterator op,
                    mv::Tiling& tiling)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto outputTensor = op->getOutputTensor("output");
    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto childTiles = tiling.childTiles();

    // Attributes query based on blacklist
    // Batch stream, need only overwrite shape
    auto attrsToCopy = op->getAttrs({"shape"});
    std::string splitStrategy = op->get<std::string>("splitStrategy");

    std::vector<mv::Data::TensorIterator> slices;
    std::vector<mv::Data::TensorIterator> newTensors(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    std::array<unsigned short, 2> kernelStride;
    if (op->hasAttr("stride"))
        kernelStride = op->get<std::array<unsigned short, 2>>("stride");
    else
        kernelStride = {1,1};//fake stride

    //NOTE: assuming order of paddings: left,right,top,bottom
    std::array<unsigned short, 4> padding;
    if (op->hasAttr("padding"))
        padding = op->get<std::array<unsigned short, 4>>("padding");
    else
        padding = {0, 0, 0, 0};

    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator newTensor;
        std::string opType = op->getOpType();
        std::string streamingOpName = op->getName() + "_stream" + tiling.getAxis() + std::to_string(split);
        if (opType == "MaxPool" || opType == "Conv" || opType == "DepthwiseConv")
        {
            auto inputTensor  = op->getInputTensor(0);
            auto outputTensor = op->getOutputTensor(0);

            auto inputQuantParams  = inputTensor->getQuantParams();
            auto outputQuantParams = outputTensor->getQuantParams();

            auto sliceShape = childTiles[split].getActivationShape();
            auto sliceStart = childTiles[split].getActivationStart();

            auto slice = om.slice(op->getName() + "_slice" + tiling.getAxis() + std::to_string(split),
                                inputTensor,
                                sliceStart,
                                sliceShape);
            slice->setQuantParams(inputQuantParams);
            om.getSourceOp(slice)->set<unsigned>("opId", opId);

            if (opType == "MaxPool")
                newTensor = om.maxPool(streamingOpName,
                                slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                padding,
                                op->get<const bool>("exclude_pad"));

            if (opType == "DepthwiseConv")
                newTensor = om.depthwiseConv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                padding,
                                op->get<unsigned>("dilationFactor"));

            if (opType == "Conv")
                newTensor = om.conv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                padding,
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"));
            newTensor->setDType(outputTensor->getDType());
            newTensor->setQuantParams(outputQuantParams);

            slices.push_back(slice);
        }
        else if (opType == "Eltwise")
        {
            auto inputSlots = op->inputSlots();
            auto eltwiseType = op->get<std::string>("eltwiseType");
            auto originalDType = op->getOutputTensor(0)->getDType();
            for (unsigned i = 0; i < inputSlots; i++)
            {
                auto inputTensor = op->getInputTensor(i);
                auto inputQuantParams = inputTensor->getQuantParams();

                auto sliceShape = childTiles[split].getActivationShape();
                auto sliceStart = childTiles[split].getActivationStart();

                auto slice = om.slice(op->getName() + "_slice"  + tiling.getAxis() + std::to_string(split) + "_" + std::to_string(i),
                                inputTensor,
                                sliceStart,
                                sliceShape);
                slice->setQuantParams(inputQuantParams);
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
                slices.push_back(slice);
            }

            auto quantParams = op->getOutputTensor(0)->getQuantParams();
            newTensor = om.eltwise(op->getName() + "_stream" + tiling.getAxis() + std::to_string(split),
                                   slices,
                                   eltwiseType);
            newTensor->setDType(originalDType);
            newTensor->setQuantParams(quantParams);
        }

        // Restore original out dtype, to account for mixed precision cases
        // where we don't want the same datatype for output as the input tensors
        newTensor->setDType(op->getOutputTensor(0)->getDType());
        auto newOp = om.getSourceOp(newTensor);

        newOp->setAttrs(attrsToCopy);
        newOp->set<bool>("splitted", true);//TODO::temporary hack. To remove once the iteration conditions are updated

        newTensors[split] = newTensor;

        bool enableSerialStreaming = true;
        if ((split > 0) && enableSerialStreaming)
            cm.defineFlow(om.getSourceOp(newTensors[split-1]), om.getSourceOp(newTensors[split]));
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;
    // Unlike Kstream, we may have different number of "newOutputs" than slices, since Ops can have multiple inputs
    // and the concept of stream is per OP not per tensor // todo:: find way with less duplication of code&logic

    auto numChildStreames = tiling.childTiles().size();
    for (auto newTensor : newTensors)
    {
        mv::Tensor::MemoryLocation outputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        if(numChildStreames > 1)
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        else
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        newTensor->set<mv::Tensor::MemoryLocation>("Location",outputLocation);

    }
    for (auto slice : slices)
    {
        mv::Tensor::MemoryLocation inputLocation(mv::Tensor::MemoryLocation::DEFAULT);
        if(numChildStreames > 1)
        {
            auto sliceInputTensor = om.getSourceOp(slice)->getInputTensor(0);
            inputLocation .relocate(sliceInputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            inputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
        }
        slice->set<mv::Tensor::MemoryLocation>("Location",inputLocation);
    }


    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if (childTiles[split].childTiles().size() > 1)
        {
            auto newStreamAxis = childTiles[split].getAxis();
            auto newStreamFunc = streamSplit[newStreamAxis];

            out = newStreamFunc(om, om.getSourceOp(newTensors[split]), childTiles[split]);
            om.removeOp(om.getSourceOp(newTensors[split]));
        }
        else
            out = newTensors[split];
        final_outputs[split] = out;
    }

    auto quantParams = op->getOutputTensor(0)->getQuantParams();
    auto concat = om.concat(op->getName() + "concat_",
                    final_outputs,
                    tiling.getAxis());
    concat->setDType(op->getOutputTensor(0)->getDType());
    concat->setQuantParams(quantParams);
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    concat->set<mv::Tensor::MemoryLocation>("Location", outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

void streamingOperationsFcn(const mv::pass::PassEntry& pass,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element& passDesc,
                                mv::Element&)
{

    mv::OpModel om(model);

    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No custom streaming strategy provided");
        return;
    }
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    //NOTE: NESTED STREAMING MEANS 2 LEVELS OF STREAMING, eg. HK, Stream Over H will stream
    //the input Tensor of the Op and then for every new Op have to stream it over K, which
    //means the weights will be repeated for the second level of streaming, this is why need
    //the data structures below...to create only one pair of nested slices

    for (auto layerNameStrategy : strategyList)
    {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        //NOTE: Graph optimizer will never do that but needs to be here for manual Scheduling
        if (!om.checkOp(nodeName))
        {
            pass.log(mv::Logger::MessageType::Info, nodeName + " is not present in model, skipping streaming");
            continue;
        }
        auto opIt =  om.getOp(nodeName);
        std::string opType = opIt->getOpType();

        //For now do streaming pass only for the DPU layers
        if ((opType != "Conv") && (opType != "DepthwiseConv") && (opType != "MaxPool") && (opType != "Eltwise"))
            continue;

        std::size_t alignment = 1;
        if(passDesc.hasAttr("alignment"))
            alignment = passDesc.get<int>("alignment");

        auto inputTensor = opIt->getInputTensor(0);
        auto inputShape = inputTensor->getShape();

        mv::Tiling masterTile;
        if((opType == "Conv") || (opType == "DepthwiseConv"))
        {
            //op has kernel
            auto kernelShape = opIt->getInputTensor(1)->getShape();
            masterTile = mv::Tiling(inputShape,kernelShape);
        }
        else
        {
            //for multi-input ops, this pass is assuming that all inputs are equalt, and the streams happens simetrically (Eltwise)
            masterTile = mv::Tiling(inputShape);
        }

        auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");

        std::vector<mv::Tiling*> tiles = {&masterTile};

        auto applyTiling = [opIt, alignment, pass](mv::Element& split, mv::Tiling& tile) -> std::vector<mv::Tiling>*
        {
            //the axis&split are stored in a map with key-> val .....
            //Don't want to if-then-else over all possible values of the axis...
            //the map should have only one key.. this is the draw-back of too generic mv::Element
            auto axis = split.attrsKeys()[0];
            auto numSplits = split.get<int>(axis);

            pass.log(mv::Logger::MessageType::Debug, opIt->getName() +
                " " + axis + " : " + std::to_string(numSplits));
            if(numSplits > 1)
            {
                tile.setAxis(axis);
                tile.setAlignment(alignment);
                tile.resizeNumberOfTiles(numSplits);
                tile.generateTiling(opIt);
                return &tile.childTiles();
            }
            else
            {
                return nullptr;
            }
        };

        for (auto split : splitList)
        {
            std::vector<mv::Tiling*> newChildTiles(0);
            for(auto tile : tiles)
            {
                auto childTiles = applyTiling(split,*tile);
                if(childTiles)
                {
                    for(auto& childTile : *childTiles)
                    {
                        newChildTiles.push_back(&childTile);
                    }
                }
                else
                {
                    newChildTiles.push_back(tile);
                }
            }
            tiles = newChildTiles;
        }

        if(masterTile.childTiles().size() > 1)
        {
            auto result = (streamSplit[masterTile.getAxis()])(om, opIt, masterTile);
            //NOTE: FlowSibling iterators seem to lose some sinks so they are replced...
            // reconnect children to subgraph
            std::vector<std::pair<mv::Data::OpListIterator,size_t>> toReturn;
            auto outputTensor = opIt->getOutputTensor()[0];
            for (auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
            {
                auto consumer = output.sink();
                std::size_t slot = 0;
                for (std::size_t input_idx = 0; input_idx < consumer->getInputTensor().size(); input_idx++)
                    if (consumer->getInputTensor()[input_idx]->getName() == outputTensor->getName())
                        slot = input_idx;
                toReturn.push_back(std::make_pair(consumer, slot));
            }

            om.removeOp(opIt);
            for (unsigned j = 0; j < toReturn.size(); ++j)
            {
                toReturn[j].first->setInputTensor(result, toReturn[j].second, false);
                om.defineFlow(result, toReturn[j].first, toReturn[j].second);
            }
        }
}
}


static void streamBinaryDataWeightsFcn(const mv::pass::PassEntry& ,
                                        mv::ComputationModel& model,
                                        mv::TargetDescriptor& ,
                                        mv::Element& ,
                                        mv::Element &)
{
    //Need to duplicate the consts to number equal to streams, cause of the binary_data
    mv::OpModel om(model);

    std::set <std::string> removeConstantsSet;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();

        if (opType == "Slice" && opIterator->getInputTensor(0)->isPopulated())
        {
            auto inTensorSlice = opIterator->getInputTensor(0);
            removeConstantsSet.insert(om.getSourceOp(inTensorSlice)->getName());
            auto outTensorSlice = opIterator->getOutputTensor(0);
            auto parentOpIt = om.getSourceOp(opIterator->getInputTensor(0));
            auto shape = outTensorSlice->getShape();
            auto quantParams = outTensorSlice->getQuantParams();

            auto newConstant = om.constantDataElement(opIterator->getName() + "_weights",
                                                      outTensorSlice->getData(), shape,
                                                      outTensorSlice->getDType(), outTensorSlice->getOrder());
            newConstant->setQuantParams(quantParams);
            newConstant->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
            auto constantOp = om.getSourceOp(newConstant);
            if(opIterator->hasAttr("opId"))
            {
                unsigned currentOpId = opIterator->get<unsigned>("opId");
                constantOp->set<unsigned>("opId", currentOpId);
            }
            opIterator = operationsReplacement(parentOpIt, newConstant, om, opIterator);
        }
    }
    for (auto& opName:removeConstantsSet)
        om.removeOp(om.getOp(opName));
}

static void streamCopyOperationsFcn(const mv::pass::PassEntry& ,
                                        mv::ComputationModel& model,
                                        mv::TargetDescriptor& ,
                                        mv::Element& ,
                                        mv::Element &)
{
    //Need to duplicate the consts to number equal to streams, cause of the binary_data
    mv::OpModel om(model);

    std::set <std::string> removeCopySet;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();

        if (opType == "Slice" && (!opIterator->getInputTensor(0)->isPopulated()))
        {
            auto previousOp = om.getSourceOp(opIterator->getInputTensor(0));
            if (previousOp->getOpType() == "Copy")
            {
                opIterator->setInputTensor(previousOp->getInputTensor(0), 0, false);
                om.defineFlow(previousOp->getInputTensor(0),opIterator , 0);
                removeCopySet.insert(previousOp->getName());
            }
        }
    }
    for (auto& opName:removeCopySet)
        om.removeOp(om.getOp(opName));
}
