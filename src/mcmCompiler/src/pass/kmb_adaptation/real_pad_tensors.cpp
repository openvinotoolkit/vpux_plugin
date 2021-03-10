#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/tensor/shape.hpp"


static void alignUnpopulatedTensorsFunc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void alignPopulatedTensorsFunc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void alignWeightsTensor(mv::OpModel& om, const mv::Data::TensorIterator &weightsTensor, mv::Shape alignedShape);
static void cropOrPadFinalOutputFunc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void alignBiasTensor(mv::Data::OpListIterator &opIt, const mv::Data::TensorIterator biasTensor, unsigned biasTensorSizePadded, mv::DataModel dm);
static void addAlignOpForInputTensorsFunc(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeCropAlignInCMXFunc(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static mv::Data::OpListIterator fuseCropAlign(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt);
static void addCropNode(mv::OpModel& om, mv::Data::OpListIterator& opIt, mv::Data::TensorIterator& outputTensor, std::size_t& outputTensorChannels);
void alignInputForChannelMajorConvolution(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AlignUnpopulatedTensors)
            .setFunc(alignUnpopulatedTensorsFunc);

        MV_REGISTER_PASS(AlignPopulatedTensors)
            .setFunc(alignPopulatedTensorsFunc)
            .setDescription(
                "Aligns I/O channels involved in DPUTask to 16");
        MV_REGISTER_PASS(AddAlignOpForInputTensors)
            .setFunc(addAlignOpForInputTensorsFunc)
            .setDescription(
                "Add implicit Align Op for input tensors where needed");
        MV_REGISTER_PASS(RemoveCropAlignInCMX)
            .setFunc(removeCropAlignInCMXFunc)
            .setDescription(
                "Remove Redundant Crop-Align when they are both in CMX");
        MV_REGISTER_PASS(CropOrPadFinalOutput)
            .setFunc(cropOrPadFinalOutputFunc)
            .setDescription(
                "Add/Remove implicit Crop Op for final Op based on padOutput value");
    }
}

// We can't use the registry at this point
// Since we only change activation tensor

// NOTE: What happens with slices?
void propagateShapeChange(mv::OpModel& om, const std::string& flowStr)
{
    auto flow = om.getDataFlow(flowStr);
    auto sink = flow.sink();

    std::string opType = sink->getOpType();

    if(opType == "DPUTask")
        opType = sink->get<std::string>("taskOp");

// Below check for Conv ensures weight (input channel) alignment for the case of Convs that follow non-Conv DPU layers
// and need weight alignment
    if(opType == "Conv")
        sink->set<bool>("alignment", true);

    if (opType == "Eltwise" ||
        opType == "DepthwiseConv" || opType == "MaxPool")
    {
        auto inputTensor = flow->getTensor();
        auto inputShape = inputTensor->getShape();

        auto outputTensor = sink->getOutputTensor(0);
        auto outputShape = outputTensor->getShape();

        sink->set<bool>("alignment", true);

        // If for whatever reason we pass through this tensor more than once, we
        // don't want to overwrite the original dimensions
        if(!outputTensor->hasAttr("oldDimensions"))
            outputTensor->set<mv::Shape>("oldDimensions", outputTensor->getShape());

        outputTensor->setShape({outputShape[mv::IO_WIDTH_DIMENSION], outputShape[mv::IO_HEIGHT_DIMENSION], inputShape[mv::IO_CHANNEL_DIMENSION], outputShape[mv::IO_BATCH_DIMENSION]});
        for (const auto& flowName : outputTensor->getFlowNames())
            propagateShapeChange(om, flowName);

        addCropNode(om, sink, outputTensor, outputShape[mv::IO_CHANNEL_DIMENSION]);
    }
}

void addCropNode(mv::OpModel& om, mv::Data::OpListIterator& opIt, mv::Data::TensorIterator& outputTensor, std::size_t& outputTensorChannels)
{
    auto cropOpName = outputTensor->getName() + "_crop";

    // check if already there's a crop for output tensor
    if (om.checkOp(cropOpName))
       return;

    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

    auto sourceFlowStart = opIt.leftmostOutput();

    for (mv::Data::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
        flowsToRemove.push_back(sinkFlow);
    }

    auto quantParams = outputTensor->getQuantParams();

    auto croppedTensor = om.crop(cropOpName,
                        outputTensor,
                        outputTensorChannels,
                        mv::IO_CHANNEL_DIMENSION);
    croppedTensor->setQuantParams(quantParams);
    croppedTensor->set<bool>("alignment", true);//TODO remove this, just for testing now
    auto cropOp = om.getOp(cropOpName);
    cropOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
    if (opIt->hasAttr("splitStrategy"))
        cropOp->set<std::string>("splitStrategy", opIt->get<std::string>("splitStrategy"));


    for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
    {
        om.undefineFlow(flowsToRemove[flowIdx]);
    }
    for(unsigned op = 0 ; op < opsToLink.size(); ++op)
    {
        opsToLink[op]->setInputTensor(croppedTensor, inputSlots[op], false);
        om.defineFlow(croppedTensor, opsToLink[op], inputSlots[op]);
    }
}

void cropOrPadFinalOutputFunc(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto globalConfigParams = model.getGlobalConfigParams();

    auto padOutput = globalConfigParams->hasAttr("PadOutput") ? globalConfigParams->get<bool>("PadOutput") : false;
    auto outputOp = om.getOutput();
    auto inputTensor = outputOp->getInputTensor(0);
    auto parentOpIt = om.getSourceOp(inputTensor);
    //Note: We should not always have a Crop Operation (case. 3 Input Channels aligns->alignedOutputChannels)
    if (padOutput)
    {
        //remove Crop layer if it's there
        if (parentOpIt->getOpType() == "Crop")
        {
            auto cropParentOpIt = om.getSourceOp(parentOpIt->getInputTensor(0));
            fuseCropAlign(cropParentOpIt, cropParentOpIt->getOutputTensor(0), om, parentOpIt);
        }
    }
    else
    {
        //make sure there's a crop layer
        if ((parentOpIt->hasAttr("alignment") && parentOpIt->getOpType() != "Crop") ||
            (parentOpIt->hasAttr("alignWidth") && parentOpIt->getOpType() != "Crop"))
        {
            if (inputTensor->hasAttr("oldDimensions"))
            {
                auto oldDimensions = inputTensor->get<mv::Shape>("oldDimensions");
                addCropNode(om, parentOpIt, inputTensor, oldDimensions[mv::IO_CHANNEL_DIMENSION]);
            }
        }
    }

}

mv::Data::OpListIterator fuseCropAlign(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt)
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
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}

void removeCropAlignInCMXFunc(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto globalConfigParams = model.getGlobalConfigParams();
    auto cropOps = om.getOps("Crop");

    for(auto vecIt = cropOps.begin(); vecIt != cropOps.end(); ++vecIt)
    {
        auto layer = *vecIt;

        auto outputTensor = layer->getOutputTensor(0);
        auto outputLocation = outputTensor->get<mv::Tensor::MemoryLocation>("Location");
        auto removeCrop = true;
        auto inputTensor = layer->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(inputTensor);
        for (const auto& flowName : outputTensor->getFlowNames())
        {
            auto sink = om.getDataFlow(flowName).sink();

            std::string opType = sink->getOpType();
            if (opType == "Align" &&
                sink->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location") == outputLocation)
            {
                fuseCropAlign(parentOpIt, parentOpIt->getOutputTensor(0), om, sink);
            }
            else
            {
                //at least one flow doesnt go to Align, we have to keep the crop
                removeCrop = false;
            }
        }
        if (removeCrop)
            om.removeOp(layer);

    }
}

void alignInputForChannelMajorConvolution(mv::ComputationModel& model, mv::Data::OpListIterator& opIt)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    const int tensorWidthMultiple = 16;

    auto inputTensor = opIt->getInputTensor(0);
    auto parentOpIt = om.getSourceOp(inputTensor);
    // NOTE: padding with PaddingConcat if padding right padding[1] != 0 
    // Align op extends the tensor width dimenstion and the buffer size, but does NOT "flush" the data from that address
    // with padding right the padded line will have random data creating sporadic different results. Resolved by PaddingConcat
    // scheduling a DMA with zero points to the aligned buffer before the input DMA. This could've been done simply with just an
    // extra DMA scheduled before input but PaddingConcat produces a cleaner model.
    auto special_padding_case = opIt->hasAttr("padding") && opIt->get<std::array<unsigned short, 4>>("padding")[1] != 0;
    // currently implemented for SOHOverlapped, regression issues with other strategies visible in
    // ADK_FP16-INT8_ModelE (FPS) AND por_caffe2_FP16-INT8_vehicle-attributes-recognition-barrier-0042 (accuracy)
    // TODO: implement for other strategies
    // auto soho = parentOpIt->hasAttr("splitStrategy") && parentOpIt->get<std::string>("splitStrategy") == "SplitOverHOverlapped";
    // special_padding_case = special_padding_case & soho;

    if ((parentOpIt->getOpType() != "Align" || parentOpIt->getOpType() != "PaddingConcat") 
        && inputTensor->getShape()[mv::IO_WIDTH_DIMENSION] % tensorWidthMultiple != 0)
    {
        inputTensor->set<bool>("alignWidth", true);
        opIt->set<bool>("alignWidth", true);

        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;
        mv::Data::TensorIterator alignToLink;

        auto sourceFlowStart = parentOpIt.leftmostOutput();

        for (mv::Data::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            flowsToRemove.push_back(sinkFlow);
        }

        auto quantParams = inputTensor->getQuantParams();
        auto outputTensorMemoryLocation = mv::Tensor::MemoryLocation::NNCMX;

        if (!special_padding_case)
        {
            auto alignOpName = inputTensor->getName() + "_align";

            auto alignedTensor = om.align(alignOpName,
                                    inputTensor,
                                    mv::IO_WIDTH_DIMENSION,
                                    tensorWidthMultiple);
            alignedTensor->setQuantParams(quantParams);
            // This will work because of the implicit flows compensatory DMA passes
            //auto outputTensorMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            alignedTensor->set<mv::Tensor::MemoryLocation>("Location", outputTensorMemoryLocation);


            alignedTensor->set<bool>("alignWidth", true);

            auto alignOp = om.getOp(alignOpName);
            alignToLink = alignedTensor;

            alignOp->set<unsigned>("opId", parentOpIt->get<unsigned>("opId"));

            if (opIt->hasAttr("padding"))
            {
                alignOp->set<std::array<unsigned short, 4>>("padding", opIt->get<std::array<unsigned short, 4>>("padding"));
            }
            if (parentOpIt->hasAttr("splitStrategy"))
            {
                alignOp->set<std::string>("splitStrategy", parentOpIt->get<std::string>("splitStrategy"));
            }

            if (inputTensor->hasAttr("splitStrategy"))
            {
                alignOp->getOutputTensor()[0]->set<std::string>("splitStrategy", inputTensor->get<std::string>("splitStrategy"));
            }
        }
        else
        {
            auto input_width = inputTensor->getShape()[mv::IO_WIDTH_DIMENSION];
            auto width_aligned = tensorWidthMultiple - input_width % tensorWidthMultiple;
            auto pad_shape = mv::Shape({width_aligned, inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION],
                inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], inputTensor->getShape()[mv::IO_BATCH_DIMENSION]});
            std::vector<int64_t> pad_align((size_t)pad_shape.totalSize(), inputTensor->getQuantParams().getZeroPoint()[0]);

            auto const_name = parentOpIt->getName() + "_padding_const";
            auto concat_name = parentOpIt->getName() + "_padding_concat";

            auto const_align = om.constantInt(const_name, pad_align, pad_shape, inputTensor->getDType(), inputTensor->getOrder());
            auto concat_align = om.implicitConcat(concat_name, {inputTensor, const_align}, "W");

            const_align->setQuantParams(quantParams);
            concat_align->setQuantParams(quantParams);
            const_align->set<bool>("paddingConcatAlignment", true);
            concat_align->set<bool>("paddingConcatAlignment", true);
            concat_align->set<mv::Tensor::MemoryLocation>("Location", outputTensorMemoryLocation);

            auto concat_align_op = om.getOp(const_name);
            auto const_align_op = om.getOp(concat_name);
            alignToLink = concat_align;

            concat_align_op->set<unsigned>("opId", parentOpIt->get<unsigned>("opId"));
            const_align_op->set<unsigned>("opId", parentOpIt->get<unsigned>("opId"));

            if (parentOpIt->hasAttr("splitStrategy"))
            {
                auto padding_startegy = parentOpIt->get<std::string>("splitStrategy");
                concat_align_op->set<std::string>("splitStrategy", padding_startegy);
                const_align_op->set<std::string>("splitStrategy", padding_startegy);
                concat_align_op->getOutputTensor()[0]->set<std::string>("splitStrategy", padding_startegy);
                const_align_op->getOutputTensor()[0]->set<std::string>("splitStrategy", padding_startegy);
            }
        }

        for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
        {
            om.undefineFlow(flowsToRemove[flowIdx]);
        }

        for(unsigned op = 0 ; op < opsToLink.size(); ++op)
        {
            opsToLink[op]->setInputTensor(alignToLink, inputSlots[op], false);
            opsToLink[op]->set<bool>("alignWidth", true);
            om.defineFlow(alignToLink, opsToLink[op], inputSlots[op]);
        }
    }
}

void addAlignOpForInputTensorsFunc(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto globalConfigParams = model.getGlobalConfigParams();
    int pad = globalConfigParams->hasAttr("VPU2ChannelPadding") ? globalConfigParams->get<int>("VPU2ChannelPadding") : 16;
    auto dpuTasks = om.getOps("DPUTask");

    for(auto vecIt = dpuTasks.begin(); vecIt != dpuTasks.end(); ++vecIt)
    {
        auto opIt = *vecIt;
        auto taskOp = opIt->get<std::string>("taskOp");
        if(taskOp == "Conv" || taskOp == "DepthwiseConv" || taskOp == "MaxPool" ||
            taskOp == "Eltwise" ||
            (taskOp == "ChannelMajorConvolution" && td.getTarget() == mv::Target::ma3720)) //channel major as zmajor in MTL
        {
            auto numInputs = 1;
            if (taskOp == "Eltwise")
                numInputs ++;
            for (auto i = 0; i < numInputs; i++)
            {
                auto inputTensor = opIt->getInputTensor(i);
                auto parentOpIt = om.getSourceOp(inputTensor);
                if (inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION] % pad != 0)
                {
                    inputTensor->set<bool>("alignment", true);
                    opIt->set<bool>("alignment", true);

                    std::vector<mv::Data::OpListIterator> opsToLink;
                    std::vector<std::size_t> inputSlots;
                    std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

                    for (auto sinkFlow = parentOpIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
                    {
                        if (sinkFlow.sink()->getOpType() != "DPUTask")
                            continue;

                        opsToLink.push_back(sinkFlow.sink());
                        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
                        flowsToRemove.push_back(sinkFlow);
                    }

                    auto alignOpName = inputTensor->getName() + "_align";
                    auto quantParams = inputTensor->getQuantParams();

                    auto alignedTensor = om.align(alignOpName,
                                        inputTensor,
                                        mv::IO_CHANNEL_DIMENSION,
                                        pad);
                    alignedTensor->setQuantParams(quantParams);
                    alignedTensor->set<bool>("alignment", true);//TODO remove this, just for testing now
                    // This will work because of the implicit flows compensatory DMA passes

                    //If ParentOp memory location of Align is in DDR, then Align can get any strategy and should get strategy of the child Op instead of parent
                    //Enables RetinaFace compilation, applicable to other networks too
                    auto outputTensorMemoryLocation = mv::Tensor::MemoryLocation::NNCMX;
                    alignedTensor->set<mv::Tensor::MemoryLocation>("Location", outputTensorMemoryLocation);
                    auto alignOp = om.getOp(alignOpName);
                    alignOp->set<unsigned>("opId", parentOpIt->get<unsigned>("opId"));

                    auto parentMemoryLocation = parentOpIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
                    if(parentOpIt->isImplicit() && parentMemoryLocation == mv::Tensor::MemoryLocation::DDR)
                        if (opIt->hasAttr("splitStrategy"))
                            alignOp->set<std::string>("splitStrategy", opIt->get<std::string>("splitStrategy"));
                        else if (parentOpIt->hasAttr("splitStrategy"))
                                alignOp->set<std::string>("splitStrategy", parentOpIt->get<std::string>("splitStrategy"));

                    for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
                    {
                        om.undefineFlow(flowsToRemove[flowIdx]);
                    }
                    for(unsigned op = 0 ; op < opsToLink.size(); ++op)
                    {
                        opsToLink[op]->setInputTensor(alignedTensor, inputSlots[op], false);
                        opsToLink[op]->set<bool>("alignment", true);
                        if (opsToLink[op]->getOpType() == "Copy")
                            opsToLink[op]->redefineOutputTensors();
                        om.defineFlow(alignedTensor, opsToLink[op], inputSlots[op]);

                        // If Copy follows Align, cascade aligned output shape
                        if (opsToLink[op]->getOpType() == "Copy")
                            opsToLink[op]->getOutputTensor(0)->setShape(alignedTensor->getShape());

                    }
                }
            }
        }
    }
}

//NOTE: REAL PADDING IN THE UNALIGNED TENSORS
void alignUnpopulatedTensorsFunc(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto dpuTasks = om.topologicalSort();
    for(auto vecIt = dpuTasks.begin(); vecIt != dpuTasks.end(); ++vecIt)
    {
        auto opIt = *vecIt;
        if(opIt->getOpType() != "DPUTask")
            continue;

        auto taskOp = opIt->get<std::string>("taskOp");
        auto outputTensor = opIt->getOutputTensor(0);
        auto outputTensorShape = outputTensor->getShape();
        auto outputTensorChannels = outputTensorShape[mv::IO_CHANNEL_DIMENSION];
        //auto opStrategy = opIt->get<std::string>("splitStrategy");
        if (outputTensorChannels % 16 != 0)
        {
            opIt->set<bool>("alignment", true);
            outputTensor->set<bool>("alignment", true);
            std::size_t outputChannelsPadded = mv::round_up(outputTensorChannels, 16);

            // If for whatever reason we pass through this tensor more than once, we
            // don't want to overwrite the original dimensions
            if(!outputTensor->hasAttr("oldDimensions"))
                outputTensor->set<mv::Shape>("oldDimensions", outputTensor->getShape());

            outputTensor->setShape(mv::Shape({outputTensorShape[mv::IO_WIDTH_DIMENSION], outputTensorShape[mv::IO_HEIGHT_DIMENSION],
                                              outputChannelsPadded, outputTensorShape[mv::IO_BATCH_DIMENSION]}));

            for (const auto& flowStr: outputTensor->getFlowNames())
                propagateShapeChange(om, flowStr);
            addCropNode(om, opIt, outputTensor, outputTensorChannels);
        }

       if(taskOp == "ChannelMajorConvolution")
            alignInputForChannelMajorConvolution(model, opIt);
    }
}

void alignPopulatedTensorsFunc(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);


    for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
    {
        if (layer->hasAttr("alignment") && layer->hasAttr("hasWeights") && layer->get<bool>("hasWeights"))
        {
            auto inputTensor = layer->getInputTensor(0);
            auto weightsTensor = layer->getInputTensor(1);
            auto outputTensor = layer->getOutputTensor(0);
            auto weightsTensorShape = weightsTensor->getShape();
            auto inputTensorShape = inputTensor->getShape();
            auto outputTensorShape = outputTensor->getShape();

            mv::Shape alignedShape;
            //NOTE: only Convs have weights (=) alignment
            auto taskOp = layer->get<std::string>("taskOp");
            if (taskOp == "ChannelMajorConvolution" && td.getTarget() == mv::Target::ma3720)
                continue;

            std::size_t outputChannelsPadded = outputTensorShape[mv::IO_CHANNEL_DIMENSION];

            if (taskOp == "Conv" ||
                (taskOp == "ChannelMajorConvolution" && td.getTarget() != mv::Target::ma3720))
                alignedShape = mv::Shape({weightsTensorShape[mv::KERNEL_WIDTH], weightsTensorShape[mv::KERNEL_HEIGHT],
                                                    inputTensorShape[mv::IO_CHANNEL_DIMENSION], outputTensorShape[mv::IO_CHANNEL_DIMENSION]});

            else if (taskOp == "DepthwiseConv")
            {
                alignedShape = mv::Shape({weightsTensorShape[mv::KERNEL_WIDTH], weightsTensorShape[mv::KERNEL_HEIGHT],
                                                            inputTensorShape[mv::IO_CHANNEL_DIMENSION], 1});
                outputChannelsPadded = inputTensorShape[mv::IO_CHANNEL_DIMENSION];
            }

            alignWeightsTensor(om, weightsTensor, alignedShape);
            if(layer->hasAttr("bias"))
            {
                auto biasTensorName = layer->get<std::string>("bias");
                auto biasTensor = om.getTensor(biasTensorName);
                alignBiasTensor(layer, biasTensor, outputChannelsPadded, dm);
            }
        }
    }

}

static void alignWeightsTensor(mv::OpModel& om, const mv::Data::TensorIterator &weightsTensor, mv::Shape alignedShape)
{
    auto weightsTensorOrder = weightsTensor->getOrder();
    auto weightsTensorDType = weightsTensor->getDType();
    auto weightsTensorShape = weightsTensor->getShape();
    int64_t zeroPoint = 0;
    mv::QuantizationParams weightsTensorQuantizationParams = mv::QuantizationParams::empty();

    if (weightsTensorShape[mv::KERNEL_OUTPUT_CHANNELS] == alignedShape[mv::KERNEL_OUTPUT_CHANNELS] &&
        weightsTensorShape[mv::KERNEL_INPUT_CHANNELS] == alignedShape[mv::KERNEL_INPUT_CHANNELS])
            return;

    if (weightsTensor->isQuantized())
    {
        weightsTensorQuantizationParams = weightsTensor->get<mv::QuantizationParams>("quantParams");
        zeroPoint = weightsTensorQuantizationParams.getZeroPoint()[0];
    }

    auto newData = std::vector<mv::DataElement>(alignedShape.totalSize(), mv::DataElement(weightsTensorDType.isDoubleType(), zeroPoint));
    auto constantOp = om.getSourceOp(weightsTensor);
    auto outFlows = mv::getOutputDataFlow(om, constantOp, false);
    mv::Data::TensorIterator newKernel = om.constantDataElement(mv::createAlignConstantName(constantOp->getName()), newData, alignedShape, weightsTensorDType, weightsTensorOrder);
    newKernel->setQuantParams(weightsTensorQuantizationParams);

    //DO NOT CHANGE THE LIMITS OF THE LOOP! THERE IS A REASON WHY IT'S DONE LIKE THIS AND NOT USING THE AUXILIARY VARIABLES
    for(unsigned oc = 0; oc < weightsTensorShape[mv::KERNEL_OUTPUT_CHANNELS]; ++oc)
        for(unsigned ic = 0; ic < weightsTensorShape[mv::KERNEL_INPUT_CHANNELS]; ++ic)
            for(unsigned kw = 0; kw < weightsTensorShape[mv::KERNEL_WIDTH]; ++kw)
                for(unsigned kh = 0; kh < weightsTensorShape[mv::KERNEL_HEIGHT]; ++kh)
                    newKernel->at({kw,kh,ic,oc}) = weightsTensor->at({kw,kh,ic,oc});

    om.getSourceOp(newKernel)->set<unsigned>("opId", constantOp->get<unsigned>("opId"));

    om.removeOp(constantOp);
    mv::setOutputDataFlow(om, newKernel, outFlows);
}

static void alignBiasTensor(mv::Data::OpListIterator &opIt, const mv::Data::TensorIterator biasTensor, unsigned biasTensorSizePadded, mv::DataModel dm)
{
    //Bias case is easier since it is 1D
    auto biasTensorDType = biasTensor->getDType();
    auto biasTensorSize = biasTensor->getShape()[0];


    auto biasTensorName = opIt->get<std::string>("bias");
    if(biasTensorSizePadded != biasTensorSize)
    {
        int64_t zeroPoint = 0;
        mv::QuantizationParams biasTensorQuantizationParams = mv::QuantizationParams::empty();

        if (biasTensor->isQuantized()) {
            biasTensorQuantizationParams = biasTensor->get<mv::QuantizationParams>("quantParams");
            zeroPoint = biasTensorQuantizationParams.getZeroPoint()[0];
        }

        auto newData = std::vector<mv::DataElement>(biasTensorSizePadded, mv::DataElement(biasTensorDType.isDoubleType(), zeroPoint));
        auto newBiasTensor = dm.defineTensor(mv::createAlignConstantName(biasTensorName), {biasTensorSizePadded}, biasTensorDType, mv::Order("W"), newData);
        newBiasTensor->setQuantParams(biasTensorQuantizationParams);

        for(unsigned i = 0; i < biasTensorSize; ++i)
            newBiasTensor->at({i}) = biasTensor->at({i});

        dm.undefineTensor(biasTensorName);
        opIt->erase("bias");
        opIt->set<std::string>("bias", newBiasTensor->getName());

        //check for other ops with the same bias tensor, and upate teh attribute
        mv::OpModel om(dm);
        auto dpuTasks = om.getOps("DPUTask");
        for(auto layer = dpuTasks.begin(); layer != dpuTasks.end(); ++layer)
        {
            auto updateOpIt = *layer;
            if(updateOpIt->hasAttr("bias") && updateOpIt->get<std::string>("bias") == biasTensorName)
            {
                updateOpIt->erase("bias");
                updateOpIt->set<std::string>("bias", newBiasTensor->getName());
            }
        }
    }
}
