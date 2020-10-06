#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <functional>

void fuseBiasFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel& model, std::string opType);
void fuseUsualPPEFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType);
void fuseMinimumFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, std::string opType);
void fuseMaximumFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, std::string opType);
void fuseEltwiseFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, std::string opType);
void fuseScaleFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, std::string opType);
void fuseBatchNormFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, std::string opType);
static void fusePostOpsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(FusePostOps)
        .setFunc(fusePostOpsFcn)
        .setDescription(
            "Fuses all the ops that will be converted to PPE Tasks and can be handled through hardware. "
            "Scale, Batch Norm from My-X\n"
        );
    }
}

void fusePostOpsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    UNUSED(fuseScaleFcn);
    UNUSED(fuseBatchNormFcn);

    mv::OpModel om(model);
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    std::unordered_map<std::string, std::function<void(mv::Data::OpListIterator &, mv::ComputationModel& , std::string &)>> fuseTaskMap =
                                    {{"Bias", fuseBiasFcn},
                                    {"Sigmoid", fuseUsualPPEFcn},
                                    {"Tanh", fuseUsualPPEFcn},
                                    {"Relu", fuseUsualPPEFcn},
                                    {"LeakyRelu", fuseUsualPPEFcn},
                                    {"Minimum", fuseMinimumFcn},
                                    {"Maximum", fuseMaximumFcn}};

    bool PPEAccuracy = globalParams->hasAttr("PPEAccuracy") ? globalParams->get<bool>("PPEAccuracy") : false;
    if (PPEAccuracy)
    {
        std::vector<mv::Data::OpListIterator> biasOperations = om.getOps("Bias");

        for (auto bias : biasOperations)
            fuseBiasFcn(bias, model, "Bias");

        provideAccuracyinPPEs(model);
        std::vector<std::string> fuse_types = {"Sigmoid", "Tanh", "Relu", "Minimum", "Maximum"};
        std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operationsOfType = om.getOpsOfTypes(fuse_types);

        //NOTE: Iterate the fuse_types vector for correct order reason according to map
        for (auto type = fuse_types.begin(); type != fuse_types.end(); type++)
        {
            auto fuseFunctor = (fuseTaskMap.at(*type));
            for (auto opIt = operationsOfType[*type].begin(); opIt != operationsOfType[*type].end();++opIt)
                fuseFunctor(*opIt, model, *type);
        }
    }
    else
    {
        std::vector<std::string> fuse_types = {"Bias", "Sigmoid", "Tanh", "Relu", "LeakyRelu", "Minimum", "Maximum"};
        std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operationsOfType = om.getOpsOfTypes(fuse_types);

        //NOTE: Iterate the fuse_types vector for correct order reason according to map
        for (auto type = fuse_types.begin(); type != fuse_types.end(); type++)
        {
            auto fuseFunctor = (fuseTaskMap.at(*type));
            for (auto opIt = operationsOfType[*type].begin(); opIt != operationsOfType[*type].end();++opIt)
                fuseFunctor(*opIt, model, *type);
        }
    }
}

mv::Data::OpListIterator linkNewOperationsFuse(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    while(paramOp != om.opEnd())
    {
        if (paramOp->getOpType() == "Constant" || paramOp->getOpType() == "ConstantInt" || paramOp->getOpType() == "ConstantDataElement")
        {
            auto backUp = paramOp;
            ++paramOp;
            om.removeOp(backUp);
        }
        else
            ++paramOp;
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

void fuseBiasFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    if (parentOpIt->getOpType() == "Conv" ||
        parentOpIt->getOpType() == "FullyConnected" ||
        parentOpIt->getOpType() == "DepthwiseConv" || parentOpIt->getOpType() == "Deconv")
    {
        auto bias = *opIt->getInputTensor(1);
        auto biasOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        if (parentOpIt->hasAttr("bias"))
        {
            auto biasTensor = model.getTensor(parentOpIt->get<std::string>("bias"));
            biasTensor->add(bias);
        }
        else
        {
            std::string biasTensorName = mv::createBiasName(parentOpIt->getName());
            mv::Data::TensorIterator biasTensor;
            if (bias.hasAttr("quantParams"))
                biasTensor = dm.defineTensor(mv::Tensor(biasTensorName, bias.getShape(), bias.getDType(), bias.getOrder(), bias.getData(), bias.get<mv::QuantizationParams>("quantParams")) );
            else
                biasTensor = dm.defineTensor(mv::Tensor(biasTensorName, bias.getShape(), bias.getDType(), bias.getOrder(), bias.getData()) );
            om.addAttr(parentOpIt, "bias", biasTensor->getName());
        }
        auto sourceTensor = parentOpIt->getOutputTensor(0);
        opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
        if (biasOutputMemoryLocation.isForced())
        {
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", biasOutputMemoryLocation);
        }
    }
}

void fuseScaleFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    auto scaleOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    if (parentOpIt->getOpType() == "Conv")
    {
        auto scale = *opIt->getInputTensor(1);
        parentOpIt->getInputTensor(1)->multiply(scale);
        if (parentOpIt->hasAttr("bias"))
        {
            auto biasTensor = model.getTensor(parentOpIt->get<std::string>("bias"));
            biasTensor->multiply(scale);
        }
        auto sourceTensor = parentOpIt->getOutputTensor(0);
        opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
        if (scaleOutputMemoryLocation.isForced())
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", scaleOutputMemoryLocation);
    }
}

void fuseUsualPPEFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto ppeOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

    if (opType == "LeakyRelu")
        parentOpIt->set<double>("leakyAlpha", opIt->get<double>("alpha"));
    else if (opIt->hasPWLActivation())
    {
        // Check for fuseable parentOp; else, execute in software
        // Check for (multiple) children, if multiple children, can't fuse into the parents
        auto optype = parentOpIt->getOpType();
        auto nextOps = mv::findSinkLayers(dm, opIt->getInputTensor(0));
        if (!(optype == "Conv" || optype == "DepthwiseConv" || optype == "CMConv") || (nextOps.size() != 1))
        {
            opIt->set<bool>("softwareExecuted", true);
            return;
        }
    }
    std::vector<std::string> postOpTypes;
    if (parentOpIt->hasAttr("postOpTypes"))
        postOpTypes = parentOpIt->get<std::vector<std::string>>("postOpTypes");

    postOpTypes.push_back(opType);
    parentOpIt->set<std::vector<std::string>>("postOpTypes", postOpTypes);

    auto sourceTensor = parentOpIt->getOutputTensor(0);
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (ppeOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", ppeOutputMemoryLocation);
}

void fuseEltwiseFcn(mv::Data::OpListIterator &opIt1, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    std::unordered_map<std::string, std::function<void(mv::Data::OpListIterator &, mv::ComputationModel& , std::string &)>> fuseEltwiseMap =
                                       {{"Minimum", fuseMinimumFcn},
                                        {"Maximum", fuseMaximumFcn},
                                        {"Power", fuseUsualPPEFcn}};

    auto eltwiseType = opIt1->get<std::string>("eltwiseType");
    auto functor = fuseEltwiseMap.find(eltwiseType);
    if(functor != fuseEltwiseMap.end())
        functor->second(opIt1, model, eltwiseType);
}

void fuseMinimumFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);

    auto minimumOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

    double minimumValue = opIt->get<double>("minimum");
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    parentOpIt->set<double>("Minimum", minimumValue);

    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (minimumOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", minimumOutputMemoryLocation);
}

void fuseMaximumFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);

    auto maximumOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

    double maximumValue = opIt->get<double>("maximum");
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    parentOpIt->set<double>("Maximum", maximumValue);

    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (maximumOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", maximumOutputMemoryLocation);
}

void fuseBatchNormFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);
    auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto batchNormName = opIt->getName();
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    auto bnMean = *opIt->getInputTensor(1);
    auto bnVar = *opIt->getInputTensor(2);
    auto bnOffset = *opIt->getInputTensor(3);
    auto bnScale = *opIt->getInputTensor(4);
    double bnEps = opIt->get<double>("eps");
    auto scaleParam = mv::math::divide(bnScale, mv::math::sqrt(mv::math::add(bnVar, bnEps)));
    auto offsetParam = mv::math::subtract(bnOffset, mv::math::multiply(bnMean, scaleParam));
    auto offset = om.constantDataElement(offsetParam.getData(), offsetParam.getShape(), offsetParam.getDType(),
        offsetParam.getOrder(),{{},{},{},{}}, batchNormName + "_offset");

    mv::Data::TensorIterator sourceTensor;

    if (bnMean.getShape().ndims() == 1)
    {
        if (parentOpIt->getOpType() == "Conv")
        {
            parentOpIt->getInputTensor(1)->multiply(scaleParam);
            sourceTensor = parentOpIt->getOutputTensor(0);
        }
        else
        {
            auto scale = om.constantDataElement(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
            sourceTensor = om.scale(opIt->getInputTensor(0), scale);
            parentOpIt = om.getSourceOp(sourceTensor);
        }
    }
    else
    {
        auto scale = om.constantDataElement(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
        sourceTensor = om.eltwise({opIt->getInputTensor(0), scale}, "Multiply");
        parentOpIt = om.getSourceOp(sourceTensor);
    }

    if (offsetParam.getShape().ndims() == 1)
        sourceTensor = om.bias(sourceTensor, offset);
    else
        sourceTensor = om.eltwise({sourceTensor, offset}, "Add");
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (outputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
}
