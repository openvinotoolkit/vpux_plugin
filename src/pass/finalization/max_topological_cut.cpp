#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/koala_graph_scheduler.hpp"
#include <iostream>

static void maxTopologicalCutAndPartialSerialisationPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MaxTopologicalCutAndPartialSerialisation)
        .setFunc(maxTopologicalCutAndPartialSerialisationPass)
        .setDescription(
            "Perform the max topological cut algorithm and partial serialisation (if required) to schedule the DAG."
        );
    }

}


void maxTopologicalCutAndPartialSerialisationPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::json::Object&)
{
    mv::KoalaGraphScheduler flowGraph;
    
    /*Convert to MCM graph to KOALA graph*/
    flowGraph.convertMcMGraphToKoalaGraph(pass, model);

    /*Calculate max topological cut and get the cut edges*/
    auto maxTopologicalCut = flowGraph.calculateMaxTopologicalCut(pass, model);
   
    /*Get CMX memory*/
    auto memDefs = target.memoryDefs();
    auto availableNNCMX = memDefs.find("VPU_CMX_NN")->second.size;

    /*Get the CMX safety factor*/
    std::shared_ptr<mv::Element> returnedParams = model.getGlobalConfigParams();
    double cmxSafetyFactor = returnedParams->get<double>("CMX_memory_overflow_safety_factor");

    /*Get the number of clusters that the VPU supports*/
    auto nceDefs = target.nceDefs();
    auto numberOfVPUClusters = nceDefs.find("Clusters")->second.totalNumber;

    auto cmxMemory = (availableNNCMX / numberOfVPUClusters) * cmxSafetyFactor;
    /*Note available CMX memory is 3760128 /number of supported VPU clusters (always 4)*/
    bool memoryHack = returnedParams->hasAttr("MemoryHack") && returnedParams->get<bool>("MemoryHack");
    if(memoryHack)
    {
        auto compilationClusters = returnedParams->get<int>("Number_of_Clusters");
        cmxMemory = (availableNNCMX / compilationClusters) * cmxSafetyFactor;
    }

    /*Repeat partial serialisation until max topological cut is less than CMX memory*/
    while (maxTopologicalCut.first > cmxMemory) {
        flowGraph.performPartialSerialisation(pass, maxTopologicalCut.second);
        maxTopologicalCut = flowGraph.calculateMaxTopologicalCut(pass, model);
    }
    
    /*Add the partial serialisaion edges to the mcmGraph*/
    flowGraph.insertpartialSerialisationEdgesInMcmGraph(model);
}
