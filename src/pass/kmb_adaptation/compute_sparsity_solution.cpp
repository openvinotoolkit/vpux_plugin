#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include <regex>

static void computeSparsitySolutionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ComputeSparsitySolution)
        .setFunc(computeSparsitySolutionFcn)
        .setDescription(
            "This pass predicts from who the unpopulated sparsity will be solved runtime/compiler."
        );
    }
}

void computeSparsitySolutionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    //IDU OF z-major Conv supports sparsity, so take all the input tensors of convs,
    //see where they are located, if they are on DDR and they need sparsity mark them
    //cause their sparsity is going to be solved from populated pass sparse. even if they
    //are unpopulated

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto convOps = om.getOps("Conv");
    auto globalParams = model.getGlobalConfigParams();
    auto referenceDevice = globalParams->get<std::string>("referenceDevice");

    for (auto &convOp : convOps) {
        if (convOp->hasAttr("floatPrecision") &&
            convOp->get<bool>("floatPrecision") &&
            referenceDevice == "A0" &&
            (!convOp->hasAttr("inputActivationSparsity") ||
            !convOp->get<bool>("inputActivationSparsity")))
        {
            convOp->set<bool>("activationSparsityCompilerSolving", true);
            convOp->set<bool>("inputActivationSparsity", true);
        }
    }
}
