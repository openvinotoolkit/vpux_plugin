#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/target/keembay/runtime_model/runtime_model.hpp"

static void generateBlobKeembayFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&td, mv::Element& passDesc, mv::json::Object& compOutput);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(GenerateBlobKeembay)
        .setFunc(generateBlobKeembayFcn)
        .setDescription(
            "Generates an executable blob file for Keembay"
        );

    }

}

void generateBlobKeembayFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::json::Object& compOutput)
{   
    mv::RuntimeModel rm;
    rm.buildGraphFileT(model, passDesc);

    if (!passDesc.hasAttr("output"))
        return;

    rm.serialize(passDesc.get<std::string>("output"));
}
