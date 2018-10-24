#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/computation/model/computation_model.hpp"

mv::ModelElement::ModelElement(ComputationModel& model, const std::string& name) :
Element(name),
model_(model)
{

}

mv::ModelElement::~ModelElement()
{

}

mv::json::Value mv::ModelElement::toJSON() const
{
    auto val = Element::toJSON();
    val["model"] = model_.getName();
    return val;
}

std::string mv::ModelElement::getLogID() const
{
    return "ModelElement:" + getName();
}