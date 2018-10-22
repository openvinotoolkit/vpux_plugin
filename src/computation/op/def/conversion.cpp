#include "include/mcm/computation/op/def/conversion.hpp"

mv::op::Conversion::Conversion(const std::string &name, mv::Order targetOrder):
ComputationOp(mv::OpType::Conversion, name),
SinkOp(mv::OpType::Conversion, 1, name),
SourceOp(mv::OpType::Conversion, 1, name)
{
    set<Order>("order", targetOrder);
    set<bool>("executable", true);
}

mv::Tensor mv::op::Conversion::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);

    //Target mv::Order handled here
    return Tensor(name_ + ":0", input->getShape(), input->getDType(), get<mv::Order>("target_mv::Order"));

}

bool mv::op::Conversion::isHardwarizeable(mv::json::Object&)
{
    return false;
}
