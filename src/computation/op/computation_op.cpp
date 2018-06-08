#include "include/fathom/computation/op/computation_op.hpp"

mv::allocator::map<mv::string, mv::size_type> mv::ComputationOp::idDict_;

mv::ComputationOp::ComputationOp(const string &opType, const string &name) :
ComputationElement(opType + "_" + [&name, &opType]() -> string { if (name.empty()) return Printable::toString(idDict_[opType]++); else return name; }())
{
    logger_.log(Logger::MessageType::MessageDebug, "Defined computation op " + toString());
    addAttr("opType", AttrType::StringType, opType);
}

mv::ComputationOp::~ComputationOp()
{

}

/*mv::ComputationOp::ComputationOp(const ComputationOp &other) :
ComputationElement(other),
dType_(other.dType_),
order_(other.order_),
inputShape_(other.inputShape_),
outputShape_(other.outputShape_)
{

}*/

bool mv::ComputationOp::validOutputDef_()
{

    if (!hasInputDef())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + "' because of undefined input/inputs");
        return false;
    }

    return true;

}

mv::string mv::ComputationOp::getOpType() const 
{
    return getAttr("opType").getContent<string>();
}

mv::string mv::ComputationOp::toString() const
{
    return getAttr("opType").getContentStr() + " '" + name_ + "' " + ComputationElement::toString();
}

mv::TensorContext::TensorIterator mv::ComputationOp::getInput(byte_type idx)
{
    return mv::TensorContext::TensorIterator();
}

mv::TensorContext::TensorIterator mv::ComputationOp::getOutput()
{
    return mv::TensorContext::TensorIterator();
}

bool mv::ComputationOp::setInput(TensorContext::TensorIterator &tensor, byte_type idx)
{
    return false;
}

bool mv::ComputationOp::setOutput(TensorContext::TensorIterator &tensor)
{
    return false;
}

bool mv::ComputationOp::hasInputDef()
{
    return false;
}

bool mv::ComputationOp::hasInputDef(byte_type idx)
{
    return false;
}

mv::byte_type mv::ComputationOp::inputSlots()
{
    return 0;
}

mv::string mv::ComputationOp::getOutputName() const
{
    return "";
}

bool mv::ComputationOp::operator==(const ComputationOp &other) const
{
    return getName() == other.getName();
}
