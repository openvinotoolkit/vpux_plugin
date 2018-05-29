#include "include/fathom/computation/model/model.hpp"

mv::allocator mv::ComputationModel::allocator_;

mv::ComputationModel::ComputationModel(Logger &logger) : 
opsGraph_(allocator_.make_owner<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedTensor>, ModelTensor::TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedTensor>, ModelTensor::TensorOrderComparator>()),
groups_(allocator_.make_set<allocator::owner_ptr<ComputationGroup>, ComputationGroup::GroupOrderComparator>()),
logger_(logger),
dataOpEnd_(dataGraph_.node_end()),
dataFlowEnd_(dataGraph_.edge_end()),
controlOpEnd_(controlGraph_.node_end()),
controlFlowEnd_(controlGraph_.edge_end())
{

}

mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime) :
opsGraph_(allocator_.make_owner<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
flowTensors_(allocator_.make_set<allocator::owner_ptr<UnpopulatedTensor>, ModelTensor::TensorOrderComparator>()),
parameterTensors_(allocator_.make_set<allocator::owner_ptr<PopulatedTensor>, ModelTensor::TensorOrderComparator>()),
groups_(allocator_.make_set<allocator::owner_ptr<ComputationGroup>, ComputationGroup::GroupOrderComparator>()),
defaultLogger_(allocator_.make_owner<StdOutLogger>(verboseLevel, logTime)),
logger_(*defaultLogger_),
dataOpEnd_(dataGraph_.node_end()),
dataFlowEnd_(dataGraph_.edge_end()),
controlOpEnd_(controlGraph_.node_end()),
controlFlowEnd_(controlGraph_.edge_end())
{

}


mv::ComputationModel::ComputationModel(const ComputationModel &other) :
opsGraph_(other.opsGraph_),
dataGraph_(other.dataGraph_),
controlGraph_(other.controlGraph_),
flowTensors_(other.flowTensors_),
parameterTensors_(other.parameterTensors_),
groups_(other.groups_),
logger_(other.logger_),
input_(other.input_),
output_(other.output_),
lastOp_(other.lastOp_),
dataOpEnd_(other.dataOpEnd_),
dataFlowEnd_(other.dataFlowEnd_),
controlOpEnd_(other.controlOpEnd_),
controlFlowEnd_(other.controlFlowEnd_)
{

}

mv::ComputationModel::~ComputationModel()
{

}

bool mv::ComputationModel::isValid() const
{
    return !dataGraph_.disjoint() && input_ != dataOpEnd_ && output_ != dataOpEnd_;
}

mv::GroupContext::GroupIterator mv::ComputationModel::addGroup(const string &name)
{
    
    
    
    if (getGroup(name) == groupEnd())
    {
        
        auto result = groups_->insert(allocator_.make_owner<ComputationGroup>(logger_, name));
        if (result.second)
            return result.first;
        
        return groups_->end();
        
    }

    return groups_->end();

}

bool mv::ComputationModel::hasGroup(const string &name)
{

    if (getGroup(name) != groupEnd())
    {
        return true;
    }

    return false;

}

mv::GroupContext::GroupIterator mv::ComputationModel::getGroup(const string &name)
{

    allocator::owner_ptr<ComputationGroup> searchGroup = allocator_.make_owner<ComputationGroup>(logger_, name);
    return groups_->find(searchGroup);

}


mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement_(allocator::owner_ptr<ComputationElement> newElement, GroupContext::GroupIterator &group)
{
    if (group != groupEnd())
    {
        return group->addElement(newElement);
    }

    return group->end();
    
}

mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement(DataContext::OpListIterator &newElement, GroupContext::GroupIterator &group)
{

    allocator::owner_ptr<ComputationOp> ptr = newElement;
    return addGroupElement_(ptr, group);

}

mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement(DataContext::FlowListIterator &newElement, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<DataFlow> ptr = newElement;
    return addGroupElement_(ptr, group);
}  

mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement(ControlContext::OpListIterator &newElement, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ComputationOp> ptr = newElement;
    return addGroupElement_(ptr, group);
}

mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement(ControlContext::FlowListIterator &newElement, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ControlFlow> ptr = newElement;
    return addGroupElement_(ptr, group);
}

mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement(GroupContext::GroupIterator &newElement, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ComputationGroup> ptr = newElement;
    return addGroupElement_(ptr, group);
}

mv::GroupContext::GroupIterator mv::ComputationModel::groupBegin()
{
    return groups_->begin();
}

mv::GroupContext::GroupIterator mv::ComputationModel::groupEnd()
{
    return groups_->end();
}

mv::GroupContext::MemberIterator mv::ComputationModel::memberBegin(GroupContext::GroupIterator &group)
{

    if (group != groupEnd())
    {
        return group->begin();
    }
    
    return GroupContext::MemberIterator();

}

mv::GroupContext::MemberIterator mv::ComputationModel::memberEnd(GroupContext::GroupIterator &group)
{

    if (group != groupEnd())
    {
        return group->end();
    }
    
    return GroupContext::MemberIterator();

}

mv::Logger &mv::ComputationModel::logger()
{

    return logger_;

}