#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/target/myriadx/nce1_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include <cmath>
#include <numeric>
#include <cmath>

static void generateSchedulingFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void barrierIndexAssignmentFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void updateBarrierRefsFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void updateBarrierRefsIdsFcn(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(GenerateExecutionSchedule)
        .setFunc(generateSchedulingFcn)
        .setDescription(
            "Gathers fields for serialization"
        );

        MV_REGISTER_PASS(BarrierIndexAssignment)
        .setFunc(barrierIndexAssignmentFcn)
        .setDescription(
            "Generates an executable blob file"
        );

        MV_REGISTER_PASS(UpdateBarrierRefs)
        .setFunc(updateBarrierRefsFcn)
        .setDescription(
            "Generates an executable blob file"
        );

        MV_REGISTER_PASS(UpdateBarrierRefsIds)
        .setFunc(updateBarrierRefsIdsFcn)
        .setDescription(
            "Generates an executable blob file"
        );

    }
}

std::vector<std::string> getBarriersProduced(mv::Data::OpListIterator task)
{
    if(task->hasAttr("BarriersProducedByTask"))
        return task->get<std::vector<std::string>>("BarriersProducedByTask");
    else
        return std::vector<std::string>();
}

std::vector<std::string> getBarriersConsumed(mv::Data::OpListIterator task)
{
    if(task->hasAttr("BarriersConsumedByTask"))
        return task->get<std::vector<std::string>>("BarriersConsumedByTask");
    else
        return std::vector<std::string>();
}

std::vector<std::string> getBarriersNeeded(mv::Data::OpListIterator task)
{
    std::vector<std::string> barriersNeeded;

    auto barriersProducedByTask = getBarriersProduced(task);
    for(auto& barrier : barriersProducedByTask)
        barriersNeeded.push_back(barrier);

    auto barriersConsumedByTask = getBarriersConsumed(task);
    for(auto& barrier : barriersConsumedByTask)
        barriersNeeded.push_back(barrier);

    return barriersNeeded;
}

bool isInsertable(mv::ComputationModel& model, const std::string& taskName, const std::unordered_set<std::string>& barriersInMemory)
{
    // A task is insertable if:
    // 1) The barriers it needs are pushable in barriersInMemory without overflowing the 8 barriers in memory limit
    // 2) The barriers it needs are already in barriersInMemory

    auto task = model.getOp(taskName);

    std::vector<std::string> barriersNeeded = getBarriersNeeded(task);

    unsigned barriersInMemorySize = barriersInMemory.size();
    unsigned barriersToInsert = 0;
    for(auto& barrier: barriersNeeded)
        if(!barriersInMemory.count(barrier))
            ++barriersToInsert;

    if(barriersInMemorySize + barriersToInsert > 8)
        return false;
    else
        return true;
}

std::vector<std::string> updateBarriersInMemoryForInsertion(mv::ComputationModel& model, const std::string& taskName, std::unordered_set<std::string>& barriersInMemory, std::unordered_set<std::string>& availableTasks)
{
    // When task is pushed to the queue the following things happen
    // 0) If the barriers involved (produced and consumed) with the task are not in memory, push them to both barriersInMemory and toReturn vector.
    // 1) Find the barriers in memory produced by the task, reduce the number of producers by 1. If any of these barriers reaches zero producers, push its consumers to availableTasks
    // 2) Find the barriers in memory consumed by the task, reduce the number of consumers by 1. If any of these barriers reaches zero consumers, remove it from the barriersInMemory list
    auto task = model.getOp(taskName);
    std::vector<std::string> toReturn;

    std::vector<std::string> barriersNeeded = getBarriersNeeded(task);
    for(auto& barrier : barriersNeeded)
    {
        if(!barriersInMemory.count(barrier))
        {
            toReturn.push_back(barrier);
            barriersInMemory.insert(barrier);
        }
    }

    auto barriersProduced = getBarriersProduced(task);
    for(auto& barrier : barriersProduced)
    {
        auto barrierTask = model.getOp(barrier);
        auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
        physicalBarrier.setNumProducers(physicalBarrier.getNumProducers() - 1);
        if(physicalBarrier.getNumProducers() == 0)
            for(auto& consumer : physicalBarrier.getConsumers())
                availableTasks.insert(consumer);

    }

    auto barriersConsumed = getBarriersConsumed(task);
    for(auto& barrier : barriersConsumed)
    {
        auto barrierTask = model.getOp(barrier);
        auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
        physicalBarrier.setNumConsumers(physicalBarrier.getNumConsumers() - 1);
        if(physicalBarrier.getNumConsumers() == 0)
            barriersInMemory.erase(barrier);

    }

    return toReturn;
}

void updateBarriersInMemoryForRemotion(mv::ComputationModel& model, const std::string& taskName, std::unordered_set<std::string>& barriersInMemory, std::unordered_set<std::string>& availableTasks, std::vector<std::string>& addedBarriers)
{
    // When task is removed from the queue the following things happen
    // 1) Find the barriers produced by the task, increment the number of producers by 1. If the previous number was 0, its consumers must be removed from the list of available tasks
    // 2) Find the barriers in memory consumed by the task, increment the number of consumers by 1. If the previous number was 0, add the barriers to barriersInMemory list
    // 3) For each barrier in addedBarriers, remove it from barriersInMemory

    auto task = model.getOp(taskName);
    auto barriersProduced = getBarriersProduced(task);
    for(auto& barrier : barriersProduced)
    {
        auto barrierTask = model.getOp(barrier);
        auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
        unsigned numProducers = physicalBarrier.getNumProducers();
        physicalBarrier.setNumProducers(numProducers + 1);
        if(numProducers == 0)
            for(auto& consumer : physicalBarrier.getConsumers())
                availableTasks.erase(consumer);

    }

    auto barriersConsumed = getBarriersConsumed(task);
    for(auto& barrier : barriersConsumed)
    {
        auto barrierTask = model.getOp(barrier);
        auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
        unsigned numConsumers = physicalBarrier.getNumConsumers();
        physicalBarrier.setNumConsumers(numConsumers + 1);
        if(numConsumers == 0)
            barriersInMemory.insert(barrier);
    }

    for(auto& barrier : addedBarriers)
        barriersInMemory.erase(barrier);

}



bool generateSchedulingRecursively(mv::ComputationModel& model, std::unordered_set<std::string>& availableTasks, std::vector<std::string>& scheduling, std::unordered_set<std::string>& barriersInMemory)
{
    std::vector<std::string> pushableTasks;

    for(auto& task: availableTasks)
        if(isInsertable(model, task, barriersInMemory))
            pushableTasks.push_back(task);

    for(auto& pushableTask: pushableTasks)
    {
        // Push task into scheduling (and thus removing it from available tasks list)
        scheduling.push_back(pushableTask);
        availableTasks.erase(pushableTask);

        //Update the barriers in memory due to newly inserted task
        auto addedBarriers = updateBarriersInMemoryForInsertion(model, pushableTask, barriersInMemory, availableTasks);

        if(barriersInMemory.empty())
            return true;

        if(generateSchedulingRecursively(model, availableTasks, scheduling, barriersInMemory))
            return true;
        else
        {
            //Can't build a schedule, we have to reset the memory structures as they were
            //before trying the next operation
            scheduling.erase(scheduling.end() - 1);
            availableTasks.insert(pushableTask);
            updateBarriersInMemoryForRemotion(model, pushableTask, barriersInMemory, availableTasks, addedBarriers);
        }
    }

    // If we arrived here, it means we tried all pushable tasks
    // It's impossible to produce a schedule at this point as we can't push nothing
    return false;
}

void generateSchedulingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::ControlModel cm(model);

    auto barrierTasks = model.getOps("BarrierTask");
    unsigned numTasks = 0;

    std::unordered_set<std::string> availableTasks;
    for(auto opIt = cm.opBegin(); opIt != cm.opEnd(); ++opIt)
    {
        if(opIt->getOpType().find("Task") != std::string::npos && opIt->getOpType() != "BarrierTask")
        {
            availableTasks.insert(opIt->getName());
            ++numTasks;
        }
    }

    // Find the tasks that have no barrier dependency, we have to do it through barriers
    // by successive eliminations
    for(auto& barrierTask : barrierTasks)
    {
        auto barrier = barrierTask->get<mv::Barrier>("Barrier");
        auto consumersNames = barrier.getConsumers();
        for(auto& consumerName: consumersNames)
        {
            if(availableTasks.count(consumerName))
                availableTasks.erase(consumerName);
        }
    }

    std::vector<std::string> scheduling;
    std::unordered_set<std::string> barriersInMemory;
    if(!generateSchedulingRecursively(model, availableTasks, scheduling, barriersInMemory))
        throw "Impossible to schedule";

    unsigned i = 0;
    for(auto& task : scheduling)
        model.getOp(task)->set<unsigned>("schedulingNumber", i++);

}

void barrierIndexAssignmentFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::ControlModel cm(model);
    mv::OpModel om(model);

    auto globalConfigParams = model.getGlobalConfigParams();
    std::string indexAssignment = globalConfigParams->get<std::string>("barrier_index_assignment");

    if (indexAssignment == "Dynamic")
    {
        auto sortedOps = cm.schedulingSort();

        int id = 0;
        for (auto op: sortedOps)
        {
            auto barriers = getBarriersNeeded(om.switchContext(op));
            for(auto& barrier : barriers)
            {
                auto barrierTask = model.getOp(barrier);
                auto& physicalBarrier = barrierTask->get<mv::Barrier>("Barrier");
                physicalBarrier.setIndex(id++);
            }
        }
    }
}

void updateBarrierRefsIdsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    auto barrierTasks = om.getOps("BarrierTask");

    for (auto bt: barrierTasks)
    {
        auto& barrier = bt->get<mv::Barrier>("Barrier");

        for (auto producer: barrier.getProducers())
        {
            auto producerOp = om.getOp(producer);

            if (!producerOp->hasAttr("BarriersProducedByTask"))
                producerOp->set<std::vector<std::string>>("BarriersProducedByTask", std::vector<std::string>());

            auto& barrierRef = producerOp->get<std::vector<std::string>>("BarriersProducedByTask");
            barrierRef.push_back(bt->getName());
        }

        for (auto consumer: barrier.getConsumers())
        {
            auto consumerOp = om.getOp(consumer);
            if (!consumerOp->hasAttr("BarriersConsumedByTask"))
                consumerOp->set<std::vector<std::string>>("BarriersConsumedByTask", std::vector<std::string>());

            auto& barrierRef = consumerOp->get<std::vector<std::string>>("BarriersConsumedByTask");
            barrierRef.push_back(bt->getName());
        }
    }
}

void updateBarrierRefsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto barrierTasks = om.getOps("BarrierTask");

    for (auto bt: barrierTasks)
    {
        auto& barrier = bt->get<mv::Barrier>("Barrier");

        for (auto producer: barrier.getProducers())
        {
            auto producerOp = om.getOp(producer);

            if (!producerOp->hasAttr("BarrierDeps"))
                producerOp->set<mv::BarrierDependencies>("BarrierDeps", mv::BarrierDependencies());

            auto& barrierRef = producerOp->get<mv::BarrierDependencies>("BarrierDeps");
            barrierRef.addUpdateBarrier(barrier.getIndex());
        }

        for (auto consumer: barrier.getConsumers())
        {
            auto consumerOp = om.getOp(consumer);
            if (!consumerOp->hasAttr("BarrierDeps"))
                consumerOp->set<mv::BarrierDependencies>("BarrierDeps", mv::BarrierDependencies());

            auto& barrierRef = consumerOp->get<mv::BarrierDependencies>("BarrierDeps");
            barrierRef.setWaitBarrier(barrier.getIndex());
        }
    }
}
