//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/disjoint_interval_set.hpp"
#include "vpux/compiler/core/feasible_scheduler_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/SmallVector.h>

namespace vpux {

struct ControlEdge {
    ControlEdge(size_t source, size_t sink): _source(source), _sink(sink) {
    }

    bool operator<(const ControlEdge& o) const {
        return (_source != o._source) ? (_source < o._source) : (_sink < o._sink);
    }

    std::string sourceName() const {
        return std::to_string(static_cast<int>(_source));
    }
    std::string sinkName() const {
        return std::to_string(static_cast<int>(_sink));
    }

    size_t _source;
    size_t _sink;
};  // struct ControlEdge //

template <>
struct IntervalTraits<ScheduledOpOneResource> {
    using UnitType = size_t;
    using IntervalType = ScheduledOpOneResource;

    static UnitType intervalOp(const IntervalType& interval) {
        return interval._op;
    }

    static UnitType intervalBegin(const IntervalType& interval) {
        return interval._addressStart;
    }

    static UnitType intervalEnd(const IntervalType& interval) {
        return interval._addressEnd;
    }

    static bool isIntervalProducer(const IntervalType& interval) {
        return (interval._resRelation == ScheduledOpOneResource::EResRelation::PRODUCER);
    }

    static bool canProducersCoexist(const IntervalType& existingProd, const IntervalType& newProd) {
        return existingProd.canOverlap(newProd);
    }

    static bool sameIntervalAccess(const IntervalType& intA, const IntervalType& intB) {
        return !intA.canOverlap(intB);
    }

    static void setInterval(IntervalType& interval, const UnitType& beg, const UnitType& end) {
        interval._addressStart = beg;
        interval._addressEnd = end;
    }
};  // struct IntervalTraits<ScheduledOpOneResource> //

class ControlEdgeSet {
public:
    using OperationType = size_t;
    ControlEdgeSet(): _controlEdgeSet() {
    }

    void operator()(const OperationType& a, const OperationType& b) {
        _controlEdgeSet.push_back(ControlEdge(a, b));
    }

    void operator()(const ScheduledOpOneResource& a, const ScheduledOpOneResource& b) {
        _controlEdgeSet.push_back(ControlEdge(a._op, b._op));
    }

    auto size() const {
        return _controlEdgeSet.size();
    }

    const auto& operator[](size_t ind) const {
        assert(ind < _controlEdgeSet.size());
        return _controlEdgeSet[ind];
    }

    llvm::SmallVector<ControlEdge>::const_iterator begin() const {
        return _controlEdgeSet.begin();
    }
    llvm::SmallVector<ControlEdge>::const_iterator end() const {
        return _controlEdgeSet.end();
    }

private:
    llvm::SmallVector<ControlEdge> _controlEdgeSet;
};  //  class ControlEdgeSet //

// Given an iterator over sorted intervals the algorithm produces control
// edges for overlapping memory ranges in order of schedule
class ControlEdgeGenerator {
public:
    using Traits = IntervalTraits<ScheduledOpOneResource>;
    using UnitType = typename Traits::UnitType;
    using IntervalType = typename Traits::IntervalType;
    using IntervalUtilsType = IntervalUtils<IntervalType>;
    using IntervalTreeType = DisjointIntervalSet<UnitType, IntervalType>;
    using ProdConsType = typename IntervalTreeType::ProdConsType;
    using IntervalQueryIteratorType = typename IntervalTreeType::IntervalIteratorType;
    struct NoopFunctorType {
        void operator()(const IntervalType&, const IntervalType&) {
        }
    };  // struct NoopFunctorType //

    ControlEdgeGenerator(): _intervalTree() {
    }

    // Iterate over provided intervals from begin to end. Track ranges producer and consumers
    // and for any overlaps create new dependency (control edge) in the output
    template <typename IntervalIterator, typename OutputFunctor = NoopFunctorType>
    size_t generateControlEdges(IntervalIterator begin, IntervalIterator end,
                                OutputFunctor& outputDependency = OutputFunctor()) {
        size_t edgeCount = 0UL;
        for (auto currIntervalIt = begin; currIntervalIt != end; ++currIntervalIt) {
            const auto currBeg = Traits::intervalBegin(*currIntervalIt);
            const auto currEnd = Traits::intervalEnd(*currIntervalIt);
            if (currBeg > currEnd) {
                continue;
            }
            edgeCount += processNextInterval(*currIntervalIt, outputDependency);
        }
        return edgeCount;
    }

protected:
    bool overlaps(const IntervalQueryIteratorType& qitr, UnitType ibegin, UnitType iend) const {
        return IntervalUtilsType::intersects(qitr.intervalBegin(), qitr.intervalEnd(), ibegin, iend);
    }

    // Process new interval and add it on top of existing ones.
    // If range is unoccupied then just new range is tracked. No control edge is needed
    // If there is some overlap then:
    //  - For intersecting part update the range:
    //    - If added interval is range producer then create control edge from previous users to this new producer
    //      Update range ownership with new producer
    //    - If added interval is range consumer then create control edge from range producer to this new consumer
    //      Add to range ownership new user. Do not remove previous producer and users
    //  - For not intersecting parts leave previous ownership for those ranges
    template <typename OutputFunctor = NoopFunctorType>
    size_t processNextInterval(const IntervalType& currInterval, OutputFunctor& outputDependency = OutputFunctor()) {
        size_t edgeCount = 0UL;
        const auto currBeg = Traits::intervalBegin(currInterval);
        const auto currEnd = Traits::intervalEnd(currInterval);
        assert(currBeg <= currEnd);
        const auto isCurrIntervalProducer = Traits::isIntervalProducer(currInterval);

        auto qitr = _intervalTree.query(currBeg, currEnd);
        auto qitrEnd = _intervalTree.end(), qitrNext = qitrEnd;

        // Invariant: [currRemBeg, currRemEnd] is the reminder of the
        // current interval which does not overlap intervals until qitr //
        auto currRemBeg = currBeg, currRemEnd = currEnd;
        while ((qitr != qitrEnd) && overlaps(qitr, currRemBeg, currRemEnd)) {  // foreach overlap //

            assert((currRemBeg <= currRemEnd) && (currBeg <= currRemBeg) && (currRemEnd <= currEnd));

            // we now have an overlap between [qbeg,qend] and
            // [currRemBeg,currRemEnd] //
            auto qbeg = qitr.intervalBegin();
            auto qend = qitr.intervalEnd();

            auto qitrProdCons = qitr.getProdCons();
            auto newProdCons = qitrProdCons;
            ProdConsType currIntervalProdCons(currInterval);
            if (isCurrIntervalProducer) {
                // Check if currInterval producer can coexist with already present producers of this interval
                // Same range can have multiple producers as each can give different view of interval
                auto canProdCoexist = llvm::all_of(qitrProdCons._producers, [&](const IntervalType& prod) {
                    return Traits::canProducersCoexist(currInterval, prod);
                });

                if (canProdCoexist) {
                    // TODO: E#120027 optimize controlEdge to avoid multiple iterations
                    // Add control edge from source of currProducers to currInterval
                    std::set<size_t> currProducers;
                    std::set<size_t> sourcesOfCurrProducers;
                    for (auto& prod : qitrProdCons._producers) {
                        currProducers.insert(Traits::intervalOp(prod));
                    }
                    for (auto itr = outputDependency.begin(); itr != outputDependency.end(); ++itr) {
                        if (llvm::is_contained(currProducers, itr->_sink) && itr->_source != itr->_sink) {
                            sourcesOfCurrProducers.insert(itr->_source);
                        }
                    }

                    for (const auto& source : sourcesOfCurrProducers) {
                        outputDependency(source, Traits::intervalOp(currInterval));
                        ++edgeCount;
                    }

                    // Add additional producer for this interval
                    newProdCons.addProducer(currInterval);
                } else {
                    if (qitrProdCons._consumers.empty()) {
                        // If previous interval had no consumers just
                        // output the control edge from it to new producer
                        // of this interval
                        for (auto& prod : qitrProdCons._producers) {
                            outputDependency(prod, currInterval);
                            ++edgeCount;
                        }
                    } else {
                        // If previous interval had consumers output
                        // control edges from all of them to new producer
                        // of this interval
                        for (auto& cons : qitrProdCons._consumers) {
                            outputDependency(cons, currInterval);
                            ++edgeCount;
                        }
                    }
                    // Set new producer for new interval. This will
                    // also clear previous info about users
                    newProdCons.newProducer(currInterval);
                }
            } else {
                // Add edge from interval producer to new user
                for (auto& prod : qitrProdCons._producers) {
                    // If range has multiple producers each providing different view of interval
                    // then add edge to new user only from producer with matching interval access
                    if (Traits::sameIntervalAccess(currInterval, prod)) {
                        outputDependency(prod, currInterval);
                        ++edgeCount;
                    }
                }

                // Update interval owners with new user
                newProdCons.addConsumer(currInterval);
            }

            // erase the current interval //
            qitrNext = qitr;
            ++qitrNext;
            _intervalTree.erase(qbeg, qend);

            // compute the intersecting interval //
            auto interBeg = std::max(currRemBeg, qbeg);
            auto interEnd = std::min(currRemEnd, qend);
            assert(interBeg <= interEnd);

            UnitType resultBeg[2UL], resultEnd[2UL];
            ProdConsType const* resultVal[2UL];
            size_t rcount;

            // now compute the xor interval(s) //
            rcount = IntervalUtilsType::intervalXor(qbeg, qend, currRemBeg, currRemEnd, resultBeg, resultEnd);

            auto nextRemBeg = std::numeric_limits<UnitType>::max();
            auto nextRemEnd = std::numeric_limits<UnitType>::min();
            assert(rcount <= 2UL);

            if (!rcount) {
                // no xor intervals //
                const bool status = _intervalTree.insert(interBeg, interEnd, newProdCons);
                assert(status);
                std::ignore = status;
            } else {
                // NOTE: that first XOR interval comes before the second if there
                // are two XOR intervals. //

                // Determine the type of interval in the xor //
                //
                // qbeg     qend
                // [--------]
                //        [----------------]
                //      currRemBeg       currRemEnd
                //
                // the xor has atmost two parts:
                // [------)
                //          (--------------]
                for (size_t r = 0; r < rcount; r++) {
                    if (IntervalUtilsType::isSubset(resultBeg[r], resultEnd[r], currRemBeg, currRemEnd)) {
                        resultVal[r] = &currIntervalProdCons;
                    } else {
                        assert(IntervalUtilsType::isSubset(resultBeg[r], resultEnd[r], qbeg, qend));
                        resultVal[r] = &qitrProdCons;
                    }
                }

                size_t nextXorIntervalIndex = 0;

                if (resultEnd[nextXorIntervalIndex] < interBeg) {
                    // insert the interval part before [interBeg, interEnd] //
                    _intervalTree.insert(resultBeg[nextXorIntervalIndex], resultEnd[nextXorIntervalIndex],
                                         *(resultVal[nextXorIntervalIndex]));
                    ++nextXorIntervalIndex;
                }

                // insert the intersection part //
                _intervalTree.insert(interBeg, interEnd, newProdCons);

                // process the interval above the intersection part //
                if (nextXorIntervalIndex < rcount) {
                    if (resultVal[nextXorIntervalIndex] != &currIntervalProdCons) {
                        // need to insert this part back in the interval tree //
                        _intervalTree.insert(resultBeg[nextXorIntervalIndex], resultEnd[nextXorIntervalIndex],
                                             *(resultVal[nextXorIntervalIndex]));
                    } else {
                        // pass the remaining part of the currInterval to the next
                        // iteraton.
                        nextRemBeg = resultBeg[nextXorIntervalIndex];
                        nextRemEnd = resultEnd[nextXorIntervalIndex];
                    }
                }
            }

            // update the remaining part of the current interval //
            currRemBeg = nextRemBeg;
            currRemEnd = nextRemEnd;
            qitr = qitrNext;
        }  // foreach overlap //

        if (currRemBeg <= currRemEnd) {
            // process the trailing part this also covers the trailing case.//
            const auto status = _intervalTree.insert(currRemBeg, currRemEnd, currInterval);
            assert(status);
            std::ignore = status;
        }

        // TODO: do the merging within the update itself //
        if (isCurrIntervalProducer) {
            mergeAbuttingIntervals(currInterval);
        }
        return edgeCount;
    }

    // If there are abutting intervals which have the same producer then
    // they can be replaced with a single interval covering whole range.
    // This function should be called after new interval producer has
    // been processed
    void mergeAbuttingIntervals(const IntervalType& currInterval) {
        VPUX_THROW_UNLESS(Traits::isIntervalProducer(currInterval),
                          "Abutting intervals can be merged only for new resource producer");
        UnitType ibeg = Traits::intervalBegin(currInterval);
        UnitType iend = Traits::intervalEnd(currInterval);
        IntervalQueryIteratorType qitr = _intervalTree.query(ibeg, iend);
        IntervalQueryIteratorType qitrEnd = _intervalTree.end(), qitrNext, qitrStart;
        const ProdConsType currIntervalProdCons(currInterval);

        if ((qitr == qitrEnd) || !(qitr.getProdCons() == currIntervalProdCons)) {
            return;
        }

        qitrStart = qitr;
        UnitType prevLeftEnd = qitr.intervalBegin();
        UnitType prevRightEnd = qitr.intervalEnd();

        ++qitr;
        while ((qitr != qitrEnd) &&
               ((qitr.getProdCons() == currIntervalProdCons) && ((prevRightEnd + 1) == qitr.intervalBegin()))) {
            prevRightEnd = qitr.intervalEnd();
            // TODO: implement an erase using iterators instead of
            // end point values so that this takes O(1) time.
            qitrNext = qitr;
            ++qitrNext;
            _intervalTree.erase(qitr.intervalBegin(), qitr.intervalEnd());
            qitr = qitrNext;
        }

        if (prevRightEnd > qitrStart.intervalEnd()) {
            _intervalTree.erase(qitrStart.intervalBegin(), qitrStart.intervalEnd());
            _intervalTree.insert(prevLeftEnd, prevRightEnd, currIntervalProdCons);
        }
    }

    IntervalTreeType _intervalTree;
};  // class ControlEdgeGenerator //

}  // namespace vpux
