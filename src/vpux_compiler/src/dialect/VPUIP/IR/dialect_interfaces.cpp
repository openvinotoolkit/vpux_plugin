//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect_interfaces.hpp"

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/batch.hpp"

using namespace vpux;

namespace detail {

/*
 * Dispatch pre-inlining callOp processing to
 * a specific processor if conditions are met
 */
struct CallOPPreInliner {
    virtual ~CallOPPreInliner() = default;
    virtual bool isApplicable(mlir::Operation*) const = 0;
    virtual void apply(mlir::Operation*, mlir::iterator_range<mlir::Region::iterator>) const = 0;
};

struct CallOPPreInlinerVisitor {
    CallOPPreInlinerVisitor(Logger log = Logger::global());

    template <class PreInliner, class... Args>
    void addPreInliner(Args&&... args) {
        preInliners.push_back(std::make_unique<PreInliner>(_log, std::forward<Args>(args)...));
    }

    void visit(mlir::Operation* op, mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const;

private:
    Logger _log;
    std::vector<std::unique_ptr<CallOPPreInliner>> preInliners;
};

namespace batching {
/*
 * Once callOp is categorized as a part of batched processing,
 * which means is has `debatched` tag at the moment of the implementation,
 * this preprocessor takes care about suitable resource mapping.
 * By default during each callOp compilation it's supposed to occupy
 * whole NPU resources like tiles/CMX, DDR and so on.
 * Having multiple callOp responsible for processing different lines of a batched tensor,
 * this means that we face to resource concurrency among different callOp because
 * eventually all of them are mapped on the same CMXs and DDR addresses using same offsets.
 * To overcome this resouce allocation limitation this preInliner was conceived.
 *
 * The preinliner responsibilities are the following:
 *  a) detemine which a callOp index I is from a batched dimension range [0...N]
 *  b) remap CMX, cluster_id, section_id etc. from the range [0...T], where T is a compilation tile count, to the range
 * [0 + I... (T + I) / N] c) Apply similar logic to DDR allocation (TODO - E###-131884)
 */
struct BatchedCallOpPreInliner : public CallOPPreInliner {
    BatchedCallOpPreInliner(Logger& log): _log(log.nest("batch-preinliner")), dispatcher(_log) {
    }

    struct CMXTypeModifier {
        mutable Logger _log;
        CMXTypeModifier(Logger& log): _log(log) {
        }
        CMXTypeModifier(Logger&& log): _log(std::move(log)) {
        }

        static size_t recalculateIndex(size_t index, const DebatchedCallOpData& callOpData,
                                       size_t totalAvailableTilesCount);
        mlir::Type transform(mlir::Type type, const DebatchedCallOpData& callOpData,
                             size_t totalAvailableTilesCount) const;

        template <class ClusterIdFunctor>
        static SmallVector<vpux::VPUIP::OutwardHaloRegionAttr> modifyOutwardHaloAttrs(
                ArrayRef<vpux::VPUIP::OutwardHaloRegionAttr> outwardHalos, ClusterIdFunctor modifier, Logger log);

        template <class ClusterIdFunctor>
        static SmallVector<vpux::VPUIP::HaloRegionAttr> modifyHaloAttrsClusterId(
                ArrayRef<vpux::VPUIP::HaloRegionAttr> haloAttrs, ClusterIdFunctor modifier, Logger log);
    };

    struct ResourceDescriptor {
        DebatchedCallOpData callOpData;
        size_t totalAvailableTilesCount;
        std::optional<size_t> singleFunctionDDRConsumptionBytes;
        size_t maxDDRBytesAvailable;

        std::string to_string() const;
        static ResourceDescriptor create(::mlir::func::CallOp callOp,
                                         ::mlir::iterator_range<::mlir::Region::iterator> inlinedBlocks, Logger log);

    private:
        ResourceDescriptor() = delete;
    };
    /*
     * Delegates processing to appropriate handler if conditions apt the hadler invocation,
     * depends on operation type and attributes it carrying
     */
    class Dispatcher {
        struct OpExtractor {
            OpExtractor(mlir::Operation& op): _op(op), _opCounter(0) {
            }
            mlir::Operation* next() {
                // once being called, it returns an operation itself
                if (_opCounter == 0) {
                    _opCounter++;
                    return &_op;
                }

                // Subsequent calls return further stacked operations from enclosed region:
                // only single stacked op is supported as part of an encompassed operation, see TaskOp.
                // In general, we will need to extract next stacked/enclosed operations until exist but at the moment
                // any task op comprises only one operation inside
                if (_opCounter == 1) {
                    _opCounter++;
                    if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(_op); taskOp != nullptr) {
                        if (auto innerTaskOp = taskOp.getInnerTaskOp(); innerTaskOp != nullptr) {
                            return innerTaskOp;
                        }
                    }
                }
                return nullptr;
            }

        private:
            mlir::Operation& _op;
            size_t _opCounter = 0;
        };

        struct SpecificOpPreInliner {
            virtual ~SpecificOpPreInliner() = default;
            virtual bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const = 0;
        };

        struct CMXModifierForDeclareOp final : public SpecificOpPreInliner, private CMXTypeModifier {
            CMXModifierForDeclareOp(Logger&& log): CMXTypeModifier(log), _log(std::move(log)) {
            }
            bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const override;

        private:
            bool applyCMX(VPURT::DeclareBufferOp& op, const DebatchedCallOpData& callOpData,
                          size_t totalAvailableTilesCount) const;
            bool applyDDR(VPURT::DeclareBufferOp& op, const DebatchedCallOpData& callOpData,
                          size_t offsetDDRAllocationBytes, size_t maxDDRBytesAvailable) const;
            mutable Logger _log;
        };

        struct CMXModifierForNCEClusterTaskOp final : public SpecificOpPreInliner, private CMXTypeModifier {
            CMXModifierForNCEClusterTaskOp(Logger&& log): CMXTypeModifier(log), _log(std::move(log)) {
            }
            bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const override;

        private:
            mutable Logger _log;
        };

        struct CMXModifierForSWKernelOp final : public SpecificOpPreInliner, private CMXTypeModifier {
            CMXModifierForSWKernelOp(Logger&& log): CMXTypeModifier(log), _log(std::move(log)) {
            }
            bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const override;

        private:
            mutable Logger _log;
        };
        mutable Logger _log;
        std::vector<std::unique_ptr<SpecificOpPreInliner>> specificPreInliners;

    public:
        Dispatcher(Logger& log);
        ~Dispatcher() = default;
        void dispatch(mlir::Operation& op, const ResourceDescriptor& resourse) const;
    };

    bool isApplicable(mlir::Operation* call) const override;
    void apply(mlir::Operation* call, mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const override;

private:
    Logger _log;
    Dispatcher dispatcher;
};

/*
 * BatchedCallOpPreInliner
 */

bool BatchedCallOpPreInliner::isApplicable(mlir::Operation* call) const {
    if (mlir::isa_and_nonnull<mlir::func::CallOp>(call) && call->hasAttr(vpux::DebatchedCallOpAttributeView::name())) {
        return true;
    }
    return false;
}

mlir::func::FuncOp getCalledFunction(mlir::func::CallOp callOp) {
    mlir::SymbolRefAttr sym = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(callOp.getCallableForCallee());
    if (!sym)
        return nullptr;
    return mlir::dyn_cast_or_null<mlir::func::FuncOp>(mlir::SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

void BatchedCallOpPreInliner::apply(mlir::Operation* call,
                                    mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const {
    ResourceDescriptor resourse =
            ResourceDescriptor::create(mlir::dyn_cast<::mlir::func::CallOp>(call), inlinedBlocks, _log);

    _log.info("apply BatchedCallOpPreInliner: {0}", resourse.to_string());
    for (mlir::Block& block : inlinedBlocks) {
        for (auto& op : block.getOperations()) {
            dispatcher.dispatch(op, resourse);
        }
    }
}

/*
 * BatchedCallOpPreInliner::ResourceDescriptor
 */

std::string BatchedCallOpPreInliner::ResourceDescriptor::to_string() const {
    std::stringstream ss;
    ss << callOpData.to_string() << ", tiles count: " << totalAvailableTilesCount;
    if (singleFunctionDDRConsumptionBytes.has_value()) {
        ss << ", DDR offset: " << singleFunctionDDRConsumptionBytes.value();
    } else {
        ss << ", DDR offset: UNDETERMINED";
    }
    ss << ", DDR available bytes: " << maxDDRBytesAvailable;
    return ss.str();
}

BatchedCallOpPreInliner::ResourceDescriptor BatchedCallOpPreInliner::ResourceDescriptor::create(
        ::mlir::func::CallOp call, ::mlir::iterator_range<::mlir::Region::iterator> inlinedBlocks, Logger log) {
    auto debatchedAttr = DebatchedCallOpAttributeView::extract(call);
    VPUX_THROW_UNLESS(debatchedAttr.has_value(), "BatchedCallOpPreInliner::apply expected an attribute: {0}",
                      DebatchedCallOpAttributeView::name());
    const DebatchedCallOpData& callOpData = debatchedAttr.value().getCallData();
    auto module = vpux::getModuleOp(call);
    auto tileOp = IE::getTileExecutor(module);
    auto tileExecutorCount = tileOp.getCount();
    // get used memory
    auto maxDDRBytesAvailable =
            checked_cast<uint64_t>(IE::getAvailableMemory(module, vpux::VPU::MemoryKind::DDR).getByteSize());
    log.debug("Procced with gathering all DDR buffer allocations to determine a device memory occupation range");
    std::map<size_t, size_t> allocationsOffsetSize;
    for (mlir::Block& block : inlinedBlocks) {
        for (auto& op : block.getOperations()) {
            if (!mlir::isa<VPURT::DeclareBufferOp>(op)) {
                continue;
            }

            auto declareOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(op);

            auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(declareOp.getType());
            if (ndType.getMemoryKind() != VPU::MemoryKind::DDR) {
                continue;
            }
            size_t offset = declareOp.getByteOffset();
            int64_t shapeSize = calcTotalShapeSize(ndType.getShape());
            int64_t memorySize = shapeSize * getElemTypeSize(ndType.getElementType()).to<Byte>().count();
            allocationsOffsetSize.emplace(offset, memorySize);
        }
    }

    if (allocationsOffsetSize.empty()) {
        log.debug("No DDR allocations, no any offset recalculation required");
        return ResourceDescriptor{callOpData, static_cast<size_t>(tileExecutorCount), {}, maxDDRBytesAvailable};
    }

    log.debug("collected DDR allocations: {0}", allocationsOffsetSize.size());
    size_t index = 0;
    std::pair<size_t, size_t> occupiedDDRAdressesRange{std::numeric_limits<size_t>::max(), 0};
    for (auto [offset, size] : allocationsOffsetSize) {
        log.trace("{0}: {1}, {2}", index, offset, size);
        index++;
        occupiedDDRAdressesRange.first = std::min(offset, occupiedDDRAdressesRange.first);           // left border
        occupiedDDRAdressesRange.second = std::max(offset + size, occupiedDDRAdressesRange.second);  // right border
        log.trace("occupied DDR adresses range: [{0},{1}]", occupiedDDRAdressesRange.first,
                  occupiedDDRAdressesRange.second);
    }
    log.debug("calculated occupied DDR adresses range: [{0},{1}]", occupiedDDRAdressesRange.first,
              occupiedDDRAdressesRange.second);
    VPUX_THROW_WHEN(occupiedDDRAdressesRange.first > occupiedDDRAdressesRange.second,
                    "DDR adress range determined incorrectly, left border: {0} cann't be greater than right: {1}",
                    occupiedDDRAdressesRange.first, occupiedDDRAdressesRange.second);
    return ResourceDescriptor{callOpData, static_cast<size_t>(tileExecutorCount),
                              occupiedDDRAdressesRange.first + occupiedDDRAdressesRange.second, maxDDRBytesAvailable};
}

/*
 * BatchedCallOpPreInliner::Dispatcher
 */

BatchedCallOpPreInliner::Dispatcher::Dispatcher(Logger& log): _log(log.nest()) {
    specificPreInliners.push_back(std::make_unique<CMXModifierForDeclareOp>(log.nest()));
    specificPreInliners.push_back(std::make_unique<CMXModifierForNCEClusterTaskOp>(log.nest()));
    specificPreInliners.push_back(std::make_unique<CMXModifierForSWKernelOp>(log.nest()));
}

void BatchedCallOpPreInliner::Dispatcher::dispatch(mlir::Operation& op, const ResourceDescriptor& res) const {
    bool processed = false;
    OpExtractor extractor(op);
    for (auto innerOp = extractor.next(); innerOp != nullptr; innerOp = extractor.next()) {
        for (const auto& p : specificPreInliners) {
            if (p->apply(*innerOp, res)) {
                processed = true;
                break;
            }
        }
        if (!processed) {
            _log.trace("Default processing of: {0} started", innerOp->getName());
            CMXTypeModifier modifier(_log);
            for (auto&& result : innerOp->getResults()) {
                auto newType = modifier.transform(result.getType(), res.callOpData, res.totalAvailableTilesCount);
                result.setType(newType);
            }
            _log.trace("Default processing of: {0} finished", innerOp->getName());
        }
    }
}

/*
 * BatchedCallOpPreInliner::CMXTypeModifier
 */

size_t BatchedCallOpPreInliner::CMXTypeModifier::recalculateIndex(size_t index, const DebatchedCallOpData& callOpData,
                                                                  size_t totalAvailableTilesCount) {
    return index + (callOpData.getCallIndex() * totalAvailableTilesCount) / callOpData.getBatchSize();
}

mlir::Type BatchedCallOpPreInliner::CMXTypeModifier::transform(mlir::Type type, const DebatchedCallOpData& callOpData,
                                                               size_t totalAvailableTilesCount) const {
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(type);
    if (ndType == nullptr || ndType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        return type;
    }

    auto memSpace = ndType.getMemSpace();
    VPUX_THROW_UNLESS(memSpace.getRootName() == "CMX_NN" && memSpace.getLeafName() == "CMX_NN",
                      "Expected memspace CMX_NN, got: {0}/{1}", memSpace.getRootName(), memSpace.getLeafName());
    if (!memSpace.getIndex().has_value()) {
        return type;
    }
    auto calculateClusterId = [&callOpData, totalAvailableTilesCount](const mlir::IntegerAttr& attr) -> size_t {
        return CMXTypeModifier::recalculateIndex(parseIntAttr<size_t>(attr), callOpData, totalAvailableTilesCount);
    };

    auto calculateCmxIndex = [calculateClusterId](const IndexedSymbolAttr& memSpace) -> size_t {
        return calculateClusterId(memSpace.getIndexAttr().value());
    };
    auto newCMXIndex = calculateCmxIndex(memSpace);
    _log.trace("mem kind: {0} has tile index: {1}, new index: {2}", stringifyEnum(ndType.getMemoryKind()),
               memSpace.getIndex().value(), newCMXIndex);
    auto newMemSpace = vpux::IndexedSymbolAttr::get(ndType.getContext(), memSpace.getRootName(), newCMXIndex);
    auto newType = ndType.changeMemSpace(newMemSpace);

    auto itiBufferTypeFromOp = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(type);
    if (itiBufferTypeFromOp) {
        // HALO regions require for additional processing
        ArrayRef<vpux::VPUIP::HaloRegionAttr> inwardHalos = itiBufferTypeFromOp.getInwardHaloRegions();
        SmallVector<vpux::VPUIP::HaloRegionAttr> newInwardsHaloAttr =
                modifyHaloAttrsClusterId(inwardHalos, calculateClusterId, _log);

        ArrayRef<vpux::VPUIP::OutwardHaloRegionAttr> outwardHalos = itiBufferTypeFromOp.getOutwardHaloRegions();
        SmallVector<vpux::VPUIP::OutwardHaloRegionAttr> newOutwardsHaloAttr =
                modifyOutwardHaloAttrs(outwardHalos, calculateClusterId, _log);

        auto newItiBufferTypeFromOp = VPUIP::ITIBufferType::get(
                itiBufferTypeFromOp.getContext(),
                itiBufferTypeFromOp.getShape()
                        .raw(),  // Despite ctor requires memShape(), we pass getShape() to avoid dimensions reordering
                itiBufferTypeFromOp.getElementType(), itiBufferTypeFromOp.getLayout(), newMemSpace,
                itiBufferTypeFromOp.getIduSegmentation(), newInwardsHaloAttr, newOutwardsHaloAttr);
        newType = newItiBufferTypeFromOp;
    }

    return newType;
}

template <class ClusterIdFunctor>
SmallVector<vpux::VPUIP::HaloRegionAttr> BatchedCallOpPreInliner::CMXTypeModifier::modifyHaloAttrsClusterId(
        ArrayRef<vpux::VPUIP::HaloRegionAttr> haloAttrs, ClusterIdFunctor modifier, Logger log) {
    SmallVector<vpux::VPUIP::HaloRegionAttr> newInwardsHaloAttr;
    newInwardsHaloAttr.reserve(haloAttrs.size());
    auto nestedLog = log.nest();
    nestedLog.trace("HaloRegionAttrs: {0}", haloAttrs.size());
    for (auto&& haloAttr : haloAttrs) {
        auto newCMXIndex = modifier(haloAttr.getClusterId());
        nestedLog.trace("HaloRegionAttr has `cluster_id`: {0}, new value: {1}",
                        parseIntAttr<size_t>(haloAttr.getClusterId()), newCMXIndex);
        auto newClusterIdAttr = getIntAttr(haloAttr.getContext(), newCMXIndex);
        auto newHaloAttr = vpux::VPUIP::HaloRegionAttr::get(haloAttr.getContext(), haloAttr.getShape(),
                                                            haloAttr.getOffset(), newClusterIdAttr);
        newInwardsHaloAttr.push_back(newHaloAttr);
    }
    return newInwardsHaloAttr;
}

template <class ClusterIdFunctor>
SmallVector<vpux::VPUIP::OutwardHaloRegionAttr> BatchedCallOpPreInliner::CMXTypeModifier::modifyOutwardHaloAttrs(
        ArrayRef<vpux::VPUIP::OutwardHaloRegionAttr> outwardHalos, ClusterIdFunctor modifier, Logger log) {
    SmallVector<vpux::VPUIP::OutwardHaloRegionAttr> newOutwardsHaloAttr;
    newOutwardsHaloAttr.reserve(outwardHalos.size());
    Logger _log = log.nest();
    _log.trace("ITIBufferType has outwardHaloRegions: {0}", outwardHalos.size());
    for (auto& haloAttr : outwardHalos) {
        auto newCMXIndex = modifier(haloAttr.getClusterId());
        _log.trace("Outward halo region has `cluster_id`: {0}, new value: {1}",
                   parseIntAttr<size_t>(haloAttr.getClusterId()), newCMXIndex);
        auto newClusterIdAttr = getIntAttr(haloAttr.getContext(), newCMXIndex);
        auto inwardFromOutwardsAttrs = haloAttr.getInwardHaloRegions();

        SmallVector<vpux::VPUIP::HaloRegionAttr> newInwardHalos = CMXTypeModifier::modifyHaloAttrsClusterId(
                parseCustomAttrArray<vpux::VPUIP::HaloRegionAttr>(inwardFromOutwardsAttrs), modifier, _log);
        SmallVector<mlir::Attribute> newInwardHalosAttrs;
        newInwardHalosAttrs.reserve(newInwardHalos.size());
        std::transform(newInwardHalos.begin(), newInwardHalos.end(), std::back_inserter(newInwardHalosAttrs),
                       [](vpux::VPUIP::HaloRegionAttr attr) {
                           return attr;
                       });
        auto newHaloAttr = vpux::VPUIP::OutwardHaloRegionAttr::get(
                haloAttr.getContext(), haloAttr.getShape(), haloAttr.getOffset(), newClusterIdAttr,
                mlir::ArrayAttr::get(haloAttr.getContext(), newInwardHalosAttrs));
        newOutwardsHaloAttr.push_back(newHaloAttr);
    }
    return newOutwardsHaloAttr;
}

/*
 * BatchedCallOpPreInliner::Dispatcher::CMXModifierForDeclareOp
 */

bool BatchedCallOpPreInliner::Dispatcher::CMXModifierForDeclareOp::apply(mlir::Operation& op,
                                                                         const ResourceDescriptor& resource) const {
    if (!mlir::isa<VPURT::DeclareBufferOp>(op)) {
        return false;
    }

    auto declareOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(op);
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(declareOp.getType());
    if (ndType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
        return applyCMX(declareOp, resource.callOpData, resource.totalAvailableTilesCount);
    } else if (ndType.getMemoryKind() == VPU::MemoryKind::DDR &&
               resource.singleFunctionDDRConsumptionBytes.has_value()) {
        return applyDDR(declareOp, resource.callOpData, resource.singleFunctionDDRConsumptionBytes.value(),
                        resource.maxDDRBytesAvailable);
    }

    return false;
}

bool BatchedCallOpPreInliner::Dispatcher::CMXModifierForDeclareOp::applyCMX(VPURT::DeclareBufferOp& op,
                                                                            const DebatchedCallOpData& callOpData,
                                                                            size_t totalAvailableTilesCount) const {
    mlir::ModuleOp module = vpux::getModuleOp(op);
    auto ctx = module.getContext();

    auto sectionArrayAttr = op.getSectionIndexAttr();
    auto sectionArray = parseIntArrayAttr<size_t>(sectionArrayAttr);
    _log.trace("{0} current section value: {1}", op->getName(), sectionArray);
    std::transform(sectionArray.begin(), sectionArray.end(), sectionArray.begin(),
                   [&callOpData, totalAvailableTilesCount](size_t index) {
                       return CMXTypeModifier::recalculateIndex(index, callOpData, totalAvailableTilesCount);
                   });

    auto declareOpResult = op->getResult(0);
    auto newType = transform(declareOpResult.getType(), callOpData, totalAvailableTilesCount);
    declareOpResult.setType(newType);

    _log.trace("{0} new section value: {1}", op->getName(), sectionArray),
            op.setSectionIndexAttr(getIntArrayAttr(ctx, sectionArray));
    return true;
}

bool BatchedCallOpPreInliner::Dispatcher::CMXModifierForDeclareOp::applyDDR(VPURT::DeclareBufferOp& op,
                                                                            const DebatchedCallOpData& callOpData,
                                                                            size_t offsetDDRAllocationBytes,
                                                                            size_t maxDDRBytesAvailable) const {
    size_t currentBatchedCallDDROffset = callOpData.getCallIndex() * offsetDDRAllocationBytes;
    auto opBytesOffset = op.getByteOffset();
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(op.getType());
    int64_t shapeSize = calcTotalShapeSize(ndType.getShape());
    int64_t memorySize = shapeSize * getElemTypeSize(ndType.getElementType()).to<Byte>().count();
    VPUX_THROW_WHEN(
            currentBatchedCallDDROffset + opBytesOffset + memorySize > maxDDRBytesAvailable,
            "Cannot substitute DDR offset: {0} by a new one: {1} as available memory range is limited by: {2} bytes."
            "Please try on \"debatching-inlining-method=naive\" instead",
            opBytesOffset, currentBatchedCallDDROffset + opBytesOffset, maxDDRBytesAvailable);
    _log.debug("{0} DDR offset old: {1}, new: {2}", op->getName(), opBytesOffset,
               currentBatchedCallDDROffset + opBytesOffset);
    op.setByteOffset(currentBatchedCallDDROffset + opBytesOffset);
    return false;
}

/*
 * BatchedCallOpPreInliner::Dispatcher::CMXModifierForNCEClusterTaskOp
 */

bool BatchedCallOpPreInliner::Dispatcher::CMXModifierForNCEClusterTaskOp::apply(
        mlir::Operation& op, const ResourceDescriptor& resource) const {
    if (!mlir::isa<vpux::VPUIP::NCEClusterTaskOp>(op)) {
        return false;
    }

    auto nceTaskOp = mlir::dyn_cast<vpux::VPUIP::NCEClusterTaskOp>(op);
    for (auto result : nceTaskOp.getResults()) {
        auto newType = transform(result.getType(), resource.callOpData, resource.totalAvailableTilesCount);
        result.setType(newType);
    }

    auto dpuTasks = nceTaskOp.getVariants().getOps<VPUIP::DPUTaskOp>();
    _log.trace("{0} has DPUTaskOps: {1}", op.getName(), !dpuTasks.empty());
    for (auto&& dpuTaskOp : dpuTasks) {
        auto clusterId = dpuTaskOp.getClusterId();
        if (clusterId.has_value()) {
            auto newClusterId = CMXTypeModifier::recalculateIndex(clusterId.value(), resource.callOpData,
                                                                  resource.totalAvailableTilesCount);
            _log.trace("{0} has `cluster_id`: {1}, new value: {2}", dpuTaskOp->getName(), clusterId.value(),
                       newClusterId);
            dpuTaskOp.setClusterId(newClusterId);
        }
    }
    return true;
}

/*
 * BatchedCallOpPreInliner::Dispatcher::CMXModifierForSWKernelOp
 */

bool BatchedCallOpPreInliner::Dispatcher::CMXModifierForSWKernelOp::apply(mlir::Operation& op,
                                                                          const ResourceDescriptor& resource) const {
    if (!mlir::isa<vpux::VPUIP::SwKernelOp>(op)) {
        return false;
    }

    auto swKernelTaskOp = mlir::dyn_cast<vpux::VPUIP::SwKernelOp>(op);
    for (auto result : swKernelTaskOp.getResults()) {
        mlir::Type newType = transform(result.getType(), resource.callOpData, resource.totalAvailableTilesCount);
        result.setType(newType);
    }

    auto tileIndex = swKernelTaskOp.getTileIndex();
    if (tileIndex.has_value()) {
        size_t newTileIndex = CMXTypeModifier::recalculateIndex(tileIndex.value(), resource.callOpData,
                                                                resource.totalAvailableTilesCount);
        swKernelTaskOp.setTileIndex(newTileIndex);
        _log.trace("{0}, has `tile_index`: {1}, new index: {2}", swKernelTaskOp->getName(), tileIndex.value(),
                   newTileIndex);
    }

    auto swKernelRuns = swKernelTaskOp.getBody().getOps<VPUIP::SwKernelRun>();
    _log.trace("{0} has SwKernelRun: {0}", swKernelTaskOp->getName(), !swKernelRuns.empty());
    for (auto&& kernelRun : swKernelRuns) {
        auto operands = kernelRun->getOperands();
        _log.trace("operands: {0}", operands.size());
        for (auto operand : operands) {
            auto kernelRunNewType =
                    transform(operand.getType(), resource.callOpData, resource.totalAvailableTilesCount);
            operand.setType(kernelRunNewType);
        }
    }
    return true;
}
}  // namespace batching

CallOPPreInlinerVisitor::CallOPPreInlinerVisitor(Logger log): _log(log.nest()) {
    addPreInliner<batching::BatchedCallOpPreInliner>();
}

void CallOPPreInlinerVisitor::visit(mlir::Operation* op,
                                    mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const {
    VPUX_THROW_UNLESS(op != nullptr, "Empty operation");
    _log.trace("CallOPPreInlinerVisitor started");
    for (const auto& p : preInliners) {
        if (p->isApplicable(op)) {
            p->apply(op, inlinedBlocks);
        }
    }
    _log.trace("CallOPPreInlinerVisitor finished");
}
}  // namespace detail

bool VPUIP::FuncInlinerInterface::isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const {
    return true;
}

bool VPUIP::FuncInlinerInterface::isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const {
    return true;
}

bool VPUIP::FuncInlinerInterface::isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const {
    return true;
}

void VPUIP::FuncInlinerInterface::handleTerminator(mlir::Operation*, mlir::ValueRange) const {
}

void VPUIP::FuncInlinerInterface::processInlinedCallBlocks(
        mlir::Operation* call, mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const {
    auto parentOp = call->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(parentOp == nullptr, "fun.call must have parent VPURT::TaskOp");

    ::detail::CallOPPreInlinerVisitor preProc;
    preProc.visit(call, inlinedBlocks);

    DenseMap<VPURT::TaskQueueType, std::pair<VPURT::TaskOp, VPURT::TaskOp>> taskQueuesFirstAndLastOpMap;
    for (mlir::Block& block : inlinedBlocks) {
        for (auto& op : block.getOperations()) {
            if (mlir::isa<mlir::func::ReturnOp, VPURT::DeclareBufferOp, VPURT::DeclareVirtualBarrierOp,
                          Const::DeclareOp>(op)) {
                continue;
            }

            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
            VPUX_THROW_WHEN(taskOp == nullptr, "Unexpected operation type: {0}", op.getName());

            const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);

            // DPU and Shave tasks are all expected to be guarded by barriers so in case such tasks in given inlined
            // block don't have either wait or update barrier connect them to parent task barriers. Logic for finding
            // first and last op in a queue does not handle Shave tasks to full extent due to lack of explicit tasks
            // list for multiple shave engines on single NCE cluster. In such case there might be multiple first or last
            // Shave task that should be connected to a parent barrier.
            if (taskQueueType.type != VPU::ExecutorKind::DMA_NN) {
                if (taskOp.getWaitBarriers().empty()) {
                    taskOp.getWaitBarriersMutable().append(parentOp.getWaitBarriers());
                }
                if (taskOp.getUpdateBarriers().empty()) {
                    taskOp.getUpdateBarriersMutable().append(parentOp.getUpdateBarriers());
                }
                continue;
            }

            if (taskQueuesFirstAndLastOpMap.find(taskQueueType) == taskQueuesFirstAndLastOpMap.end()) {
                // First occurrence of task on this queue
                taskQueuesFirstAndLastOpMap[taskQueueType] = std::make_pair(taskOp, taskOp);
            } else {
                // In case new task spotted, update last task info
                taskQueuesFirstAndLastOpMap[taskQueueType].second = taskOp;
            }
        }
    }
    // Identify first and last task on each execution queue.
    // For first tasks if they do no wait on any barrier connect them with start barrier
    // For end tasks if they do not update any barrier connect then to end barrier
    for (auto& taskQueuesFirstAndLastOp : taskQueuesFirstAndLastOpMap) {
        auto queueFirstOp = taskQueuesFirstAndLastOp.second.first;
        auto queueLastOp = taskQueuesFirstAndLastOp.second.second;
        if (queueFirstOp.getWaitBarriers().empty() && !parentOp.getWaitBarriers().empty()) {
            // Empty "waits" barriers means
            // this operation is one of the first operations from the callable region
            // Add "waits" barriers(if exist) from the parent VPURT::TaskOp
            // to wait operators from the previous callable region
            queueFirstOp.getWaitBarriersMutable().append(parentOp.getWaitBarriers());
        }

        if (queueLastOp.getUpdateBarriers().empty() && !parentOp.getUpdateBarriers().empty()) {
            // Empty "update" barriers means
            // this operation is one of the last operations from the callable region
            // Add "update" barriers(if exist) from the parent VPURT::TaskOp
            // to notify operators from the next callable region
            queueLastOp.getUpdateBarriersMutable().append(parentOp.getUpdateBarriers());
        }
    }
}

std::tuple<mlir::Block*, mlir::Block::iterator> VPUIP::FuncInlinerInterface::getInlineBlockAndPoint(
        mlir::Operation* call) const {
    VPUX_THROW_WHEN(call == nullptr, "fun.call must not be empty");
    auto taskOp = call->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOp == nullptr, "fun.call must have parent VPURT::TaskOp");

    return std::make_tuple(taskOp->getBlock(), std::next(taskOp->getIterator()));
}

void VPUIP::FuncInlinerInterface::eraseCall(mlir::Operation* call) const {
    VPUX_THROW_WHEN(call == nullptr, "fun.call must not be empty");
    auto taskOp = call->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOp == nullptr, "fun.call must have parent VPURT::TaskOp");

    taskOp->erase();
}
