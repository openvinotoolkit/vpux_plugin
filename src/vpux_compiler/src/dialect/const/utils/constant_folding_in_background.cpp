//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"

using namespace vpux;
using namespace vpux::Const;

BackgroundConstantFolding::BackgroundConstantFolding(mlir::MLIRContext* ctx, size_t maxConcurrentTasks,
                                                     bool collectStatistics, Logger log)
        : _ctx(ctx), _maxConcurrentTasks(maxConcurrentTasks), _log(log) {
    if (!_ctx->isMultithreadingEnabled()) {
        _log.warning("Multi thread is disabled, background constant folding is disabled");
        _isEnabled = false;
        return;
    }

    auto& threadPool = _ctx->getThreadPool();

    if (threadPool.getThreadCount() <= 1) {
        _log.warning("The thread pool size is not greater than 1, background constant folding is disabled");
        _isEnabled = false;
        return;
    }

    auto& cacheManager = ConstantFoldingCacheManager::getInstance();
    cacheManager.addCache(_ctx);
    if (collectStatistics) {
        cacheManager.get(_ctx).enableStatisticsCollection();
    }

    _listenerThread = initFoldingListener(threadPool);
}

BackgroundConstantFolding::~BackgroundConstantFolding() {
    if (!_isEnabled) {
        return;
    }

    try {
        stopFoldingListener();

        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return _activeTasks == 0;
        });
    } catch (...) {
        _log.error("Exception caught in BackgroundConstantFolding destructor");
    }
}

// Checks if the ContentAttr with the list of transformations up to the new one already has the folded result in the
// cache. If it does, it will be reused and only the new transformation (and any other that is placed after the new
// transformation in the list) will be applied.
// Returns true if this partial folding has succeeded, otherwise false.
//
// For example:
// New ContentAttr: [Transformation1, Transformation2, NewTransformation, Transformation3]
// Checks cache for presence of ContentAttr: [Transformation1, Transformation2]
// If the folded value is found in the cache, transformations [NewTransformation, Transformation3] are called over it
bool tryFoldingPartially(vpux::Const::ConstantFoldingCache& cache, Const::ContentAttr request,
                         Const::TransformAttrInterface newTransformation) {
    if (newTransformation == nullptr) {
        return false;
    }
    auto partialContentAttr = request.stripTransformationsFrom(newTransformation);
    auto maybeFoldedPartialContent = cache.getContent(partialContentAttr);
    if (!maybeFoldedPartialContent.has_value()) {
        return false;
    }
    auto foldedPartialContent = maybeFoldedPartialContent.value();

    auto lastTransformations = request.getLastTransformationsFrom(newTransformation);
    if (lastTransformations.empty()) {
        return false;
    }

    auto partialContent =
            Const::Content::fromRawBuffer(foldedPartialContent.getType(), foldedPartialContent.getRawTempBuf(),
                                          foldedPartialContent.getStorageElemType(), foldedPartialContent.isSplat());
    for (auto tr : lastTransformations) {
        partialContent = tr.transform(partialContent);
    }

    // Create a copy of the Content which will own the referenced buffer
    // This is done since the Content object obtained after folding may reference an external object without
    // owning it. If that object is erased, the Content object from the cache would point to an invalid object
    cache.addContent(request, partialContent.copyUnownedBuffer());
    cache.removeContent(partialContentAttr);

    return true;
}

class TaskCompletionNotifier {
public:
    TaskCompletionNotifier(std::atomic<size_t>& activeTasks, std::condition_variable& cv)
            : _activeTasks(&activeTasks), _cv(&cv) {
    }

    ~TaskCompletionNotifier() {
        (*_activeTasks)--;
        _cv->notify_one();
    }

private:
    std::atomic<size_t>* const _activeTasks;
    std::condition_variable* const _cv;
};

void BackgroundConstantFolding::processFoldingRequest(const FoldingRequest& foldingRequest,
                                                      ConstantFoldingCache& cache) {
    // As the main compilation process also utilizes the pool concurrently, _maxConcurrentTasks is introduced to
    // manually control the resources used by constant folding in background.
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] {
        return _activeTasks < _maxConcurrentTasks;
    });

    _activeTasks++;
    _ctx->getThreadPool().async([this, foldingRequest, &cache]() {
        TaskCompletionNotifier notifier(_activeTasks, _cv);

        Const::ContentAttr request;
        if (auto equivalenceRequest = mlir::dyn_cast_or_null<Const::EquivalenceRequestAttr>(foldingRequest.attr)) {
            if (cache.replaceContentAttr(equivalenceRequest.getOriginalAttr(), equivalenceRequest.getNewAttr())) {
                return;
            }
            request = equivalenceRequest.getNewAttr();
        } else {
            request = mlir::dyn_cast_if_present<Const::ContentAttr>(foldingRequest.attr);
        }
        VPUX_THROW_WHEN(request == nullptr, "Invalid folding request");

        if (cache.hasContent(request)) {
            if (cache.isStatisticsCollectionEnabled()) {
                cache.getStatistics().numDuplicatedRequests++;
            }
            return;
        }

        // Try folding partially if the new transformation is added to the end of the list of transformations
        // In this case, it is likely that the previous ContentAttr (without the new transformation) is already
        // in the cache, so its folded result can be reused
        if (tryFoldingPartially(cache, request, foldingRequest.newTransformation)) {
            return;
        }

        // Create a copy of the Content which will own the referenced buffer
        // This is done since the Content object obtained after folding may reference an external object without
        // owning it. If that object is erased, the Content object from the cache would point to an invalid
        // object
        cache.addContent(request, request.fold(/*bypassCache=*/true).copyUnownedBuffer());
    });
}

std::shared_future<void> BackgroundConstantFolding::initFoldingListener(llvm::ThreadPool& threadPool) {
    auto listenerThread = threadPool.async([this]() {
        auto& cacheManager = ConstantFoldingCacheManager::getInstance();
        auto& cache = cacheManager.get(_ctx);

        while (true) {
            auto foldingRequest = cache.getRequest();
            if (mlir::isa_and_nonnull<Const::TerminateRequestAttr>(foldingRequest.attr)) {
                break;
            }

            processFoldingRequest(foldingRequest, cache);
        }
    });

    return listenerThread;
}

void BackgroundConstantFolding::stopFoldingListener() {
    auto& cacheManager = ConstantFoldingCacheManager::getInstance();
    auto& cache = cacheManager.get(_ctx);
    auto terminationAttr = Const::TerminateRequestAttr::get(_ctx);
    cache.enqueueRequest(Const::FoldingRequest{terminationAttr, nullptr});

    _listenerThread.wait();

    if (cache.isStatisticsCollectionEnabled()) {
        _log.setName("constant-folding-in-background");
        auto& statistics = cacheManager.get(_ctx).getStatistics();
        _log.info("Cache statistics");
        _log.nest().info("number of cache hits:                       {0}", statistics.numCacheHits);
        _log.nest().info("number of cache misses:                     {0}", statistics.numCacheMisses);
        _log.nest().info("maximum number of requests in queue:        {0}", statistics.getMaxNumRequestsInQueue());
        _log.nest().info("maximum number of elements in cache:        {0}", statistics.getMaxCacheSize());
        _log.nest().info("current memory used by cache:               {0}", statistics.memoryUsedCache);
        _log.nest().info("maximum memory used by cache:               {0}", statistics.getMaxMemoryUsedCache());
        _log.nest().info("number of duplicated requests:              {0}", statistics.numDuplicatedRequests);
        _log.nest().info("total number of elements added to cache:    {0}", statistics.numElementsAddedToCache);
        _log.nest().info("total number of elements erased from cache: {0}", statistics.numElementsErasedFromCache);
    }

    cacheManager.removeCache(_ctx);
}

#endif
