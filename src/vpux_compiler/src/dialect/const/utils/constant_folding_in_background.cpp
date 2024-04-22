//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/utils/constant_folding_in_background.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"

using namespace vpux;

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

    cache.setRequestState(request, Const::FoldingRequestState::PENDING);

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

    cache.setRequestState(request, Const::FoldingRequestState::COMPLETED);

    return true;
}

std::shared_future<void> initFoldingThread(mlir::MLIRContext* ctx) {
    auto& threadPool = ctx->getThreadPool();
    auto constantFoldingThread = threadPool.async([ctx]() {
        auto& cacheManager = Const::ConstantFoldingCacheManager::getInstance();
        auto& cache = cacheManager.get(ctx);

        while (true) {
            auto foldingRequest = cache.getRequest();
            if (foldingRequest.attr.isa_and_nonnull<Const::TerminateRequestAttr>()) {
                break;
            }

            auto request = foldingRequest.attr.dyn_cast_or_null<Const::ContentAttr>();
            VPUX_THROW_WHEN(request == nullptr, "Invalid folding request");

            // Skip requests that have been received before. These requests would have their state set as either PENDING
            // or COMPLETED after they are picked for processing
            if (cache.getRequestState(request).has_value()) {
                if (cache.isStatisticsCollectionEnabled()) {
                    cache.getStatistics().numDuplicatedRequests++;
                }
                continue;
            }

            // Try folding partially if the new transformation is added to the end of the list of transformations
            // In this case, it is likely that the previous ContentAttr (without the new transformation) is already in
            // the cached, so its folded result can be reused
            if (tryFoldingPartially(cache, request, foldingRequest.newTransformation)) {
                continue;
            }

            cache.setRequestState(request, Const::FoldingRequestState::PENDING);

            // Create a copy of the Content which will own the referenced buffer
            // This is done since the Content object obtained after folding may reference an external object without
            // owning it. If that object is erased, the Content object from the cache would point to an invalid object
            cache.addContent(request, request.fold(/*bypassCache=*/true).copyUnownedBuffer());

            cache.setRequestState(request, Const::FoldingRequestState::COMPLETED);
        }
    });

    return constantFoldingThread;
}

SmallVector<std::shared_future<void>> Const::initBackgroundConstantFoldingThreads(mlir::MLIRContext* ctx,
                                                                                  size_t numThreads,
                                                                                  bool collectStatistics) {
    auto& cacheManager = Const::ConstantFoldingCacheManager::getInstance();
    cacheManager.addCache(ctx);
    if (collectStatistics) {
        cacheManager.get(ctx).enableStatisticsCollection();
    }

    SmallVector<std::shared_future<void>> foldingThreads(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        foldingThreads[i] = initFoldingThread(ctx);
    }
    return foldingThreads;
}

void Const::stopBackgroundConstantFoldingThreads(mlir::MLIRContext* ctx,
                                                 ArrayRef<std::shared_future<void>> foldingThreads,
                                                 bool collectStatistics, Logger log) {
    auto& cacheManager = Const::ConstantFoldingCacheManager::getInstance();
    auto& cache = cacheManager.get(ctx);

    // Add termination requests to the queue, one for each folding thread
    auto terminationAttr = Const::TerminateRequestAttr::get(ctx);
    for (size_t i = 0; i < foldingThreads.size(); ++i) {
        cache.enqueueRequest(Const::FoldingRequest{terminationAttr, nullptr});
    }

    for (auto& foldingThread : foldingThreads) {
        foldingThread.wait();
    }

    if (collectStatistics) {
        log.setName("constant-folding-in-background");
        auto& statistics = cacheManager.get(ctx).getStatistics();
        log.trace("Cache statistics");
        log.nest().trace("number of cache hits:                       {0}", statistics.numCacheHits);
        log.nest().trace("number of cache misses:                     {0}", statistics.numCacheMisses);
        log.nest().trace("maximum number of requests in queue:        {0}", statistics.getMaxNumRequestsInQueue());
        log.nest().trace("maximum number of elements in cache:        {0}", statistics.getMaxCacheSize());
        log.nest().trace("current memory used by cache:               {0}", statistics.memoryUsedCache);
        log.nest().trace("maximum memory used by cache:               {0}", statistics.getMaxMemoryUsedCache());
        log.nest().trace("number of duplicated requests:              {0}", statistics.numDuplicatedRequests);
        log.nest().trace("total number of elements added to cache:    {0}", statistics.numElementsAddedToCache);
        log.nest().trace("total number of elements erased from cache: {0}", statistics.numElementsErasedFromCache);
    }

    cacheManager.removeCache(ctx);
}

#endif
