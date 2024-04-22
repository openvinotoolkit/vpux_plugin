//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/content.hpp"

#include <mlir/IR/MLIRContext.h>

#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_queue.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <utility>

namespace vpux {
namespace Const {

//
// FoldingRequest
//
// Contains two elements:
// - attr: Attribute representing the ContentAttr which should be folded by a background thread
// - newTransformation: The new transformation which was last added to the list of transformations of `attr`.
//                      This is used internally to recreate the ContentAttr that has all the transformations up to
//                      `newTransformation`, in order to reuse the existing folded values from the cache. This value is
//                      optional, in which case `nullptr` can be used.
struct FoldingRequest {
    Const::RequestQueueAttrInterface attr;
    Const::TransformAttrInterface newTransformation;
};

//
// FoldingRequestState
//
// State PENDING means that the request has been received by a thread and is currently being folded, while COMPLETED
// means that the folding has been done and the result has been added to the cache

enum class FoldingRequestState { PENDING, COMPLETED };

//
// details
//

namespace details {

struct CacheContent {
    FoldingRequestState state = FoldingRequestState::PENDING;
    Const::Content content;
};

struct ContentAttrHash {
    static size_t hash(Const::ContentAttr attr) {
        auto hashValue = mlir::hash_value(attr);
        return static_cast<size_t>(hashValue);
    }
    static bool equal(Const::ContentAttr lhs, Const::ContentAttr rhs) {
        return lhs == rhs;
    }
};

using RequestQueue = tbb::concurrent_bounded_queue<FoldingRequest>;
using ContentMap = tbb::concurrent_hash_map<Const::ContentAttr, CacheContent, ContentAttrHash>;

struct CacheStatistics {
    std::atomic<size_t> memoryUsedCache = 0;
    std::atomic<size_t> numElementsAddedToCache = 0;
    std::atomic<size_t> numElementsErasedFromCache = 0;
    std::atomic<size_t> numCacheHits = 0;
    std::atomic<size_t> numCacheMisses = 0;
    std::atomic<size_t> numDuplicatedRequests = 0;

    void updateMaxNumRequestsInQueue(size_t newNumRequests);
    void updateMaxCacheSize(size_t newCacheSize);
    void updateMaxMemoryUsedCache(size_t newMemoryUsedCache);

    size_t getMaxNumRequestsInQueue();
    size_t getMaxCacheSize();
    size_t getMaxMemoryUsedCache();

private:
    size_t _maxNumRequestsInQueue = 0;
    size_t _maxCacheSize = 0;
    size_t _maxMemoryUsedCache = 0;

    std::mutex _mtx{};
};

}  // namespace details

//
// ConstantFoldingCache
//

class ConstantFoldingCache {
public:
    /**
     * @brief Add a request to the queue for folding in background. The folding request contains the ContentAttr that
     * should be folded in background and optionally the new transformation that was added in the ContentAttr
     * @details This method is thread-safe
     * @param `foldingRequest`: request to be folded
     */
    void enqueueRequest(const Const::FoldingRequest& foldingRequest);

    /**
     * @brief Gets the folding request found at the top of the queue or waits until one becomes available. When found,
     * the element is removed from the queue
     * @details This method is thread-safe
     * @return The folding request at the top of the queue
     */
    FoldingRequest getRequest();

    /**
     * @brief Sets the state of a folding request
     * @details This method is thread-safe
     * @param `attr`: the folding request whose state should be set
     * @param `state`: the new state to be set for the request
     */
    void setRequestState(Const::ContentAttr attr, const Const::FoldingRequestState& state);

    /**
     * @brief Tries to get the state of a folding request, if the request has been received
     * @details This method is thread-safe
     * @return The state of the request if it exists or an empty optional otherwise
     */
    std::optional<Const::FoldingRequestState> getRequestState(Const::ContentAttr attr);

    /**
     * @brief Adds a folding result to the cache
     * @details This method is thread-safe
     * @param `attr`: the folding request whose folding result should be added to the cache
     * @param `content`: the folding result
     */
    void addContent(Const::ContentAttr attr, const Const::Content& content);

    /**
     * @brief Removes a folding result from the cache
     * @details This method is thread-safe
     * @param `attr`: the folding request whose folding result should be removed from the cache
     */
    void removeContent(Const::ContentAttr attr);

    /**
     * @brief Tries to get the folding result from the cache for the given request (represented as an attribute). In
     * case the request is marked as PENDING, the function will wait until it is marked as COMPLETED and then the value
     * is returned. In case the request has no state set, the function will return nothing
     * @details This method is thread-safe
     * @param `attr`: the folding request whose folding result should be obtained from the cache
     * @return The folding result if it has been found or an empty optional otherwise
     */
    std::optional<Const::Content> getContent(Const::ContentAttr attr);

    /**
     * @brief Enable the collection of statistics for the cache
     * @details This method is NOT thread-safe
     */
    void enableStatisticsCollection();

    /**
     * @brief Disable the collection of statistics for the cache
     * @details This method is NOT thread-safe
     */
    void disableStatisticsCollection();

    /**
     * @brief Get the status on whether cache statistics are being collected
     * @details This method is NOT thread-safe
     * @return True if statistics are being collected
     */
    bool isStatisticsCollectionEnabled();

    /**
     * @brief Get the cache statistics collected from the creation of this cache object
     * @details This method is NOT thread-safe
     * @return The statistics
     */
    Const::details::CacheStatistics& getStatistics();

private:
    Const::details::RequestQueue _requestQueue{};
    Const::details::ContentMap _cache{};

    bool _collectStatistics = false;
    Const::details::CacheStatistics _statistics{};

    std::mutex _mtxState{};
    std::condition_variable _cvState{};
};

//
// ConstantFoldingCacheManager
//

class ConstantFoldingCacheManager {
public:
    /**
     * @brief Get the unique instance of the constant folding cache manager
     * @details This method is thread-safe
     * @return The instance of the constant folding cache manager
     */
    static ConstantFoldingCacheManager& getInstance();

    /**
     * @brief Creates a cache object for the given MLIRContext, if one does not already exist
     * @details This method is thread-safe
     * @param `ctx`: the MLIRContext associated with the cache
     * @return True if a new cache object has been created
     */
    bool addCache(mlir::MLIRContext* ctx);

    /**
     * @brief Removes the cache object of the given MLIRContext, if it exists
     * @details This method is thread-safe
     * @param `ctx`: the MLIRContext associated with the cache
     * @return True if a cache object was found and removed, false otherwise
     */
    bool removeCache(mlir::MLIRContext* ctx);

    /**
     * @brief Checks whether a cache object exists for the given MLIRContext
     * @details This method is thread-safe
     * @param `ctx`: the MLIRContext associated with a cache
     * @return True if a cache object was found, false otherwise
     */
    bool contains(mlir::MLIRContext* ctx);

    /**
     * @brief Returns the cache object associated with the given MLIRContext. In case no cache exists for the given
     * context, an error is thrown
     * @details This method is thread-safe
     * @param `ctx`: the MLIRContext associated with the cache
     * @return A reference to the cache object
     */
    Const::ConstantFoldingCache& get(mlir::MLIRContext* ctx);

private:
    ConstantFoldingCacheManager() = default;
    ~ConstantFoldingCacheManager() = default;
    ConstantFoldingCacheManager(const ConstantFoldingCacheManager&) = delete;
    ConstantFoldingCacheManager(ConstantFoldingCacheManager&&) = delete;
    ConstantFoldingCacheManager operator=(const ConstantFoldingCacheManager&) = delete;
    ConstantFoldingCacheManager operator=(ConstantFoldingCacheManager&&) = delete;

private:
    std::unordered_map<mlir::MLIRContext*, Const::ConstantFoldingCache> _caches;

    std::mutex _mtx;
};

}  // namespace Const
}  // namespace vpux

#endif
