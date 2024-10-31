//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/content.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <mlir/IR/MLIRContext.h>

#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_queue.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <variant>

namespace vpux {
namespace Const {

// Marks two ContentAttr objects as being equivalent for background folding.
//
// This attribute is meant to be used in the request queue for the background
// folding feature. It shows that two ContentAttr objects are equivalent in
// terms of folding, meaning that the `fold()` method will produce the same
// result for both attributes. The main use-case for this attribute is to allow
// the list of transformations of a ContentAttr to be optimized (e.g. produce
// the same result but with fewer transformations or less compute per
// transformation), without losing the computation that was already performed.
struct EquivalenceRequest {
    Const::ContentAttr originalAttr = {};
    Const::ContentAttr newAttr = {};
};

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
    using TerminationToken = std::monostate;
    std::variant<TerminationToken, Const::ContentAttr, Const::EquivalenceRequest> attr = {};
    Const::TransformAttrInterface newTransformation = nullptr;
};

//
// details
//

namespace details {

struct ContentAttrHashCode : llvm::hash_code {
    vpux::Byte totalAllocSize = {};  // Note: used for memory usage statistics.

    ContentAttrHashCode(const Const::ContentAttr& attr)
            : llvm::hash_code(hash_value(attr)),
              totalAllocSize(mlir::cast<NDTypeInterface>(attr.getType()).getTotalAllocSize()) {
    }
};

struct IdentityHashCompare {
    static size_t hash(const ContentAttrHashCode& hash) {
        return static_cast<size_t>(hash);
    }
    static bool equal(const ContentAttrHashCode& x, const ContentAttrHashCode& y) {
        return x == y;
    }
};

//
// CachedContent
//
// Contains two elements:
// - content: Represents the folded content
// - refCount: An integer that represents the number of times the content has been queried minus the number of times it
// has been retrieved. Based on observation, the more times the content is queried, the more likely it is going to be
// used in the future. Conversely, the more times the content is retrieved, the less likely it is going to be used in
// the future. Therefore, a smaller refCount indicates a higher priority for removing the cache entry if the memory
// limit is reached. This is a temporary solution for the basic cache clean mechanism and will be further optimized in
// the future.
struct CachedContent {
    Const::Content content;
    std::atomic<int> refCount{1};
};

using RequestQueue = tbb::concurrent_bounded_queue<FoldingRequest>;
using ContentMap = tbb::concurrent_hash_map<ContentAttrHashCode, CachedContent, IdentityHashCompare>;

struct CacheStatistics {
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
     * @brief Checks whether the given attribute is found in the cache
     * @details This method is thread-safe
     * @return true if the attribute has been found
     */
    bool hasContent(Const::ContentAttr attr);

    /**
     * @brief Adds a folding result to the cache
     * @details This method is thread-safe
     * @param `attr`: the folding request whose folding result should be added to the cache
     * @param `content`: the folding result
     */
    void addContent(Const::details::ContentAttrHashCode attr, Const::Content&& content);

    /**
     * @brief Sets the memory usage limit for the cache
     * @details This method is not thread-safe but assumed not to be used in contexts
     * where multi-threading scenarios are involved
     * @param `memoryUsageLimit`: the memory usage limit in bytes
     */
    void setMemoryUsageLimit(vpux::Byte memoryUsageLimit);

    /**
     * @brief Sets the cache clean threshold
     * @details This method is not thread-safe but assumed not to be used in contexts
     * where multi-threading scenarios are involved
     * @param `cacheCleanThreshold`: the cache clean ratio
     */
    void setCacheCleanThreshold(double cacheCleanThreshold);

    /**
     * @brief Gets the memory used by the cache
     * @details This method is not thread-safe but assumed not to be used in contexts
     * where multi-threading scenarios are involved
     */
    size_t getMemoryUsedCache() const;

    /**
     * @brief Clean the cache to cacheCleanThreshold based on the refCount
     * @details This method is thread-safe
     */
    void cleanUpCache();

    /**
     * @brief Checks if the cache memory consumption has reached the limit
     * @details This method is not thread-safe but assumed not to be used in contexts
     * where multi-threading scenarios are involved
     */
    bool isMemoryLimitReached() const;

    /**
     * @brief Removes a folding result from the cache
     * @details This method is thread-safe
     * @param `attr`: the folding request whose folding result should be removed from the cache
     */
    void removeContent(Const::details::ContentAttrHashCode attr);

    /**
     * @brief Tries to get the folding result from the cache for the given request (represented as an attribute). In
     * case the cache does not contain the result, the function will return nothing
     * @details This method is thread-safe
     * @param `attr`: the folding request whose folding result should be obtained from the cache
     * @return The folding result if it has been found or an empty optional otherwise
     */
    std::optional<Const::Content> getContent(Const::ContentAttr attr);

    /**
     * @brief Replaces the original attribute from the cache with a new one, while keeping the same folded content.
     * In case the original attribute is not found in the cache, nothing is done and false is returned
     * @details This method is thread-safe
     * @param `originalAttr`: the attribute from the cache that should be replaced
     * @param `newAttr`: the attribute which should replace the original one
     * @return true if the replacement has been done successfully
     */
    bool replaceContentAttr(Const::ContentAttr originalAttr, Const::ContentAttr newAttr);

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

    std::mutex _mutex;
    bool _collectStatistics = false;
    size_t _memoryUsageLimit = 0;
    double _cacheCleanThreshold = 0.8;
    std::atomic<size_t> _memoryUsedCache = 0;
    Const::details::CacheStatistics _statistics{};
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
