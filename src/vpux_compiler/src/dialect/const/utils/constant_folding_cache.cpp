//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifdef BACKGROUND_FOLDING_ENABLED

#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"

using namespace vpux;

//
// ConstantFoldingCache
//

void Const::ConstantFoldingCache::enqueueRequest(const Const::FoldingRequest& foldingRequest) {
    _requestQueue.push(foldingRequest);
    if (_collectStatistics) {
        _statistics.updateMaxNumRequestsInQueue(
                std::max(static_cast<int64_t>(0), checked_cast<int64_t>(_requestQueue.size())));
    }
}

Const::FoldingRequest Const::ConstantFoldingCache::getRequest() {
    Const::FoldingRequest result;
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
    _requestQueue.pop(result);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    return result;
}

bool Const::ConstantFoldingCache::hasContent(Const::ContentAttr attr) {
    Const::details::ContentMap::const_accessor accessor;
    return _cache.find(accessor, attr) && !accessor.empty();
}

void Const::ConstantFoldingCache::addContent(Const::ContentAttr attr, const Const::Content& content) {
    Const::details::ContentMap::accessor accessor;
    _cache.insert(accessor, attr);
    VPUX_THROW_WHEN(accessor.empty(), "Failed to add folding request to cache");
    accessor->second = content;
    accessor.release();

    if (_collectStatistics) {
        _statistics.numElementsAddedToCache++;
        auto size = attr.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize();
        _statistics.memoryUsedCache += size.count();

        _statistics.updateMaxMemoryUsedCache(_statistics.memoryUsedCache.load());
        _statistics.updateMaxCacheSize(_cache.size());
    }
}

void Const::ConstantFoldingCache::removeContent(Const::ContentAttr attr) {
    Const::details::ContentMap::accessor accessor;
    if (_cache.find(accessor, attr) && !accessor.empty()) {
        _cache.erase(accessor);
        accessor.release();

        if (_collectStatistics) {
            _statistics.numElementsErasedFromCache++;

            auto size = attr.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize();
            _statistics.memoryUsedCache -= size.count();
        }
    }
}

std::optional<Const::Content> Const::ConstantFoldingCache::getContent(Const::ContentAttr attr) {
    // Return the value from the cache if it contains it
    Const::details::ContentMap::const_accessor accessor;
    if (_cache.find(accessor, attr) && !accessor.empty()) {
        if (_collectStatistics) {
            _statistics.numCacheHits++;
        }
        return accessor->second;
    }

    if (_collectStatistics) {
        _statistics.numCacheMisses++;
    }
    return std::nullopt;
}

bool Const::ConstantFoldingCache::replaceContentAttr(Const::ContentAttr originalAttr, Const::ContentAttr newAttr) {
    Const::details::ContentMap::accessor originalAttrAccessor;
    if (_cache.find(originalAttrAccessor, originalAttr) && !originalAttrAccessor.empty()) {
        Const::details::ContentMap::accessor newAttrAccessor;
        _cache.insert(newAttrAccessor, newAttr);
        VPUX_THROW_WHEN(newAttrAccessor.empty(), "Failed to add folding request to cache");
        newAttrAccessor->second = std::move(originalAttrAccessor->second);
        _cache.erase(originalAttrAccessor);
        return true;
    }
    return false;
}

void Const::ConstantFoldingCache::enableStatisticsCollection() {
    _collectStatistics = true;
}

void Const::ConstantFoldingCache::disableStatisticsCollection() {
    _collectStatistics = false;
}

bool Const::ConstantFoldingCache::isStatisticsCollectionEnabled() {
    return _collectStatistics;
}

Const::details::CacheStatistics& Const::ConstantFoldingCache::getStatistics() {
    return _statistics;
}

//
// CacheStatistics
//

void Const::details::CacheStatistics::updateMaxNumRequestsInQueue(size_t newNumRequests) {
    std::lock_guard<std::mutex> lock(_mtx);
    _maxNumRequestsInQueue = std::max(_maxNumRequestsInQueue, newNumRequests);
}

void Const::details::CacheStatistics::updateMaxCacheSize(size_t newCacheSize) {
    std::lock_guard<std::mutex> lock(_mtx);
    _maxCacheSize = std::max(_maxCacheSize, newCacheSize);
}

void Const::details::CacheStatistics::updateMaxMemoryUsedCache(size_t newMemoryUsedCache) {
    std::lock_guard<std::mutex> lock(_mtx);
    _maxMemoryUsedCache = std::max(_maxMemoryUsedCache, newMemoryUsedCache);
}

size_t Const::details::CacheStatistics::getMaxNumRequestsInQueue() {
    std::lock_guard<std::mutex> lock(_mtx);
    return _maxNumRequestsInQueue;
}

size_t Const::details::CacheStatistics::getMaxCacheSize() {
    std::lock_guard<std::mutex> lock(_mtx);
    return _maxCacheSize;
}

size_t Const::details::CacheStatistics::getMaxMemoryUsedCache() {
    std::lock_guard<std::mutex> lock(_mtx);
    return _maxMemoryUsedCache;
}

//
// ConstantFoldingCacheManager
//

Const::ConstantFoldingCacheManager& Const::ConstantFoldingCacheManager::getInstance() {
    static Const::ConstantFoldingCacheManager instance;
    return instance;
}

bool Const::ConstantFoldingCacheManager::addCache(mlir::MLIRContext* ctx) {
    std::lock_guard<std::mutex> lock(_mtx);
    if (contains(ctx)) {
        return false;
    }
    // Create the cache object for the given context
    _caches[ctx];
    return true;
}

bool Const::ConstantFoldingCacheManager::removeCache(mlir::MLIRContext* ctx) {
    std::lock_guard<std::mutex> lock(_mtx);
    if (auto it = _caches.find(ctx); it != _caches.end()) {
        _caches.erase(it);
        return true;
    }
    return false;
}

bool Const::ConstantFoldingCacheManager::contains(mlir::MLIRContext* ctx) {
    return _caches.find(ctx) != _caches.end();
}

Const::ConstantFoldingCache& Const::ConstantFoldingCacheManager::get(mlir::MLIRContext* ctx) {
    auto it = _caches.find(ctx);
    if (it != _caches.end()) {
        return it->second;
    }
    VPUX_THROW("Unable to find cache for {0}", ctx);
}

#endif
