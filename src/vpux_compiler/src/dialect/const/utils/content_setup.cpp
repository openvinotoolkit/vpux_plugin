//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"

vpux::Const::ContentSetup::ContentSetup(mlir::ElementsAttr baseContent,
                                        ArrayRef<TransformAttrInterface> transformations)
        : _baseContent(baseContent), _transformations(transformations) {
    VPUX_THROW_WHEN(_baseContent == nullptr, "baseContent must not be an empty attribute");
}

vpux::Const::ContentSetup::ContentSetup(ContentSetup&& other)
        : _baseContent(std::exchange(other._baseContent, nullptr)),
          _transformations(std::move(other._transformations)) {
}

vpux::Const::ContentSetup& vpux::Const::ContentSetup::operator=(ContentSetup&& other) {
    ContentSetup tmp(std::move(other));
    // avoids calling move assignment operator when using std::swap(*this, tmp)
    std::swap(_baseContent, tmp._baseContent);
    std::swap(_transformations, tmp._transformations);
    return *this;
}

namespace {
void enqueueCache(mlir::MLIRContext* ctx, const vpux::Const::ContentAttr& attr) {
#ifdef BACKGROUND_FOLDING_ENABLED
    auto& cacheManager = vpux::Const::ConstantFoldingCacheManager::getInstance();
    if (cacheManager.contains(ctx)) {
        auto& cache = cacheManager.get(ctx);
        cache.enqueueRequest(vpux::Const::FoldingRequest{attr, nullptr});
    }
#else
    std::ignore = ctx;
    std::ignore = attr;
#endif
}
}  // namespace

vpux::Const::ContentAttr vpux::Const::ContentSetup::get() {
    checkInvalidated();
    auto resultAttr = ContentAttr::get(_baseContent, _transformations);
    enqueueCache(getContext(), resultAttr);
    return resultAttr;
}

void vpux::Const::ContentSetup::checkInvalidated() const {
    VPUX_THROW_WHEN(isInvalidated(),
                    "ContentSetup was marked invalidated because it was moved. Did you forget to call clone()?");
}
