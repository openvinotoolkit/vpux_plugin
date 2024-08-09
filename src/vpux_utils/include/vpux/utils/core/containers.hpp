//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Various helper functions for STL containers.
//

#pragma once

#include <utility>

namespace vpux {

//
// addToContainer - generic way to add element to the container.
//

namespace details {

template <class Container, typename T>
auto addToContainerImpl(Container& cont, T&& val, int) -> decltype(cont.push_back(std::forward<T>(val))) {
    return cont.push_back(std::forward<T>(val));
}

template <class Container, typename T>
auto addToContainerImpl(Container& cont, T&& val, ...) -> decltype(cont.insert(std::forward<T>(val))) {
    return cont.insert(std::forward<T>(val));
}

}  // namespace details

template <class Container, typename T>
void addToContainer(Container& cont, T&& val) {
    details::addToContainerImpl(cont, std::forward<T>(val), 0);
}

}  // namespace vpux
