/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __GISMO_CONCURRENT_MAP_H
#define __GISMO_CONCURRENT_MAP_H

#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <optional>
#include <functional>

template<typename _Key, typename _Value> class gismoConcurrentMap {
public:
    inline gismoConcurrentMap() {};
    inline ~gismoConcurrentMap() {};

    inline std::optional<_Value>
    get(_Key k) const {
        std::shared_lock guard(mtx_);
        auto itr = values_.find(k);
        if (itr != values_.end()) {
            return std::optional<_Value>(itr->second);
        }
        return std::nullopt;
    }

    inline _Value
    getAndPut(_Key k, const std::function<_Value(_Key &)> &creator) {
        // try get firstly
        auto ret = get(k);
        if (ret.has_value()) {
            return ret.value();
        }
        // double check with write lock
        std::unique_lock guard(mtx_);
        auto itr = values_.find(k);
        if (itr != values_.end()) {
            return itr->second;
        }
        // still failed, call creator
        auto v = creator(k);
        values_[k] = v;
        return v;
    }

    inline _Value
    update(_Key k, const std::function<void(_Value &)> &updater) {
        std::unique_lock guard(mtx_);
        auto itr = values_.find(k);
        auto v = _Value();
        if (itr != values_.end()) {
            v = itr->second;
        }
        updater(v);
        values_[k] = v;
        return v;
    }

    inline void
    removeIf(_Key k, const std::function<bool(_Value &)> &checker) {
        std::unique_lock guard(mtx_);
        auto itr = values_.find(k);
        if (itr != values_.end() && checker(itr->second)) {
            values_.erase(itr);
        }
    }

    inline int
    countIf(const std::function<bool(_Value &)> &checker) {
        int size = 0;
        std::shared_lock guard(mtx_);
        for (auto itr = values_.begin(); itr != values_.end(); ++itr) {
            if (checker(itr->second)) {
                ++size;
            }
        }
        return size;
    }

    inline bool
    contains(_Key k) {
        std::shared_lock guard(mtx_);
        auto itr = values_.find(k);
        return (itr != values_.end());
    }

    inline void
    put(_Key k, _Value v) {
        std::unique_lock guard(mtx_);
        values_[k] = v;
    }

    inline void
    remove(_Key k) {
        std::unique_lock guard(mtx_);
        auto itr = values_.find(k);
        if (itr != values_.end()) {
            values_.erase(itr);
        }
    }

    inline void
    toVector(std::vector<std::pair<_Key, _Value>> &result) {
        std::shared_lock guard(mtx_);
        for (auto &itr : values_) {
            result.push_back(std::pair<_Key, _Value>(itr.first, itr.second));
        }
    }

    inline void
    keys(std::vector<_Key> &result) {
        std::shared_lock guard(mtx_);
        for (auto &itr : values_) {
            result.push_back(itr.first);
        }
    }

    inline size_t
    size() {
        std::shared_lock guard(mtx_);
        return values_.size();
    }

private:
    // underlying map
    std::unordered_map<_Key, _Value> values_;
    // synchronization
    mutable std::shared_mutex mtx_;
};


#endif //__GISMO_CONCURRENT_MAP_H