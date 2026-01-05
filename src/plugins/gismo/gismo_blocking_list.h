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

#ifndef __GISMO_BLOCKING_LIST_H
#define __GISMO_BLOCKING_LIST_H

#include <list>
#include <mutex>
#include <vector>
#include <optional>
#include <shared_mutex>

template<typename _T> class gismoBlockingList {
public:
    inline gismoBlockingList() {};
    inline ~gismoBlockingList() {};

    inline std::optional<_T>
    pop_front() {
        std::unique_lock guard(mtx_);
        if (values_.size() == 0) {
            return std::nullopt;
        }
        auto t = values_.front();
        values_.pop_front();
        return std::optional<_T>(t);
    }

    inline bool
    contains(_T k) {
        std::shared_lock guard(mtx_);
        for (auto &itr : values_) {
            if (itr == k) {
                return true;
            }
        }
        return false;
    }

    inline void
    push_back(_T k) {
        std::unique_lock guard(mtx_);
        values_.emplace_back(k);
    }

    inline void
    push_back(const std::vector<_T> &k) {
        std::unique_lock guard(mtx_);
        for (auto &i : k) {
            values_.emplace_back(i);
        }
    }

    inline void
    remove(_T k) {
        std::unique_lock guard(mtx_);
        for (auto it = values_.begin(); it != values_.end();) {
            if (k == *it) {
                it = values_.erase(it);
            } else {
                ++it;
            }
        }
    }

    inline void
    toVector(std::vector<_T> &result) {
        std::shared_lock guard(mtx_);
        for (auto &itr : values_) {
            result.emplace_back(itr);
        }
    }

    inline void
    move(std::vector<_T> &result) {
        std::unique_lock guard(mtx_);
        for (auto itr = values_.begin(); itr != values_.end();) {
            result.emplace_back(*itr);
            itr = values_.erase(itr);
        }
    }

    inline size_t
    size() {
        std::shared_lock guard(mtx_);
        return values_.size();
    }

    inline bool
    empty() {
        std::shared_lock guard(mtx_);
        return values_.empty();
    }

private:
    // underlying list
    std::list<_T> values_;
    // synchronization
    std::shared_mutex mtx_;
};

#endif //__GISMO_BLOCKING_LIST_H